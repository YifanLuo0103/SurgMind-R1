# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import Counter

from io import BytesIO
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Image, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image as PILImage


def _extract_tag(text: str, tag: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _normalize_answer(text: str) -> str:
    """
    Normalize for robust matching across English/Chinese:
    - lowercase
    - remove surrounding whitespace
    - remove punctuation and whitespace (keep alnum + CJK)
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # keep: a-z, 0-9, and basic CJK range; drop other chars (punctuation/whitespace)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    return text


def _f1_similarity(pred: str, gold: str) -> float:
    """
    Token-level F1 if whitespace tokens exist, otherwise char-level F1 (works for Chinese).
    """
    if not pred or not gold:
        return 0.0
    # Prefer whitespace tokens when both look tokenizable
    if (" " in pred) or (" " in gold) or ("\n" in pred) or ("\n" in gold) or ("\t" in pred) or ("\t" in gold):
        pred_tokens = [t for t in re.split(r"\s+", pred.strip()) if t]
        gold_tokens = [t for t in re.split(r"\s+", gold.strip()) if t]
    else:
        pred_tokens = list(pred)
        gold_tokens = list(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / max(1, len(pred_tokens))
    recall = common / max(1, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def openqa_reward(completions, solution, image_ids, problems, **kwargs):
    """
    Reward for open-ended QA.

    - Extract <answer>...</answer> from both prediction and ground-truth solution
    - Reward = 1.0 for exact match after normalization
    - Else reward = F1 similarity in [0, 1]
    """
    rewards = []
    for comp, sol, image_id, problem in zip(completions, solution, image_ids, problems):
        content = comp[0]["content"]

        gt_answer_raw = _extract_tag(sol, "answer") or (sol.strip() if isinstance(sol, str) else "")
        pred_answer_raw = _extract_tag(content, "answer")

        gt_norm = _normalize_answer(gt_answer_raw)
        pred_norm = _normalize_answer(pred_answer_raw)

        if not pred_norm:
            reward = 0.0
        elif pred_norm == gt_norm and gt_norm:
            reward = 1.0
        else:
            reward = _f1_similarity(pred_norm, gt_norm)

        rewards.append(float(reward))

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} OpenQA reward: {reward:.4f} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Problem: {problem}\n\n")
                f.write(f"Pred content: {content}\n")
                f.write(f"GT solution: {sol}\n")
                f.write(f"Pred <answer>: {pred_answer_raw}\n")
                f.write(f"GT <answer>: {gt_answer_raw}\n")
    return rewards



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
        dataset_selection (`str`):
            Dataset selection for medical imaging. Possible values: 'MR', 'CT', 'XRAY', 'COMBINED'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    dataset_selection: str = field(
        default="MR",
        metadata={"help": "Dataset selection for medical imaging. Possible values: 'MR', 'CT', 'XRAY', 'COMBINED'"},
    )


def accuracy_reward(completions, solution, image_ids, problems, **kwargs):
    """
    Reward function for multiple-choice answers:
      - 1.0 if exactly one occurrence of the correct letter (A–J) and no other text.
      - 0.5 if exactly one occurrence of the correct letter plus extra text.
      - 0.0 otherwise (no letters, wrong letter, multiple letters, repeated letters, etc).
    """
    rewards = []

    for comp, sol, image_id, problem in zip(completions, solution, image_ids, problems):
        content = comp[0]["content"]  # typical structure: [{ "role":"assistant", "content": "..."}]
        reward = 0.0

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        # Extract ground truth letter
        sol_match = re.search(r"<answer>(.*?)</answer>", sol, flags=re.DOTALL|re.IGNORECASE)
        if sol_match:
            ground_truth = sol_match.group(1).strip()
        else:
            ground_truth = sol.strip()

        # Extract student's answer
        content_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL|re.IGNORECASE)
        if content_match:
            student_answer = content_match.group(1).strip()
        else:
            student_answer = ""  # no <answer> => 0.0

        # Prepare comparisons
        gt_letter = re.sub(r"\s+", "", ground_truth.upper())
        sa_str = student_answer.strip()  
        sa_upper = sa_str.upper()

        if not sa_upper:
            reward = 0.0
        else:
            letters_found = re.findall(r"[A-J]", sa_upper)
            distinct_letters = set(letter.upper() for letter in letters_found)

            if len(letters_found) == 1 and len(distinct_letters) == 1:
                found_letter = distinct_letters.pop()
                if found_letter == gt_letter:
                    # Check leftover text
                    leftover = re.sub(r"(?i)" + found_letter, "", sa_str, count=1)
                    if leftover.strip():
                        # Has extra text
                        reward = 0.5
                    else:
                        # Perfect single-letter match
                        reward = 1.0
                else:
                    # Single letter but not correct
                    reward = 0.0
            else:
                # no letters, multiple letters, repeated letter, etc.
                reward = 0.0
        
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Problem: {problem}\n\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, image_ids=None, problems=None, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]

    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]

    # Optional debug logging (align with other reward funcs)
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        # image_ids / problems are passed by Qwen2VLGRPOTrainer when available
        for i, (content, r) in enumerate(zip(completion_contents, rewards)):
            img_id = image_ids[i] if isinstance(image_ids, list) and i < len(image_ids) else ""
            prob = problems[i] if isinstance(problems, list) and i < len(problems) else ""
            with open(log_path, "a") as f:
                f.write(f"--- {current_time} Format reward: {r:.1f} ---\n")
                if img_id:
                    f.write(f"Image ID: {img_id}\n")
                if prob:
                    f.write(f"Problem: {prob}\n\n")
                f.write(f"Pred content: {content}\n\n")

    return rewards


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────

def _set_f1(pred_set: set, gold_set: set) -> float:
    """Compute set-level F1 for tools, triplets, or components."""
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall    = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _parse_tool_phase(answer: str):
    """
    Parse '[grasper, hook]; [Calot Triangle Dissection]'.
    Return the lowercased tool set and phase string.
    """
    answer = answer.strip()

    parts = [p.strip().strip("[]").strip() for p in answer.split(";")]
    if len(parts) < 2:

        tools_str = parts[0] if parts else ""
        tools = {t.strip().lower() for t in tools_str.split(",") if t.strip() and t.strip().lower() != "null"}
        return tools, ""
    tools_str = parts[0]
    tools = {t.strip().lower() for t in tools_str.split(",") if t.strip() and t.strip().lower() != "null"}
    phase = ";".join(parts[1:]).strip().lower()
    return tools, phase


def _parse_triplets(answer: str):
    """
    Parse semicolon-separated [instrument, verb, target] triplets.
    Return lowercased triplet, instrument, verb, and target sets.
    Exclude triplets containing a 'null' component.
    """
    triplets    = set()
    instruments = set()
    verbs       = set()
    targets     = set()
    for part in answer.split(";"):
        part = part.strip().strip("[]").strip()
        if not part:
            continue
        items = [x.strip().lower() for x in part.split(",")]
        if len(items) == 3:
            inst, verb, tgt = items
            if "null" in (inst, verb, tgt):
                continue
            triplets.add((inst, verb, tgt))
            instruments.add(inst)
            verbs.add(verb)
            targets.add(tgt)
    return triplets, instruments, verbs, targets


def surg_accuracy_reward(completions, solution, image_ids, problems, task=None, **kwargs):
    """
    Compute task-aware accuracy rewards with a maximum of 1.0.

    Tool/phase: 0.5 * tool F1 + 0.5 * phase exact match.
    Triplet: 0.5 * triplet F1 + (instrument F1 + verb F1 + target F1) / 6.
    The dataset task column is passed through the trainer's reward kwargs.
    """
    rewards = []
    for i, (comp, sol, image_id, problem) in enumerate(
        zip(completions, solution, image_ids, problems)
    ):
        content   = comp[0]["content"]
        task_type = (task[i] if task else None) or "triplet"


        pred_raw = _extract_tag(content, "answer")
        gt_raw   = _extract_tag(sol, "answer") or sol.strip()

        if not pred_raw.strip():
            rewards.append(0.0)
            continue

        if task_type == "tool&phase":
            pred_tools, pred_phase = _parse_tool_phase(pred_raw)
            gt_tools,   gt_phase   = _parse_tool_phase(gt_raw)

            tool_reward  = _set_f1(pred_tools, gt_tools) * 0.5
            phase_reward = 0.5 if (pred_phase and pred_phase == gt_phase) else 0.0
            reward = tool_reward + phase_reward

        else:  # "triplet"
            pred_tri, pred_inst, pred_verb, pred_tgt = _parse_triplets(pred_raw)
            gt_tri,   gt_inst,   gt_verb,   gt_tgt   = _parse_triplets(gt_raw)

            triplet_reward = _set_f1(pred_tri,  gt_tri)  * 0.5
            inst_reward    = _set_f1(pred_inst, gt_inst) * (1.0 / 6)
            verb_reward    = _set_f1(pred_verb, gt_verb) * (1.0 / 6)
            tgt_reward     = _set_f1(pred_tgt,  gt_tgt)  * (1.0 / 6)
            reward = triplet_reward + inst_reward + verb_reward + tgt_reward

        rewards.append(float(reward))

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"--- {current_time} SurgAccuracy reward={reward:.4f} task={task_type} ---\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Pred <answer>: {pred_raw}\n")
                f.write(f"GT  <answer>: {gt_raw}\n\n")

    return rewards


def accuracy_reward_test(completions, solution, image_ids, problems, **kwargs):
    """
    Accuracy reward test function:
      - 2.0 if the answer is exactly the same as ground truth
      - 0.0 otherwise
    """
    rewards = []
    
    for comp, sol, image_id, problem in zip(completions, solution, image_ids, problems):
        content = comp[0]["content"]  # typical structure: [{ "role":"assistant", "content": "..."}]
        
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        # Extract ground truth answer
        gt_match = re.search(r"<answer>(.*?)</answer>", sol, flags=re.DOTALL|re.IGNORECASE)
        if gt_match:
            ground_truth = gt_match.group(1).strip()
        else:
            ground_truth = sol.strip() if isinstance(sol, str) else ""
        
        # Extract predicted answer
        pred_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL|re.IGNORECASE)
        if pred_match:
            predicted_answer = pred_match.group(1).strip()
        else:
            predicted_answer = ""
        
        # Compare answers exactly (case-sensitive, whitespace-sensitive)
        if predicted_answer == ground_truth and predicted_answer:
            reward = 2.0
        else:
            reward = 0.0
        
        rewards.append(float(reward))
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} AccuracyTest reward: {reward:.1f} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Problem: {problem}\n\n")
                f.write(f"Pred content: {content}\n")
                f.write(f"GT solution: {sol}\n")
                f.write(f"Pred <answer>: {predicted_answer}\n")
                f.write(f"GT <answer>: {ground_truth}\n")
                f.write(f"Match: {predicted_answer == ground_truth}\n\n")
    
    return rewards


reward_funcs_registry = {
    "accuracy":      accuracy_reward,
    "format":        format_reward,
    "openqa":        openqa_reward,
    "surg_accuracy": surg_accuracy_reward,
    "accuracy_test": accuracy_reward_test,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset_env = os.environ.get("DATASET") or ""
    if dataset_env == "Huatuo" or dataset_env == "ISIC":
        # for medical multi-choice questions
        QUESTION_TEMPLATE = """
        {Question} 
        Your task: 
        1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
        2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
        3. No extra information or text outside of these tags.
        """
    elif dataset_env in {"CUSTOM_OPENQA", "VQA-RAD", "SLAKE", "CholecT45", "Cholec_test"}:
        # open-ended VQA / open QA
        QUESTION_TEMPLATE = (
            "{Question}\n\n"
            "Your task:\n"
            "1. Think step by step and put your reasoning in <think>...</think>.\n"
            "2. Put ONLY the final answer in <answer>...</answer>.\n"
            "3. Do NOT output any extra text outside of these tags."
        )
    else:
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    if  os.environ.get("DATASET") == 'VQA-RAD':
        # todo: potential bugs for CPU not GPU training
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)['train']

        def transform_example(example):
            # Rename 'question' to 'problem'
            example['problem'] = example.pop('question')
            # Create 'solution' column from 'answer'
            example['solution'] = f"<answer> {example['answer']} </answer>"
            return example

        dataset = dataset.map(transform_example)
        dataset = dataset.remove_columns('answer')

    elif os.environ.get("DATASET") == 'SLAKE':
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split='train').select(range(2000))
        base_image_dir = f'{script_args.dataset_name}/SLAKE/imgs'


        def transform_example(example):
            # Construct the full image path
            image_path = os.path.join(base_image_dir, example['img_name'])
            
            # Load the image
            example['image'] = PILImage.open(image_path).convert('RGBA')
            
            # Rename 'question' to 'problem'
            example['problem'] = example.pop('question')
            
            # Create 'solution' column from 'answer'
            example['solution'] = f"<answer> {example['answer']} </answer>"
            
            return example
        
        dataset = dataset.filter(lambda example: example['answer_type'] == 'CLOSED' and example['modality'] == 'CT')
        # Apply the transformation to the dataset
        dataset = dataset.map(transform_example, num_proc=2)
        # Remove the original columns that are no longer needed
        dataset = dataset.remove_columns(['img_id', 'img_name', 'answer', 'q_lang', 'location', 'modality', 'answer_type', 'base_type', 'content_type', 'triple', 'qid'])
    
    elif os.environ.get("DATASET") == 'Huatuo':
        dataset = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data", split="test")

        dataset_mr = dataset.filter(lambda x: x["subset"] == "MR (Mag-netic Resonance Imaging)").select(range(500))
        dataset_ct = dataset.filter(lambda x: x["subset"] == "CT(Computed Tomography)").select(range(500))
        dataset_xray = dataset.filter(lambda x: x["subset"] == "X-Ray").select(range(500))

        # Select dataset based on the parameter
        if script_args.dataset_selection == "MR":
            dataset = dataset_mr
        elif script_args.dataset_selection == "CT":
            dataset = dataset_ct
        elif script_args.dataset_selection == "XRAY":
            dataset = dataset_xray
        elif script_args.dataset_selection == "COMBINED":
            dataset = concatenate_datasets([dataset_mr, dataset_ct, dataset_xray])
        else:
            raise ValueError(f"Invalid dataset_selection: {script_args.dataset_selection}. Must be one of: 'MR', 'CT', 'XRAY', 'COMBINED'")
        
        # shuffle the dataset
        dataset = dataset.shuffle(seed=42)



        print("dataset Huatuo loaded")
        def create_problem_solution(example, base_image_path):
            """
            1) Create a new column 'problem' by combining the question and options into a multiple-choice format.
            2) Convert the 'answer' to its corresponding letter (A, B, C, ...) and store it in 'solution'.
            3) Load the image from base_image_path and store the actual image object in 'image'.
            4) Remove all other keys, keeping only 'image', 'problem', 'solution'.
            """
            # We'll support up to 10 options (A-J).
            letter_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            
            # Combine question + labeled options
            labeled_options = [f"{letter_map[i]}) {opt}" for i, opt in enumerate(example["options"])]
            example['problem'] = example["question"] + "\n" + "\n".join(labeled_options)
            
            # Convert text answer to letter
            correct_index = example["options"].index(example["answer"])
            example["solution"] = letter_map[correct_index]
            
            # Load the actual image from disk

            example['image_id'] = example["image"][0]
            image_path = os.path.join(base_image_path, example["image"][0])
            example["image"] = PILImage.open(image_path).convert("RGB")
            return example
        
        base_image_path = script_args.dataset_name
        dataset = dataset.map(lambda x: create_problem_solution(x, base_image_path), num_proc=2)
        dataset = dataset.remove_columns(["question", "dataset", "subset", "answer"])
        dataset = dataset.cast_column("image", Image()) # todo: to optimize 
    
    elif os.environ.get("DATASET") == 'ISIC':
        dataset = load_dataset(f"{script_args.dataset_name}/ISIC-2017", split="train").select(range(1600))
        print("dataset ISIC loaded")

        def filter_by_pixels(example):
            # Get width, height from the PIL Image
            width, height = example["image"].size
            # Keep only those whose pixel count is under 1e6
            return (width * height) < 1_000_000

        def add_problem_and_solution(example):
            """
            Given an example of the form:
            {
            'image': <some PIL image>,
            'label': <integer: 0,1,2>
            }
            we add two columns:
            - 'problem': A fixed multiple-choice question about skin cancer.
            - 'solution': The corresponding letter ('A','B','C') derived from 'label'.
            """
            # Fixed question text
            problem_text = (
                """Which type of skin cancer is it?\n
                A) melanoma\n
                B) nevus\n
                C) seborrheic_keratosis"""
            )

            # Map label -> letter
            label_map = ["A", "B", "C"]
            # Derive the solution letter based on example["label"]
            solution_letter = label_map[example["label"]]

            # Update the example with new columns
            example["problem"] = problem_text
            example["solution"] = solution_letter
            return example

        dataset = dataset.filter(filter_by_pixels)
        dataset = dataset.map(add_problem_and_solution)
        dataset = dataset.remove_columns(["label"])

    elif os.environ.get("DATASET") == 'CholecT45':
        """
        CholecT45 surgical video dataset for [instrument, verb, target] recognition.
        """

        dataset = load_from_disk(script_args.dataset_name)
        
        print(f"Loaded CholecT45 dataset with {len(dataset)} samples")
        

        # - image: PIL Image



        

        num_with_triplets = sum(1 for x in dataset if x['num_triplets'] > 0)
        num_multiple_triplets = sum(1 for x in dataset if x['num_triplets'] > 1)
        print(f"   - Frames with annotations: {num_with_triplets}")
        print(f"   - Frames with multiple triplets: {num_multiple_triplets}")
        

        # dataset = dataset.filter(lambda x: x['num_triplets'] > 0)

        

        columns_to_remove = [col for col in dataset.column_names 
                            if col not in ['image', 'problem', 'solution', 'image_id']]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
    
    elif os.environ.get("DATASET") == 'Cholec_test':
        """
        CholecT45 test subset for [instrument, verb, target] recognition.
        """

        dataset = load_from_disk(script_args.dataset_name)
        
        print(f"Loaded Cholec_test dataset with {len(dataset)} samples")
        

        if 'num_triplets' in dataset.column_names:
            num_with_triplets = sum(1 for x in dataset if x['num_triplets'] > 0)
            num_multiple_triplets = sum(1 for x in dataset if x['num_triplets'] > 1)
            print(f"   - Frames with annotations: {num_with_triplets}")
            print(f"   - Frames with multiple triplets: {num_multiple_triplets}")
        

        columns_to_remove = [col for col in dataset.column_names 
                            if col not in ['image', 'problem', 'solution', 'image_id']]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

    elif os.environ.get("DATASET") == "CUSTOM_OPENQA":
        """
        Custom open-ended QA dataset saved by `datasets.Dataset.save_to_disk`.
        Required columns: image, problem, solution (contains <think>/<answer>), image_id
        """
        dataset = load_from_disk(script_args.dataset_name)
        print(f"Loaded CUSTOM_OPENQA dataset with {len(dataset)} samples")

        # Ensure required columns
        if "image_id" not in dataset.column_names:
            dataset = dataset.add_column("image_id", [str(i) for i in range(len(dataset))])
        if "problem" not in dataset.column_names:
            raise ValueError("CUSTOM_OPENQA dataset is missing required column: problem")
        if "solution" not in dataset.column_names:
            raise ValueError("CUSTOM_OPENQA dataset is missing required column: solution")
        if "image" not in dataset.column_names:
            raise ValueError("CUSTOM_OPENQA dataset is missing required column: image")

        columns_to_keep = {"image", "problem", "solution", "image_id", "task"}
        columns_to_remove = [c for c in dataset.column_names if c not in columns_to_keep]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

    
    if "image" in dataset.features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    
    trainer_cls = Qwen2VLGRPOTrainer


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
