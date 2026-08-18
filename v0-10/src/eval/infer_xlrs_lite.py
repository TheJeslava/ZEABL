#!/usr/bin/env python3
"""Run released ZoomEarth weights on the official XLRS-Bench-lite split."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import re
import time
import traceback

from accelerate import Accelerator
from datasets import load_from_disk
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl


Image.MAX_IMAGE_PIXELS = None

DATASET_ID = "initiacms/XLRS-Bench-lite"
DATASET_REVISION = "e540ee2aa745ce9a83784ae76541ddb7f79f03ac"
TASK_PAIRS = [
    "Complex reasoning/Anomaly Detection and Interpretation",
    "Complex reasoning/Environmental condition reasoning",
    "Complex reasoning/Route planning",
    "Counting/Counting with changing detection",
    "Counting/Counting with complex reasoning",
    "Counting/Overall counting",
    "Counting/Regional counting",
    "Land use classification/Overall Land use classification",
    "Land use classification/Regional Land use classification",
    "Object properties/Object classification",
    "Object properties/Object color",
    "Object properties/Object motion state",
    "Object spatial relationship/Object spatial relationship",
]
EXPECTED_CATEGORY_COUNTS = {
    "Complex reasoning/Anomaly Detection and Interpretation": 100,
    "Complex reasoning/Environmental condition reasoning": 100,
    "Complex reasoning/Route planning": 100,
    "Counting/Counting with changing detection": 60,
    "Counting/Counting with complex reasoning": 100,
    "Counting/Overall counting": 60,
    "Counting/Regional counting": 100,
    "Land use classification/Overall Land use classification": 100,
    "Land use classification/Regional Land use classification": 200,
    "Object properties/Object classification": 800,
    "Object properties/Object color": 800,
    "Object properties/Object motion state": 60,
    "Object spatial relationship/Object spatial relationship": 500,
}
FULL_DATASET_SIZE = sum(EXPECTED_CATEGORY_COUNTS.values())

SYSTEM_PREFIX = """
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
"""
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
MULTI_SELECT_CATEGORY = "Land use classification/Overall Land use classification"
REFERENCE_BBOX_PATTERN = re.compile(
    r"Image resolution:\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?).*?"
    r"Bounding box:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"(\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)
REFERENCE_RESOLUTION_PATTERN = re.compile(
    r"Image resolution:\s*\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)
REFERENCE_BOX_ONLY_PATTERN = re.compile(
    r"Bounding box:\s*\[\s*\d+(?:\.\d+)?\s*,\s*"
    r"\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*"
    r"\d+(?:\.\d+)?\s*\]",
    re.IGNORECASE,
)

# Instruction from the official ZoomEarth inference entry point, with the final
# answer sentence linked to the XLRS answer format as in ver1.
ZOOMEARTH_INSTRUCTION = """
You are an intelligent remote sensing analyst.
Given a natural language question about a satellite image, generate a structured reasoning answer as follows:
1. <think> ... </think>
    - Provide a neutral one-sentence description of the whole image scene.
    - Cropping task: "This question is asking about <short intent>, therefore I need to crop the image to examine the surroundings of the mentioned target."
    - Non-cropping task: "This question is asking about <short intent>, therefore I need to analyze the entire image without cropping."
    - Include:
        * Question Intent: describe the type of question (object category, spatial relation, count, etc.) and needed visual info.
        * Localization Strategy:
            - Cropping: approximate referent object location in natural language (no coordinates).
            - Non-cropping: strategy to detect all relevant objects.      * Reasoning Result:
    - Cropping: output exactly one JSON-formatted bbox for the referent:          [{"bbox_2d": [x_min,y_min,x_max,y_max], "label": "<short description>"}]
    - Non-cropping: summarize how detected objects will be used to produce the count.
2. <think> ... </think> (only when saw the cropped image)
    - Explain how to reason step by step from the referent (or detected objects) to the final answer. 
3. <answer> ... </answer>
    - Your final answer, strictly following the format requirements in the XLRS answer format below.
Rules: 
    - Always return exactly one <answer> block, for tasks that need cropping, you can provide the bounding box of the object you are intrested, after given the cropped image, you can generate another <think> block to find the answer. 
    - For cropping tasks, also include a bounidng box in <stage_2_reasoning> block 
    - If unsure about localization, make a best guess—never say uncertain.
"""

# These formats validate generated answers; prompt text is built by
# answer_protocol() using the same structure as ver1.
ANSWER_FORMAT_SINGLE_4 = "single_4"
ANSWER_FORMAT_SINGLE_2 = "single_2"
ANSWER_FORMAT_MULTI_4 = "multi_4"

ANSWER_FORMAT_BY_CATEGORY = {
    category: ANSWER_FORMAT_SINGLE_4
    for category in TASK_PAIRS
}
ANSWER_FORMAT_BY_CATEGORY["Object properties/Object motion state"] = ANSWER_FORMAT_SINGLE_2
ANSWER_FORMAT_BY_CATEGORY["Land use classification/Overall Land use classification"] = (
    ANSWER_FORMAT_MULTI_4
)

def answer_format_for_category(category: str) -> str:
    """Return the output protocol selected for one XLRS category."""
    try:
        return ANSWER_FORMAT_BY_CATEGORY[category]
    except KeyError as error:
        raise ValueError(f"unknown XLRS category: {category}") from error


def answer_protocol(sample: dict) -> str:
    """Build the same option-aware XLRS answer protocol used by ver1."""
    options = sample["multi-choice options"]
    labels = "".join(chr(ord("A") + index) for index in range(len(options)))
    if sample["category"] == MULTI_SELECT_CATEGORY:
        protocol = (
            "Select every option that applies. Inside <answer>, output only the "
            f"applicable uppercase letters from {labels}, in alphabetical order, "
            "with no spaces or separators."
        )
    else:
        protocol = (
            "Select exactly one best option. Inside <answer>, output exactly one "
            f"uppercase letter from {labels}, with no other text."
        )
    return f"\nXLRS answer format:\n{protocol}\n"


def xlrs_doc_to_text(doc: dict, question: str | None = None) -> str:
    """Build the question and choices; output formatting is added separately."""
    if doc["category"] not in TASK_PAIRS:
        raise ValueError(f"unknown XLRS category: {doc['category']}")
    return (
        (doc["question"] if question is None else question)
        + "\n\nThe choices are listed below:\n"
        + "\n".join(doc["multi-choice options"])
    )


def extract_characters_regex(text: object) -> str:
    """Official XLRS answer extraction, normalized only for stable JSON output."""
    if isinstance(text, dict) or text is None:
        text = ""
    text = str(text).strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option isThe correct option is",
        "Best answer:Best option:",
    ]
    for answer_prefix in answer_prefixes:
        text = text.replace(answer_prefix, "")
    if not re.search("[ABCDE]", text):
        return ""
    matches = re.findall(r"\(([a-eA-E])\)", text)
    if not matches:
        matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", text)
    if not matches:
        matches = re.findall(r"[a-eA-E]", text)
    return "" if not matches else "".join(sorted({match.upper() for match in matches}))


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1) if match else None


def extract_answer_payload(text: str) -> str | None:
    """Extract a wrapped answer or an unambiguous bare-letter response."""
    wrapped = extract_answer(text)
    if wrapped is not None:
        return wrapped
    payload = text.strip()
    return payload if re.fullmatch(r"[A-E]+", payload, re.IGNORECASE) else None


def answer_format(answer: object) -> str:
    """Classify generated answer payloads without changing official scoring."""
    if answer is None:
        return "missing"
    normalized = re.sub(r"[\s,]+", "", str(answer)).upper()
    if not normalized:
        return "empty"
    return "letters" if re.fullmatch(r"[A-E]+", normalized) else "semantic"


def answer_obeys_protocol(answer: object, category: str) -> bool:
    """Check the raw <answer> payload against the active category protocol."""
    if answer is None:
        return False
    payload = str(answer).strip().upper()
    output_format = answer_format_for_category(category)
    if output_format == ANSWER_FORMAT_SINGLE_4:
        return re.fullmatch(r"[A-D]", payload) is not None
    if output_format == ANSWER_FORMAT_SINGLE_2:
        return re.fullmatch(r"[AB]", payload) is not None
    if re.fullmatch(r"[A-D]+", payload) is None:
        return False
    return payload == "".join(sorted(set(payload)))


def extract_bboxes(text: str, scale: float = 1.0) -> list[list[float]]:
    matches = re.findall(r'"bbox_2d"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    bboxes = []
    for match in matches:
        try:
            numbers = [float(value.strip()) * scale for value in match.split(",")]
        except ValueError:
            continue
        if len(numbers) == 4:
            bboxes.append(numbers)
    return bboxes


def extract_reference_bbox(question: str) -> tuple[float, float, list[float]] | None:
    """Read an XLRS reference bbox expressed in the question's image coordinates."""
    match = REFERENCE_BBOX_PATTERN.search(question)
    if not match:
        return None
    image_width, image_height, x1, y1, x2, y2 = map(float, match.groups())
    if image_width <= 0 or image_height <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid reference bbox in question: {match.group(0)!r}")
    return image_width, image_height, [x1, y1, x2, y2]


def reference_bbox_for_image(
    reference: tuple[float, float, list[float]], image: Image.Image
) -> list[float]:
    """Map question coordinates to the actual decoded image dimensions."""
    reference_width, reference_height, bbox = reference
    x_scale = image.width / reference_width
    y_scale = image.height / reference_height
    return [
        bbox[0] * x_scale,
        bbox[1] * y_scale,
        bbox[2] * x_scale,
        bbox[3] * y_scale,
    ]


def map_bbox_between_images(
    bbox: list[float], source: Image.Image, target: Image.Image
) -> list[float]:
    """Map a bbox between decoded image coordinate spaces with exact x/y scales."""
    x_scale = target.width / source.width
    y_scale = target.height / source.height
    return [
        bbox[0] * x_scale,
        bbox[1] * y_scale,
        bbox[2] * x_scale,
        bbox[3] * y_scale,
    ]


def clip_bbox_to_image(bbox: list[float], image: Image.Image) -> list[float]:
    """Clip a bbox to an image while requiring a non-empty intersection."""
    clipped = [
        min(max(bbox[0], 0.0), float(image.width)),
        min(max(bbox[1], 0.0), float(image.height)),
        min(max(bbox[2], 0.0), float(image.width)),
        min(max(bbox[3], 0.0), float(image.height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"bbox does not intersect image: bbox={bbox}, size={image.size}")
    return clipped


def _format_coordinate(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def question_with_reference_bbox(
    question: str, image: Image.Image, bbox: list[float]
) -> str:
    """Rewrite a reference question for a specific image and bbox coordinate space."""
    if extract_reference_bbox(question) is None:
        raise ValueError("question does not contain a reference bbox")
    rewritten = REFERENCE_RESOLUTION_PATTERN.sub(
        f"Image resolution: {image.width} x {image.height}",
        question,
        count=1,
    )
    bbox_text = ", ".join(_format_coordinate(value) for value in bbox)
    return REFERENCE_BOX_ONLY_PATTERN.sub(
        f"Bounding box: [{bbox_text}]", rewritten, count=1
    )


def question_for_global_image(question: str, global_image: Image.Image) -> str:
    """Express a question's reference bbox in the image stage 1 actually sees."""
    reference = extract_reference_bbox(question)
    if reference is None:
        return question
    bbox = reference_bbox_for_image(reference, global_image)
    return question_with_reference_bbox(question, global_image, bbox)


def resize_image(image: Image.Image, max_size: int) -> tuple[Image.Image, float]:
    """Resize and coordinate-scale logic from the released infer.py."""
    width, height = image.size
    scale = max_size / max(width, height)
    if scale < 1:
        resized = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC
        )
        image = resized
    return image, 1 / scale


def crop_box_for_image(
    image: Image.Image, bbox: list[float], min_size: int = 512
) -> list[int]:
    """Return the crop window using the released infer.py geometry."""
    x1, y1, x2, y2 = map(int, bbox)
    width, height = x2 - x1, y2 - y1
    if width < min_size or height < min_size:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size
        if new_x1 < 0:
            new_x2 += -new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 += -new_y1
            new_y1 = 0
        if new_x2 > image.width:
            new_x1 -= new_x2 - image.width
            new_x2 = image.width
        if new_y2 > image.height:
            new_y1 -= new_y2 - image.height
            new_y2 = image.height
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image.width, new_x1 + min_size)
        new_y2 = min(image.height, new_y1 + min_size)
        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    return [x1, y1, x2, y2]


def cut_image(image: Image.Image, bbox: list[float], min_size: int = 512) -> Image.Image:
    """Crop an image using the released infer.py geometry."""
    return image.crop(tuple(crop_box_for_image(image, bbox, min_size)))


def make_stage1_prompt(question_prompt: str, image_count: int, sample: dict) -> str:
    return (
        SYSTEM_PREFIX
        + IMAGE_TOKEN * image_count
        + question_prompt
        + ZOOMEARTH_INSTRUCTION
        + answer_protocol(sample)
        + "<|im_end|><|im_start|>assistant\n"
    )


STAGE1_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def make_stage2_prompt(
    stage1_prompt: str,
    stage1_output: str,
    crop_count: int,
    *,
    omit_stage1_think: bool = False,
) -> str:
    reasoning_prefix = stage1_output.split("<answer>")[0]
    if omit_stage1_think:
        reasoning_prefix = STAGE1_THINK_PATTERN.sub("", reasoning_prefix)
    return stage1_prompt + reasoning_prefix + IMAGE_TOKEN * crop_count


def make_reference_stage2_prompt(question_prompt: str, sample: dict) -> str:
    """Build a crop-only prompt with the v0-3 ZoomEarth instruction block."""
    return (
        SYSTEM_PREFIX
        + IMAGE_TOKEN
        + question_prompt
        + ZOOMEARTH_INSTRUCTION
        + answer_protocol(sample)
        + "<|im_end|><|im_start|>assistant\n"
    )


def normalize_images(value: object) -> list[Image.Image]:
    images = value if isinstance(value, (list, tuple)) else [value]
    normalized = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"dataset image decoded as {type(image)!r}, expected PIL.Image")
        normalized.append(image.convert("RGB"))
    if not normalized:
        raise ValueError("sample contains no images")
    return normalized


def generate(
    prompt: str,
    images: list[Image.Image],
    processor: Qwen2_5_VLProcessor,
    accelerator: Accelerator,
    model: torch.nn.Module,
) -> str:
    processor_images: object = images if len(images) == 1 else [images]
    inputs = processor(
        text=[prompt], images=processor_images, return_tensors="pt", padding="longest"
    ).to(accelerator.device)
    with torch.inference_mode():
        generated = accelerator.unwrap_model(model).generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            num_beams=1,
            temperature=0.01,
        )
    input_length = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(
        generated[0, input_length:], skip_special_tokens=True
    ).strip()


def segmented_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate the vision block-diagonal attention one segment at a time."""
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if position_embeddings is None:
        if rotary_pos_emb is None:
            raise ValueError("vision attention requires rotary position embeddings")
        embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = embedding.cos().float()
        sin = embedding.sin().float()
    else:
        cos, sin = position_embeddings
    q, k = modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(q, k, cos, sin)

    outputs = []
    boundaries = cu_seqlens.tolist()
    for start, end in zip(boundaries, boundaries[1:]):
        query = q[start:end].transpose(0, 1)
        key = k[start:end].transpose(0, 1)
        value = v[start:end].transpose(0, 1)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0
        )
        outputs.append(output.transpose(0, 1))
    if not outputs:
        raise ValueError("vision attention received no sequence segments")
    attention_output = torch.cat(outputs, dim=0).reshape(seq_length, -1)
    return self.proj(attention_output)


def enable_segmented_vision_attention() -> None:
    """Install a memory-bounded equivalent of the upstream block-diagonal SDPA."""
    modeling_qwen2_5_vl.Qwen2_5_VLVisionSdpaAttention.forward = (
        segmented_vision_attention_forward
    )


def load_previous(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[int(record["dataset_position"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def parse_positions(spec: str, dataset_size: int) -> list[int]:
    """Parse comma-separated positions and inclusive ranges, preserving order."""
    positions = []
    seen = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending position range: {item}")
            values = range(start, end + 1)
        else:
            values = [int(item)]
        for position in values:
            if not 0 <= position < dataset_size:
                raise ValueError(
                    f"dataset position {position} is outside 0..{dataset_size - 1}"
                )
            if position not in seen:
                positions.append(position)
                seen.add(position)
    if not positions:
        raise ValueError("--positions did not select any dataset rows")
    return positions


def evenly_spaced_offsets(total: int, count: int) -> list[int]:
    """Select stable offsets spread from the first sample across a category."""
    if total <= 0:
        raise ValueError("total must be positive")
    if count <= 0 or count > total:
        raise ValueError(f"count must be in [1, {total}], got {count}")
    if count == total:
        return list(range(total))
    if count == 1:
        return [0]
    if count == 2:
        return [0, total - 1]

    upper = total - 2
    denominator = count - 1
    offsets = [
        (2 * position * upper + denominator) // (2 * denominator)
        for position in range(count)
    ]
    if len(set(offsets)) != count:
        raise RuntimeError(f"failed to select {count} unique offsets from {total}")
    return offsets


def select_category_positions(dataset, samples_per_category: int) -> tuple[list[int], dict]:
    """Select evenly distributed rows after sorting each category by sample index."""
    sample_indices = dataset["index"]
    rows_by_category: dict[str, list[tuple[int, int]]] = {
        category: [] for category in TASK_PAIRS
    }
    for position, (category, sample_index) in enumerate(
        zip(dataset["category"], sample_indices)
    ):
        if category not in rows_by_category:
            raise ValueError(f"unknown XLRS category: {category}")
        rows_by_category[category].append((int(sample_index), position))

    selected_by_category = {}
    for category in TASK_PAIRS:
        rows = sorted(rows_by_category[category])
        offsets = evenly_spaced_offsets(len(rows), samples_per_category)
        selected_by_category[category] = [rows[offset][1] for offset in offsets]

    positions = sorted(
        position
        for category_positions in selected_by_category.values()
        for position in category_positions
    )
    manifest = {
        "samples": len(positions),
        "samples_per_category": samples_per_category,
        "categories": {
            category: {
                "available": len(rows_by_category[category]),
                "selected": len(selected_by_category[category]),
                "dataset_positions": selected_by_category[category],
                "indices": [
                    int(sample_indices[position])
                    for position in selected_by_category[category]
                ],
            }
            for category in TASK_PAIRS
        },
    }
    return positions, manifest


def score_records(records: list[dict]) -> dict:
    latest = {int(record["dataset_position"]): record for record in records}
    category_stats = {
        category: {"correct": 0, "total": 0, "accuracy": 0.0}
        for category in TASK_PAIRS
    }
    stage1_bbox_failures = 0
    stage2_attempted_samples = 0
    stage2_samples = 0
    error_samples = 0
    multi_image_samples = 0
    correct = 0
    stage1_correct = 0
    fixed_by_stage2 = 0
    harmed_by_stage2 = 0
    stage1_semantic_answers = 0
    stage2_semantic_answers = 0
    stage1_protocol_violations = 0
    stage2_protocol_violations = 0
    question_bbox_samples = 0
    model_bbox_samples = 0
    for record in latest.values():
        category = record["category"]
        if category not in category_stats:
            raise ValueError(f"unknown category in output: {category}")
        is_correct = set(record.get("prediction", "")) == set(str(record["answer"]))
        category_stats[category]["correct"] += int(is_correct)
        category_stats[category]["total"] += 1
        correct += int(is_correct)
        stage1_is_correct = set(record.get("stage1_prediction", "")) == set(
            str(record["answer"])
        )
        stage1_correct += int(stage1_is_correct)
        fixed_by_stage2 += int(not stage1_is_correct and is_correct)
        harmed_by_stage2 += int(stage1_is_correct and not is_correct)
        stage1_bbox_failures += int(not record.get("bbox_resized"))
        stage2_attempted_samples += int(bool(record.get("stage2_prompt")))
        stage2_samples += int(record.get("stage2_used", False))
        error_samples += int(record.get("status") != "ok")
        multi_image_samples += int(record.get("image_count", 0) > 1)
        stage1_semantic_answers += int(
            answer_format(record.get("stage1_answer")) == "semantic"
        )
        stage2_semantic_answers += int(
            bool(record.get("stage2_output"))
            and answer_format(record.get("stage2_answer")) == "semantic"
        )
        stage1_protocol_violations += int(
            not answer_obeys_protocol(record.get("stage1_answer"), category)
        )
        stage2_protocol_violations += int(
            bool(record.get("stage2_output"))
            and not answer_obeys_protocol(record.get("stage2_answer"), category)
        )
        question_bbox_samples += int(record.get("bbox_source") == "question")
        model_bbox_samples += int(record.get("bbox_source") == "model")
    for stats in category_stats.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    weighted_correct_equivalent = sum(
        category_stats[category]["accuracy"] * full_count
        for category, full_count in EXPECTED_CATEGORY_COUNTS.items()
    )
    for category, stats in category_stats.items():
        stats["full_dataset_samples"] = EXPECTED_CATEGORY_COUNTS[category]
        stats["weighted_correct_equivalent"] = (
            stats["accuracy"] * EXPECTED_CATEGORY_COUNTS[category]
        )
    total = len(latest)
    macro = sum(stats["accuracy"] for stats in category_stats.values()) / len(TASK_PAIRS)
    observed_categories = [stats for stats in category_stats.values() if stats["total"]]
    observed_macro = (
        sum(stats["accuracy"] for stats in observed_categories)
        / len(observed_categories)
        if observed_categories
        else 0.0
    )
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "jsonl_record_lines": len(records),
        "duplicate_record_lines": len(records) - len(latest),
        "samples": total,
        "missing_from_full_split": 3080 - total,
        "correct": correct,
        "micro_accuracy": correct / total if total else 0.0,
        "macro_accuracy": macro,
        "observed_categories": len(observed_categories),
        "observed_macro_accuracy": observed_macro,
        "full_dataset_samples": FULL_DATASET_SIZE,
        "category_count_weighted_correct_equivalent": weighted_correct_equivalent,
        "category_count_weighted_accuracy": (
            weighted_correct_equivalent / FULL_DATASET_SIZE
        ),
        "stage1_correct": stage1_correct,
        "stage1_micro_accuracy": stage1_correct / total if total else 0.0,
        "fixed_by_stage2": fixed_by_stage2,
        "harmed_by_stage2": harmed_by_stage2,
        "stage1_bbox_failures": stage1_bbox_failures,
        "stage2_attempted_samples": stage2_attempted_samples,
        "stage2_samples": stage2_samples,
        "stage2_empty_output_fallbacks": stage2_attempted_samples - stage2_samples,
        "error_samples": error_samples,
        "multi_image_samples": multi_image_samples,
        "stage1_semantic_answers": stage1_semantic_answers,
        "stage2_semantic_answers": stage2_semantic_answers,
        "stage1_protocol_violations": stage1_protocol_violations,
        "stage2_protocol_violations": stage2_protocol_violations,
        "question_bbox_samples": question_bbox_samples,
        "model_bbox_samples": model_bbox_samples,
        "categories": category_stats,
    }


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_metrics(output_path: Path, metrics_path: Path) -> dict:
    metrics = score_records(read_records(output_path))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics


def run(args: argparse.Namespace) -> None:
    dataset = load_from_disk(str(args.data_path))["train"]
    if len(dataset) != FULL_DATASET_SIZE:
        raise RuntimeError(f"expected 3,080 XLRS-lite rows, loaded {len(dataset)}")
    observed_counts = Counter(dataset["category"])
    if observed_counts != Counter(EXPECTED_CATEGORY_COUNTS):
        raise RuntimeError(
            f"XLRS-lite category counts differ: {dict(sorted(observed_counts.items()))}"
        )
    selection_manifest = None
    if args.positions:
        positions = parse_positions(args.positions, len(dataset))
    elif args.samples_per_category is not None:
        positions, selection_manifest = select_category_positions(
            dataset, args.samples_per_category
        )
    else:
        stop = (
            len(dataset)
            if args.limit is None
            else min(len(dataset), args.start + args.limit)
        )
        positions = range(args.start, stop)

    if args.selection_output is not None:
        if selection_manifest is None:
            raise ValueError("--selection-output requires --samples-per-category")
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(
            json.dumps(selection_manifest, indent=2, ensure_ascii=False) + "\n"
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.segment_vision_attention:
        enable_segmented_vision_attention()

    previous = load_previous(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints", log_with=[])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path), torch_dtype=torch.float16
    )
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        str(args.model_path), trust_remote_code=True, max_pixels=128 * 128 * 28 * 28
    )
    processor.tokenizer.padding_side = "left"
    model.generation_config.temperature = 0.01
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model = accelerator.prepare(model)
    torch_initial_seed = torch.initial_seed()
    torch_cuda_initial_seed = torch.cuda.initial_seed()

    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8", buffering=1) as output_handle:
        for position in tqdm(positions, desc="Evaluating XLRS-lite"):
            old_record = previous.get(position)
            if args.resume and old_record and old_record.get("status") == "ok":
                continue
            started = time.monotonic()
            doc = dataset[position]
            base_record = {
                "dataset_position": position,
                "index": doc["index"],
                "path": doc["path"],
                "question": doc["question"],
                "multi_choice_options": doc["multi-choice options"],
                "answer": doc["answer"],
                "category": doc["category"],
                "l2_category": doc["l2-category"],
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "model_path": str(args.model_path),
                "requested_seed": args.seed,
                "torch_initial_seed": torch_initial_seed,
                "torch_cuda_initial_seed": torch_cuda_initial_seed,
                "output_format": answer_format_for_category(doc["category"]),
                "segmented_vision_attention": args.segment_vision_attention,
            }
            try:
                original_images = normalize_images(doc["image"])
                global_images = []
                scales = []
                for image in original_images:
                    resized, scale = resize_image(image, max_size=1024)
                    global_images.append(resized)
                    scales.append(scale)
                reference_bbox = extract_reference_bbox(doc["question"])
                if reference_bbox is not None and len(global_images) != 1:
                    raise ValueError(
                        "reference-bbox questions must contain exactly one image"
                    )
                stage1_question = (
                    question_for_global_image(doc["question"], global_images[0])
                    if reference_bbox is not None
                    else doc["question"]
                )
                question_prompt = xlrs_doc_to_text(doc, question=stage1_question)
                stage1_prompt = make_stage1_prompt(
                    question_prompt, len(global_images), doc
                )
                stage1_started = time.monotonic()
                stage1_output = generate(
                    stage1_prompt, global_images, processor, accelerator, model
                )
                stage1_seconds = time.monotonic() - stage1_started
                resized_bboxes = extract_bboxes(stage1_output)
                model_bbox_raw = resized_bboxes[0] if resized_bboxes else None
                if reference_bbox is not None:
                    original_bboxes = [
                        reference_bbox_for_image(reference_bbox, image)
                        for image in original_images
                    ]
                    bbox_resized = reference_bbox_for_image(
                        reference_bbox, global_images[0]
                    )
                    bbox_source = "question"
                elif model_bbox_raw is not None:
                    original_bboxes = [
                        map_bbox_between_images(model_bbox_raw, global_image, image)
                        for global_image, image in zip(global_images, original_images)
                    ]
                    bbox_resized = model_bbox_raw
                    bbox_source = "model"
                else:
                    original_bboxes = []
                    bbox_resized = None
                    bbox_source = "none"
                crop_images = []
                raw_crop_images = []
                crop_boxes_original = []
                stage2_output = ""
                stage2_seconds = 0.0
                stage2_question = None
                stage2_bbox = None
                stage2_prompt_style = None
                if bbox_resized is not None:
                    for image, original_bbox in zip(original_images, original_bboxes):
                        crop_box = crop_box_for_image(
                            image, original_bbox, min_size=512
                        )
                        raw_crop = image.crop(tuple(crop_box))
                        cropped, _ = resize_image(raw_crop, max_size=512)
                        crop_boxes_original.append(crop_box)
                        raw_crop_images.append(raw_crop)
                        crop_images.append(cropped)
                    if reference_bbox is not None:
                        crop_box = crop_boxes_original[0]
                        original_bbox = original_bboxes[0]
                        local_raw_bbox = [
                            original_bbox[0] - crop_box[0],
                            original_bbox[1] - crop_box[1],
                            original_bbox[2] - crop_box[0],
                            original_bbox[3] - crop_box[1],
                        ]
                        stage2_bbox = map_bbox_between_images(
                            local_raw_bbox, raw_crop_images[0], crop_images[0]
                        )
                        stage2_bbox = clip_bbox_to_image(
                            stage2_bbox, crop_images[0]
                        )
                        stage2_question = question_with_reference_bbox(
                            doc["question"], crop_images[0], stage2_bbox
                        )
                        stage2_question_prompt = xlrs_doc_to_text(
                            doc, question=stage2_question
                        )
                        stage2_prompt = make_reference_stage2_prompt(
                            stage2_question_prompt, doc
                        )
                        stage2_images = crop_images
                        stage2_prompt_style = "reference_crop_local"
                    else:
                        stage2_prompt = make_stage2_prompt(
                            stage1_prompt, stage1_output, len(crop_images)
                        )
                        stage2_images = global_images + crop_images
                        stage2_prompt_style = "released_global_plus_crop"
                    stage2_started = time.monotonic()
                    stage2_output = generate(
                        stage2_prompt,
                        stage2_images,
                        processor,
                        accelerator,
                        model,
                    )
                    stage2_seconds = time.monotonic() - stage2_started
                else:
                    stage2_prompt = ""

                final_output = stage2_output if stage2_output else stage1_output
                stage1_answer = extract_answer_payload(stage1_output)
                stage1_prediction = extract_characters_regex(
                    stage1_answer if stage1_answer is not None else stage1_output
                )
                stage2_answer = extract_answer_payload(stage2_output)
                final_answer = extract_answer_payload(final_output)
                prediction = extract_characters_regex(
                    final_answer if final_answer is not None else final_output
                )
                record = {
                    **base_record,
                    "status": "ok",
                    "error": None,
                    "image_count": len(original_images),
                    "original_image_sizes": [list(image.size) for image in original_images],
                    "global_image_sizes": [list(image.size) for image in global_images],
                    "global_to_original_scales": scales,
                    "stage1_question": stage1_question,
                    "question_prompt": question_prompt,
                    "stage1_prompt": stage1_prompt,
                    "stage1_output": stage1_output,
                    "stage1_answer": stage1_answer,
                    "stage1_answer_format": answer_format(stage1_answer),
                    "stage1_answer_obeys_protocol": answer_obeys_protocol(
                        stage1_answer, doc["category"]
                    ),
                    "stage1_prediction": stage1_prediction,
                    "stage1_correct": set(stage1_prediction) == set(str(doc["answer"])),
                    "model_bbox_raw": model_bbox_raw,
                    "reference_bbox": reference_bbox,
                    "stage1_reference_bbox": (
                        [
                            float(global_images[0].width),
                            float(global_images[0].height),
                            bbox_resized,
                        ]
                        if reference_bbox is not None
                        else None
                    ),
                    "bbox_resized": bbox_resized,
                    "bbox_resized_coordinate_space": "global_image",
                    "bboxes_original": original_bboxes,
                    "bbox_source": bbox_source,
                    "crop_boxes_original": crop_boxes_original,
                    "crop_image_sizes": [list(image.size) for image in crop_images],
                    "stage2_used": bool(stage2_output),
                    "stage2_prompt_style": stage2_prompt_style,
                    "stage2_question": stage2_question,
                    "stage2_bbox": stage2_bbox,
                    "stage2_bbox_coordinate_space": (
                        "crop_image" if stage2_bbox is not None else None
                    ),
                    "stage2_prompt": stage2_prompt,
                    "stage2_output": stage2_output,
                    "stage2_answer": stage2_answer,
                    "stage2_answer_wrapped": (
                        extract_answer(stage2_output) is not None
                        if stage2_output
                        else None
                    ),
                    "stage2_answer_format": answer_format(stage2_answer),
                    "stage2_answer_obeys_protocol": (
                        answer_obeys_protocol(
                            stage2_answer, doc["category"]
                        )
                        if stage2_output
                        else None
                    ),
                    "final_answer": final_answer,
                    "prediction": prediction,
                    "correct": set(prediction) == set(str(doc["answer"])),
                    "stage1_seconds": stage1_seconds,
                    "stage2_seconds": stage2_seconds,
                    "total_seconds": time.monotonic() - started,
                }
            except Exception as error:
                record = {
                    **base_record,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "image_count": 0,
                    "bbox_resized": None,
                    "stage2_used": False,
                    "prediction": "",
                    "stage1_prediction": "",
                    "correct": False,
                    "total_seconds": time.monotonic() - started,
                }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            del doc
            gc.collect()

    metrics = write_metrics(args.output, args.metrics_output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--positions",
        help="comma-separated dataset positions and inclusive ranges, e.g. 760-767,860",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        help="select this many evenly distributed samples from every category",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        help="write the balanced selection manifest as JSON",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--segment-vision-attention",
        action="store_true",
        help="compute each block-diagonal vision-attention segment separately",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.score_only and (args.model_path is None or args.data_path is None):
        parser.error("--model-path and --data-path are required unless --score-only is used")
    selectors = int(bool(args.positions)) + int(args.samples_per_category is not None)
    if selectors > 1:
        parser.error("--positions and --samples-per-category are mutually exclusive")
    if selectors and (args.start != 0 or args.limit is not None):
        parser.error(
            "--positions/--samples-per-category cannot be combined with --start or --limit"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.score_only:
        metrics = write_metrics(args.output, args.metrics_output)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        run(args)


if __name__ == "__main__":
    main()
