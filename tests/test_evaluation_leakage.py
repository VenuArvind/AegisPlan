"""
Automated Verification of Evaluation Integrity & Zero Data Contamination.

Ensures that the 25 benchmark clinical briefs (20 feasible + 5 infeasible stress tests)
in aegisplan.benchmark.BENCHMARK_BRIEFS have zero overlap with:
  1. The 400 SFT training examples (dataset/train.jsonl)
  2. The 50 validation examples (dataset/val.jsonl)
  3. The 500 GRPO RLVR exploration prompts (dataset/grpo_rlvr_prompts.jsonl)
"""

import json
from pathlib import Path
import pytest
from aegisplan.benchmark import BENCHMARK_BRIEFS

BASE_DIR = Path(__file__).resolve().parent.parent

def _normalize_text(text: str) -> str:
    """Normalize text for strict semantic comparison."""
    return " ".join(text.lower().strip().split())

def _load_train_prompts():
    train_file = BASE_DIR / "dataset" / "train.jsonl"
    prompts = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for m in item.get("messages", []):
                if m.get("role") == "user":
                    prompts.add(_normalize_text(m["content"]))
    return prompts

def _load_val_prompts():
    val_file = BASE_DIR / "dataset" / "val.jsonl"
    prompts = set()
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for m in item.get("messages", []):
                if m.get("role") == "user":
                    prompts.add(_normalize_text(m["content"]))
    return prompts

def _load_grpo_prompts():
    grpo_file = BASE_DIR / "dataset" / "grpo_rlvr_prompts.jsonl"
    prompts = set()
    with open(grpo_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            brief_text = item.get("brief", "")
            if brief_text:
                prompts.add(_normalize_text(brief_text))
            prompt_field = item.get("prompt")
            if isinstance(prompt_field, str):
                prompts.add(_normalize_text(prompt_field))
            elif isinstance(prompt_field, list):
                for m in prompt_field:
                    if isinstance(m, dict) and m.get("role") == "user":
                        prompts.add(_normalize_text(m["content"]))
    return prompts

def test_zero_contamination_against_training_set():
    """Assert 0 benchmark briefs overlap with the 400 training examples."""
    train_prompts = _load_train_prompts()
    assert len(train_prompts) >= 390

    for b in BENCHMARK_BRIEFS:
        norm_brief = _normalize_text(b["brief"])
        for tp in train_prompts:
            assert norm_brief not in tp, f"Contamination detected! Brief '{b['id']}' found in train.jsonl"

def test_zero_contamination_against_validation_set():
    """Assert 0 benchmark briefs overlap with the 50 validation examples."""
    val_prompts = _load_val_prompts()
    assert len(val_prompts) >= 45

    for b in BENCHMARK_BRIEFS:
        norm_brief = _normalize_text(b["brief"])
        for vp in val_prompts:
            assert norm_brief not in vp, f"Contamination detected! Brief '{b['id']}' found in val.jsonl"

def test_zero_contamination_against_grpo_prompts():
    """Assert 0 benchmark briefs overlap with the 500 GRPO RLVR exploration prompts."""
    grpo_prompts = _load_grpo_prompts()
    assert len(grpo_prompts) >= 480

    for b in BENCHMARK_BRIEFS:
        norm_brief = _normalize_text(b["brief"])
        assert norm_brief not in grpo_prompts, f"Contamination detected! Brief '{b['id']}' found in grpo_rlvr_prompts.jsonl"

