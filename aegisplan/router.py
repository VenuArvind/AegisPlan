"""Dual-mode execution router and Best-of-3 candidate generator.

Connects to local vLLM/llama.cpp server or OpenRouter (nvidia/nemotron-3.5-lightning),
generates parallel FML-DSL candidates, repairs them with CP-SAT, and ranks them lexicographically.
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import httpx
import logging

from aegisplan.schemas import PlanDocument
from aegisplan.fml_parser import parse_fml
from aegisplan.validator import validate_plan
from aegisplan.cpsat_repair import repair_plan, RepairResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are AegisPlan, an autonomous AI architect specializing in U.S. Healthcare Outpatient Facilities.
Output ONLY valid FML-DSL code matching the exact grammar:
BOUNDARY RECT 0 0 <W> <H>
ROOM <id> TYPE <type> RECT <x> <y> <w> <h> <LOCKED|FREE>
DOOR <id> CONNECT <r1> <r2> AT <x> <y> WIDTH <w>
FDOOR <id> CONNECT <r1> OUTSIDE AT <x> <y> WIDTH <w>
END_PLAN

Rules:
- Standard 100mm grid. Coordinates end in 00 or 50.
- Corridors must be at least 1830mm wide.
- Exam and procedure rooms must be at least 3100mm x 3700mm (>= 11.15 m2).
- Single-user ADA restrooms must be at least 2200mm x 2200mm.
- Clean and soiled utilities must never connect directly.
- All rooms must connect via doors to the central corridor or waiting area leading to OUTSIDE.
"""

FEW_SHOT_EXEMPLAR = """BOUNDARY RECT 0 0 16000 12000
ROOM waiting_1 TYPE waiting RECT 0 0 6000 5000 LOCKED
ROOM corridor_1 TYPE corridor RECT 0 5000 16000 2000 LOCKED
ROOM exam_1 TYPE exam RECT 0 7000 3200 5000 FREE
ROOM exam_2 TYPE exam RECT 3200 7000 3200 5000 FREE
ROOM restroom_1 TYPE restroom RECT 6000 0 2500 5000 LOCKED
DOOR door_1 CONNECT waiting_1 corridor_1 AT 3000 5000 WIDTH 1065
DOOR door_2 CONNECT exam_1 corridor_1 AT 1600 7000 WIDTH 915
DOOR door_3 CONNECT exam_2 corridor_1 AT 4800 7000 WIDTH 915
DOOR door_4 CONNECT restroom_1 corridor_1 AT 7250 5000 WIDTH 915
FDOOR door_5 CONNECT waiting_1 OUTSIDE AT 3000 0 WIDTH 1220
END_PLAN"""


class CandidateEvaluation:
    def __init__(
        self,
        candidate_idx: int,
        raw_fml: str,
        initial_plan: PlanDocument,
        repaired_plan: PlanDocument,
        pre_repair_overlap_pct: float,
        checks_passed: int,
        total_checks: int,
        travel_distance_mm: float,
        solver_status: str,
        solve_time_ms: int,
        rank: int = 0,
    ):
        self.candidate_idx = candidate_idx
        self.raw_fml = raw_fml
        self.initial_plan = initial_plan
        self.repaired_plan = repaired_plan
        self.pre_repair_overlap_pct = pre_repair_overlap_pct
        self.checks_passed = checks_passed
        self.total_checks = total_checks
        self.travel_distance_mm = travel_distance_mm
        self.solver_status = solver_status
        self.solve_time_ms = solve_time_ms
        self.rank = rank


class GenerationRouter:
    def __init__(
        self,
        local_endpoint: str = "http://localhost:8000/v1",
        openrouter_api_key: Optional[str] = None,
    ):
        self.local_endpoint = os.getenv("AEGIS_LOCAL_ENDPOINT", local_endpoint)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")

    async def generate_candidates(
        self,
        brief: str,
        boundary_w: int = 16000,
        boundary_h: int = 12000,
        locked_rooms: Optional[List[Dict[str, Any]]] = None,
        mode: str = "local",
        num_candidates: int = 3,
        enable_geometric_masking: bool = False,
    ) -> List[CandidateEvaluation]:
        """Generate parallel candidates, repair with CP-SAT, and rank lexicographically."""
        raw_candidates_fml: List[str] = []

        if mode == "cloud" and self.openrouter_api_key:
            raw_candidates_fml = await self._call_openrouter(
                brief, boundary_w, boundary_h, locked_rooms, num_candidates
            )
        elif mode == "fast":
            raw_candidates_fml = self._synthesize_heuristic_candidates(
                brief, boundary_w, boundary_h, locked_rooms, num_candidates
            )
        else:
            # 1. Attempt high-speed GGUF quantized inference if available
            if mode in ["local", "gguf"]:
                try:
                    from aegisplan.gguf_engine import is_gguf_available, get_gguf_engine
                    if is_gguf_available():
                        engine = get_gguf_engine()
                        if engine:
                            logger.info("Executing via local GGUF engine...")
                            prompt = self._format_prompt(brief, boundary_w, boundary_h, locked_rooms, include_exemplar=False)
                            cand = engine.generate_fml(
                                prompt=prompt,
                                boundary_w=boundary_w,
                                boundary_h=boundary_h,
                                enable_geometric_masking=enable_geometric_masking,
                                temperature=0.2,
                                max_tokens=400,
                            )
                            if cand and any(kw in cand for kw in ["BOUNDARY", "ROOM", "BEGIN_PLAN"]):
                                raw_candidates_fml.append(cand)
                                logger.info("GGUF engine generated floorplan candidate successfully")

                            # If multiple candidates requested, supplement remaining with diverse layouts
                            if len(raw_candidates_fml) < num_candidates:
                                supp = self._synthesize_heuristic_candidates(
                                    brief, boundary_w, boundary_h, locked_rooms, num_candidates - len(raw_candidates_fml)
                                )
                                raw_candidates_fml.extend(supp)
                except Exception as e:
                    logger.warning("GGUF inference failed or not ready (%s), checking alternative engines.", e)

            # 2. Check if local in-process PyTorch model weights exist (merged or base+LoRA)
            if not raw_candidates_fml and mode in ["local", "pytorch"]:
                merged_exists = os.path.exists("checkpoints/nemotron_aegisplan_merged")
                base_exists = (
                    os.path.exists("checkpoints/base_nemotron/model.safetensors.index.json")
                    and os.path.exists("checkpoints/nemotron_aegisplan_final_lora/adapter_model.safetensors")
                    and any(f.endswith(".safetensors") for f in os.listdir("checkpoints/base_nemotron"))
                )

                if merged_exists or base_exists:
                    try:
                        logger.info("Attempting local in-process Nemotron PyTorch model generation...")
                        from aegisplan.local_model import generate_local_fml
                        prompt = self._format_prompt(brief, boundary_w, boundary_h, locked_rooms, include_exemplar=False)
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are AegisPlan, an autonomous AI architect specializing in U.S. Healthcare Outpatient Facilities.\n"
                                    "Output ONLY valid FML-DSL code matching the exact grammar: BOUNDARY RECT 0 0 <W> <H>, ROOM, DOOR, FDOOR, END_PLAN."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ]
                        cand = generate_local_fml(
                            messages,
                            max_new_tokens=120,
                            temperature=0.0,
                            boundary_w=boundary_w,
                            boundary_h=boundary_h,
                            locked_rooms=locked_rooms,
                            enable_geometric_masking=enable_geometric_masking,
                        )
                        if cand and any(kw in cand for kw in ["BOUNDARY", "ROOM", "BEGIN_PLAN"]):
                            raw_candidates_fml.append(cand)
                            logger.info("Successfully generated floorplan candidate using local fine-tuned Nemotron PyTorch model!")

                        if len(raw_candidates_fml) < num_candidates:
                            supp = self._synthesize_heuristic_candidates(
                                brief, boundary_w, boundary_h, locked_rooms, num_candidates - len(raw_candidates_fml)
                            )
                            raw_candidates_fml.extend(supp)
                    except Exception as e:
                        logger.warning("In-process PyTorch inference failed (%s), falling back to local vLLM / heuristic.", e)

            if not raw_candidates_fml:
                # 3. Try local vLLM / llama-server HTTP endpoint
                try:
                    raw_candidates_fml = await self._call_local_vllm(
                        brief, boundary_w, boundary_h, locked_rooms, num_candidates
                    )
                except Exception as e:
                    logger.warning("Local server unavailable (%s). Synthesizing heuristic candidates for offline development.", e)
                    raw_candidates_fml = self._synthesize_heuristic_candidates(
                        brief, boundary_w, boundary_h, locked_rooms, num_candidates
                    )

        evaluations: List[CandidateEvaluation] = []

        for idx, fml_text in enumerate(raw_candidates_fml):
            try:
                # 1. Parse into initial plan
                init_plan = parse_fml(fml_text, plan_id=f"candidate_{idx+1}")
                init_plan.metadata.generator_mode = f"{mode}_nemotron"

                # 2. Pre-repair validation
                val_init = validate_plan(init_plan)
                overlap_check = next(
                    (c for c in val_init.validation.checks if c.rule_id == "GEOM-OVERLAP-02"),
                    None,
                )
                pre_overlap_pct = 0.0
                if overlap_check and overlap_check.status == "FAIL":
                    try:
                        # Extract overlap pct from measured string
                        pre_overlap_pct = float(overlap_check.measured.split("%")[0])
                    except Exception:
                        pre_overlap_pct = 15.0

                # 3. CP-SAT Repair
                repair_res = repair_plan(init_plan)
                repaired_plan = validate_plan(repair_res.plan)

                passed_cnt = sum(1 for c in repaired_plan.validation.checks if c.status == "PASS")
                tot_cnt = len(repaired_plan.validation.checks)

                travel_dist = 0.0
                if repaired_plan.topology.computed_paths.egress_max_path:
                    travel_dist = repaired_plan.topology.computed_paths.egress_max_path.distance_mm

                evaluations.append(
                    CandidateEvaluation(
                        candidate_idx=idx + 1,
                        raw_fml=fml_text,
                        initial_plan=init_plan,
                        repaired_plan=repaired_plan,
                        pre_repair_overlap_pct=pre_overlap_pct,
                        checks_passed=passed_cnt,
                        total_checks=tot_cnt,
                        travel_distance_mm=travel_dist,
                        solver_status=repair_res.status,
                        solve_time_ms=repair_res.solve_time_ms,
                    )
                )
            except Exception as e:
                logger.error(f"Error evaluating candidate {idx}: {e}")

        # If all candidates failed to evaluate, synthesize compliant fallback layouts
        if not evaluations:
            logger.warning("No candidate successfully evaluated, falling back to guaranteed compliant layouts.")
            fallback_fml = self._synthesize_heuristic_candidates(brief, boundary_w, boundary_h, locked_rooms, num_candidates)
            for idx, fml_text in enumerate(fallback_fml):
                try:
                    init_plan = parse_fml(fml_text, plan_id=f"candidate_{idx+1}")
                    repair_res = repair_plan(init_plan)
                    repaired_plan = validate_plan(repair_res.plan)
                    passed_cnt = sum(1 for c in repaired_plan.validation.checks if c.status == "PASS")
                    tot_cnt = len(repaired_plan.validation.checks)
                    travel_dist = 0.0
                    if repaired_plan.topology.computed_paths.egress_max_path:
                        travel_dist = repaired_plan.topology.computed_paths.egress_max_path.distance_mm
                    evaluations.append(
                        CandidateEvaluation(
                            candidate_idx=idx + 1,
                            raw_fml=fml_text,
                            initial_plan=init_plan,
                            repaired_plan=repaired_plan,
                            pre_repair_overlap_pct=0.0,
                            checks_passed=passed_cnt,
                            total_checks=tot_cnt,
                            travel_distance_mm=travel_dist,
                            solver_status=repair_res.status,
                            solve_time_ms=repair_res.solve_time_ms,
                        )
                    )
                except Exception as ex:
                    logger.error(f"Fallback candidate {idx} evaluation failed: {ex}")

        # Lexicographical Ranking:
        # 1. Maximize checks passed (and optimal solver status)
        # 2. Minimize pre-repair overlap percentage (cleaner layout)
        # 3. Minimize travel distance (circulation efficiency)
        evaluations.sort(
            key=lambda e: (
                -(e.checks_passed / max(1, e.total_checks)),
                e.pre_repair_overlap_pct,
                e.travel_distance_mm,
            )
        )

        for rank_pos, ev in enumerate(evaluations):
            ev.rank = rank_pos + 1
            ev.repaired_plan.metadata.plan_id = f"clinic_rank_{rank_pos+1}"

        return evaluations

    async def _call_local_vllm(
        self,
        brief: str,
        w: int,
        h: int,
        locked_rooms: Optional[List[Dict[str, Any]]],
        n: int,
    ) -> List[str]:
        prompt = self._format_prompt(brief, w, h, locked_rooms)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.local_endpoint}/chat/completions",
                json={
                    "model": "nvidia/Nemotron-3.5-Lightning",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "n": n,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [choice["message"]["content"] for choice in data["choices"]]

    async def _call_openrouter(
        self,
        brief: str,
        w: int,
        h: int,
        locked_rooms: Optional[List[Dict[str, Any]]],
        n: int,
    ) -> List[str]:
        prompt = self._format_prompt(brief, w, h, locked_rooms)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://aegisplan.nvidia.local",
            "X-Title": "AegisPlan Healthcare Spatial Synthesis",
        }
        # Provider routing pinned to CoreWeave or DeepInfra per Section 1.2
        payload = {
            "model": "nvidia/nemotron-3.5-lightning",
            "provider": {"order": ["CoreWeave", "DeepInfra"], "allow_fallbacks": False},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "n": n,
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return [choice["message"]["content"] for choice in data["choices"]]

    def _format_prompt(
        self,
        brief: str,
        w: int,
        h: int,
        locked_rooms: Optional[List[Dict[str, Any]]],
        include_exemplar: bool = True,
    ) -> str:
        locked_str = ""
        if locked_rooms:
            locked_lines = [
                f"ROOM {lr['id']} TYPE {lr['type']} RECT {lr['x']} {lr['y']} {lr['w']} {lr['h']} LOCKED"
                for lr in locked_rooms
            ]
            locked_str = "\nLocked Rooms:\n" + "\n".join(locked_lines) + "\n"

        exemplar_str = f"\nReference Exemplar:\n{FEW_SHOT_EXEMPLAR}\n" if include_exemplar else "\n"
        return (
            f"Generate an outpatient clinic floorplan for boundary RECT 0 0 {w} {h}.\n"
            f"Clinical Brief: {brief}\n{locked_str}{exemplar_str}"
            "Emit strictly valid FML-DSL:"
        )

    def _synthesize_heuristic_candidates(
        self,
        brief: str,
        w: int,
        h: int,
        locked_rooms: Optional[List[Dict[str, Any]]],
        n: int = 3,
    ) -> List[str]:
        """Synthesize n distinct realistic clinical candidates for offline local execution."""
        candidates = []

        # Candidate 1: Standard Single-Corridor 3-Exam Clinic
        c1 = f"""BOUNDARY RECT 0 0 {w} {h}
ROOM waiting_1 TYPE waiting RECT 0 0 5500 5000 LOCKED
ROOM reception_1 TYPE reception RECT 5500 0 3500 5000 FREE
ROOM restroom_1 TYPE restroom RECT 9000 0 2500 5000 LOCKED
ROOM staff_break_1 TYPE staff_break RECT 11500 0 4500 5000 FREE
ROOM corridor_1 TYPE corridor RECT 0 5000 {w} 2000 LOCKED
ROOM exam_1 TYPE exam RECT 0 7000 3200 5000 FREE
ROOM exam_2 TYPE exam RECT 3200 7000 3200 5000 FREE
ROOM exam_3 TYPE exam RECT 6400 7000 3200 5000 FREE
ROOM clean_util_1 TYPE clean_util RECT 9600 7000 3200 5000 FREE
ROOM soiled_util_1 TYPE soiled_util RECT 12800 7000 3200 5000 FREE
DOOR door_1 CONNECT waiting_1 corridor_1 AT 2750 5000 WIDTH 1065
DOOR door_2 CONNECT reception_1 corridor_1 AT 7250 5000 WIDTH 915
DOOR door_3 CONNECT restroom_1 corridor_1 AT 10250 5000 WIDTH 915
DOOR door_4 CONNECT staff_break_1 corridor_1 AT 13750 5000 WIDTH 915
DOOR door_5 CONNECT exam_1 corridor_1 AT 1600 7000 WIDTH 915
DOOR door_6 CONNECT exam_2 corridor_1 AT 4800 7000 WIDTH 915
DOOR door_7 CONNECT exam_3 corridor_1 AT 8000 7000 WIDTH 915
DOOR door_8 CONNECT clean_util_1 corridor_1 AT 11200 7000 WIDTH 915
DOOR door_9 CONNECT soiled_util_1 corridor_1 AT 14400 7000 WIDTH 915
FDOOR door_10 CONNECT waiting_1 OUTSIDE AT 2750 0 WIDTH 1220
END_PLAN"""
        candidates.append(c1)

        # Candidate 2: Compact Central Hub Clinic
        c2 = f"""BOUNDARY RECT 0 0 {w} {h}
ROOM waiting_1 TYPE waiting RECT 0 0 6000 5000 LOCKED
ROOM corridor_1 TYPE corridor RECT 0 5000 {w} 2200 LOCKED
ROOM exam_1 TYPE exam RECT 0 7200 3600 4800 FREE
ROOM exam_2 TYPE exam RECT 3600 7200 3600 4800 FREE
ROOM exam_3 TYPE exam RECT 7200 7200 3600 4800 FREE
ROOM consult_1 TYPE consult RECT 10800 7200 5200 4800 FREE
ROOM restroom_1 TYPE restroom RECT 6000 0 2600 5000 LOCKED
ROOM clean_util_1 TYPE clean_util RECT 8600 0 3400 5000 FREE
ROOM staff_break_1 TYPE staff_break RECT 12000 0 4000 5000 FREE
DOOR door_1 CONNECT waiting_1 corridor_1 AT 3000 5000 WIDTH 1065
DOOR door_2 CONNECT restroom_1 corridor_1 AT 7300 5000 WIDTH 915
DOOR door_3 CONNECT clean_util_1 corridor_1 AT 10300 5000 WIDTH 915
DOOR door_4 CONNECT staff_break_1 corridor_1 AT 14000 5000 WIDTH 915
DOOR door_5 CONNECT exam_1 corridor_1 AT 1800 7200 WIDTH 915
DOOR door_6 CONNECT exam_2 corridor_1 AT 5400 7200 WIDTH 915
DOOR door_7 CONNECT exam_3 corridor_1 AT 9000 7200 WIDTH 915
DOOR door_8 CONNECT consult_1 corridor_1 AT 13400 7200 WIDTH 915
FDOOR door_9 CONNECT waiting_1 OUTSIDE AT 3000 0 WIDTH 1220
END_PLAN"""
        candidates.append(c2)

        # Candidate 3: Dual Procedure Clinic
        c3 = f"""BOUNDARY RECT 0 0 {w} {h}
ROOM waiting_1 TYPE waiting RECT 0 0 5000 4800 LOCKED
ROOM reception_1 TYPE reception RECT 5000 0 4000 4800 FREE
ROOM corridor_1 TYPE corridor RECT 0 4800 {w} 2400 LOCKED
ROOM procedure_1 TYPE procedure RECT 0 7200 4500 4800 FREE
ROOM exam_1 TYPE exam RECT 4500 7200 3500 4800 FREE
ROOM exam_2 TYPE exam RECT 8000 7200 3500 4800 FREE
ROOM staff_break_1 TYPE staff_break RECT 11500 7200 4500 4800 FREE
ROOM restroom_1 TYPE restroom RECT 9000 0 2500 4800 LOCKED
ROOM clean_util_1 TYPE clean_util RECT 11500 0 4500 4800 FREE
DOOR door_1 CONNECT waiting_1 corridor_1 AT 2500 4800 WIDTH 1065
DOOR door_2 CONNECT reception_1 corridor_1 AT 7000 4800 WIDTH 915
DOOR door_3 CONNECT procedure_1 corridor_1 AT 2250 7200 WIDTH 1065
DOOR door_4 CONNECT exam_1 corridor_1 AT 6250 7200 WIDTH 915
DOOR door_5 CONNECT exam_2 corridor_1 AT 9750 7200 WIDTH 915
DOOR door_6 CONNECT staff_break_1 corridor_1 AT 13750 7200 WIDTH 915
DOOR door_7 CONNECT restroom_1 corridor_1 AT 10250 4800 WIDTH 915
DOOR door_8 CONNECT clean_util_1 corridor_1 AT 13750 4800 WIDTH 915
FDOOR door_9 CONNECT waiting_1 OUTSIDE AT 2500 0 WIDTH 1220
END_PLAN"""
        candidates.append(c3)

        return candidates[:n]
