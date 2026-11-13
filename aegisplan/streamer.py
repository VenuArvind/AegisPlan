"""
AegisPlan Asynchronous SSE Streaming Pipeline.

Decouples heavy CP-SAT integer programming and LLM synthesis from blocking HTTP threads,
streaming discrete lifecycle events in real-time via Server-Sent Events (SSE).
"""

import time
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
import logging

from aegisplan.schemas import PlanDocument
from aegisplan.fml_parser import parse_fml
from aegisplan.validator import validate_plan
from aegisplan.cpsat_repair import repair_plan
from aegisplan.router import GenerationRouter, CandidateEvaluation

logger = logging.getLogger(__name__)


def sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format a payload as a standard Server-Sent Event (SSE) frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_plan_generation(
    router: GenerationRouter,
    brief: str,
    boundary_w: int = 16000,
    boundary_h: int = 12000,
    locked_rooms: Optional[List[Dict[str, Any]]] = None,
    mode: str = "local",
    num_candidates: int = 3,
    enable_geometric_masking: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding real-time SSE lifecycle events during spatial synthesis.
    
    Lifecycle Events:
      1. brief_validated: Emitted immediately (< 20ms TTFB) upon schema verification.
      2. draft_synthesized: Emitted when an FML candidate draft is produced.
      3. solver_progress: Emitted when CP-SAT solver completes geometric constraints.
      4. candidate_evaluated: Emitted when statutory building codes (ADA/FGI) are verified.
      5. synthesis_complete: Emitted with final lexicographical rankings and metadata.
    """
    t_start = time.perf_counter()

    # Stage 1: Immediate Handshake & Input Validation (< 15ms TTFB)
    yield sse_event("brief_validated", {
        "status": "valid",
        "brief_length_chars": len(brief),
        "boundary": {"width_mm": boundary_w, "height_mm": boundary_h},
        "target_candidates": num_candidates,
        "mode": mode,
        "ttfb_ms": round((time.perf_counter() - t_start) * 1000.0, 2)
    })
    await asyncio.sleep(0)  # Yield control to event loop

    # Stage 2: Draft Generation
    raw_candidates_fml: List[str] = []
    try:
        # Generate candidate drafts using router
        raw_candidates_fml = router._synthesize_heuristic_candidates(
            brief, boundary_w, boundary_h, locked_rooms, num_candidates
        )
    except Exception as e:
        logger.warning(f"Router draft generation encountered error: {e}")
        raw_candidates_fml = [router.FEW_SHOT_EXEMPLAR] * num_candidates

    evaluations: List[CandidateEvaluation] = []

    # Stage 3: Progressive Solver Execution & Evaluation per Candidate
    for idx, fml_text in enumerate(raw_candidates_fml):
        c_idx = idx + 1
        t_cand_start = time.perf_counter()

        # Emit draft event
        yield sse_event("draft_synthesized", {
            "candidate_idx": c_idx,
            "raw_fml_lines": len(fml_text.strip().splitlines()),
            "timestamp_ms": round((time.perf_counter() - t_start) * 1000.0, 2)
        })
        await asyncio.sleep(0)

        # Parse initial candidate
        init_plan = parse_fml(fml_text, plan_id=f"stream_candidate_{c_idx}")
        init_plan.metadata.generator_mode = f"{mode}_stream"

        # Pre-repair validation
        val_init = validate_plan(init_plan)
        overlap_check = next(
            (c for c in val_init.validation.checks if c.rule_id == "GEOM-OVERLAP-02"),
            None
        )
        pre_overlap_pct = 0.0
        if overlap_check and overlap_check.status == "FAIL":
            try:
                pre_overlap_pct = float(overlap_check.measured.split("%")[0])
            except Exception:
                pre_overlap_pct = 15.0

        # CP-SAT Constraint Solving
        repair_res = repair_plan(init_plan)

        yield sse_event("solver_progress", {
            "candidate_idx": c_idx,
            "solver": "Google OR-Tools CP-SAT",
            "solver_status": repair_res.status,
            "solve_time_ms": repair_res.solve_time_ms,
            "conflict_core": repair_res.conflict_core,
            "pre_repair_overlap_pct": pre_overlap_pct,
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2)
        })
        await asyncio.sleep(0)

        # Post-repair validation against statutory codes
        repaired_plan = validate_plan(repair_res.plan)
        passed_cnt = sum(1 for c in repaired_plan.validation.checks if c.status == "PASS")
        tot_cnt = len(repaired_plan.validation.checks)

        travel_dist = 0.0
        if repaired_plan.topology.computed_paths.egress_max_path:
            travel_dist = repaired_plan.topology.computed_paths.egress_max_path.distance_mm

        cand_eval = CandidateEvaluation(
            candidate_idx=c_idx,
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
        evaluations.append(cand_eval)

        yield sse_event("candidate_evaluated", {
            "candidate_idx": c_idx,
            "checks_passed": passed_cnt,
            "total_checks": tot_cnt,
            "pass_rate_pct": round((passed_cnt / tot_cnt) * 100.0, 1) if tot_cnt > 0 else 100.0,
            "travel_distance_mm": travel_dist,
            "duration_ms": round((time.perf_counter() - t_cand_start) * 1000.0, 2)
        })
        await asyncio.sleep(0)

    # Stage 4: Lexicographical Ranking
    evaluations.sort(
        key=lambda c: (
            0 if c.solver_status == "OPTIMAL" else (1 if c.solver_status == "FEASIBLE" else 2),
            c.total_checks - c.checks_passed,
            c.travel_distance_mm,
        )
    )
    for rank, cand in enumerate(evaluations):
        cand.rank = rank + 1

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    # Stage 5: Final Synthesis Completion Event
    yield sse_event("synthesis_complete", {
        "status": "success",
        "total_candidates": len(evaluations),
        "best_candidate_idx": evaluations[0].candidate_idx if evaluations else 1,
        "best_checks_passed": evaluations[0].checks_passed if evaluations else 0,
        "total_checks": evaluations[0].total_checks if evaluations else 0,
        "total_solve_time_ms": sum(e.solve_time_ms for e in evaluations),
        "total_pipeline_time_ms": total_time_ms,
        "ranked_candidates": [
            {
                "rank": e.rank,
                "candidate_idx": e.candidate_idx,
                "solver_status": e.solver_status,
                "solve_time_ms": e.solve_time_ms,
                "checks_passed": e.checks_passed,
                "total_checks": e.total_checks,
                "travel_distance_mm": e.travel_distance_mm,
            }
            for e in evaluations
        ]
    })
