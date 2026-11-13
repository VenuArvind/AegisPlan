"""FastAPI Server for AegisPlan: Neuro-Symbolic Healthcare Spatial Synthesis."""

from typing import List, Dict, Any, Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aegisplan.schemas import PlanDocument
from aegisplan.fml_parser import parse_fml, serialize_to_fml
from aegisplan.validator import validate_plan
from aegisplan.cpsat_repair import repair_plan
from aegisplan.router import GenerationRouter, FEW_SHOT_EXEMPLAR
from aegisplan.streamer import stream_plan_generation

app = FastAPI(
    title="AegisPlan API",
    description="Neuro-Symbolic Healthcare Spatial Synthesis Engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = GenerationRouter()


class GenerateRequest(BaseModel):
    brief: str
    boundary_w: int = 16000
    boundary_h: int = 12000
    locked_rooms: Optional[List[Dict[str, Any]]] = None
    mode: str = "local"  # "local" | "cloud"
    num_candidates: int = 3
    enable_geometric_masking: bool = False


class RepairRequest(BaseModel):
    plan: Optional[PlanDocument] = None
    fml_text: Optional[str] = None
    enforce_door_adjacencies: bool = True
    max_time_seconds: float = 2.0


class EditRoomRequest(BaseModel):
    plan: PlanDocument
    room_id: str
    delta_w: int = 0
    delta_h: int = 0
    toggle_lock: Optional[bool] = None


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "AegisPlan",
        "version": "0.1.0",
        "engine": "CP-SAT + Nemotron 3.5 Lightning",
    }


@app.get("/api/sample")
async def get_sample_plan():
    """Return the baseline validated outpatient clinic plan."""
    plan = parse_fml(FEW_SHOT_EXEMPLAR, plan_id="sample_clinic")
    plan = validate_plan(plan)
    return plan


@app.post("/api/generate")
async def generate_plans(req: GenerateRequest):
    """Generate candidates, repair with CP-SAT, and rank lexicographically."""
    try:
        candidates = await router.generate_candidates(
            brief=req.brief,
            boundary_w=req.boundary_w,
            boundary_h=req.boundary_h,
            locked_rooms=req.locked_rooms,
            mode=req.mode,
            num_candidates=req.num_candidates,
            enable_geometric_masking=req.enable_geometric_masking,
        )
        return {
            "mode": req.mode,
            "privacy_status": (
                "100% Confidential (Zero briefs leave workstation)"
                if req.mode == "local"
                else "Cloud-Burst (Pinned CoreWeave/DeepInfra)"
            ),
            "candidates": [
                {
                    "rank": c.rank,
                    "candidate_idx": c.candidate_idx,
                    "plan": c.repaired_plan,
                    "solver_status": c.solver_status,
                    "solve_time_ms": c.solve_time_ms,
                    "pre_repair_overlap_pct": c.pre_repair_overlap_pct,
                    "checks_passed": c.checks_passed,
                    "total_checks": c.total_checks,
                    "travel_distance_mm": c.travel_distance_mm,
                    "raw_fml": c.raw_fml,
                }
                for c in candidates
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/stream")
async def generate_plans_stream(req: GenerateRequest):
    """
    Asynchronous non-blocking Server-Sent Events (SSE) streaming endpoint.
    Streams incremental synthesis milestones: brief verification, candidate drafting,
    CP-SAT variable solving, building code checks, and final lexicographical rankings.
    """
    return StreamingResponse(
        stream_plan_generation(
            router=router,
            brief=req.brief,
            boundary_w=req.boundary_w,
            boundary_h=req.boundary_h,
            locked_rooms=req.locked_rooms,
            mode=req.mode,
            num_candidates=req.num_candidates,
            enable_geometric_masking=req.enable_geometric_masking,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/validate")
async def validate_endpoint(plan: PlanDocument):
    """Validate a plan.json against ADA, FGI, and IBC codes."""
    return validate_plan(plan)


@app.post("/api/repair")
async def repair_endpoint(req: RepairRequest):
    """Execute CP-SAT combinatorial geometry repair on a plan or raw FML."""
    if req.plan:
        plan = req.plan
    elif req.fml_text:
        plan = parse_fml(req.fml_text)
    else:
        raise HTTPException(status_code=400, detail="Must provide plan or fml_text")

    result = repair_plan(
        plan,
        max_time_seconds=req.max_time_seconds,
        enforce_door_adjacencies=req.enforce_door_adjacencies,
    )
    validated = validate_plan(result.plan)
    return {
        "status": result.status,
        "solve_time_ms": result.solve_time_ms,
        "conflict_core": result.conflict_core,
        "plan": validated,
    }


@app.post("/api/edit")
async def edit_room(req: EditRoomRequest):
    """Interactive room modification with immediate sub-50ms CP-SAT re-solve."""
    plan = req.plan
    target_room = next((r for r in plan.rooms if r.id == req.room_id), None)
    if not target_room:
        raise HTTPException(status_code=404, detail=f"Room {req.room_id} not found")

    if req.toggle_lock is not None:
        target_room.locked = req.toggle_lock

    target_room.bbox.w = max(1500, target_room.bbox.w + req.delta_w)
    target_room.bbox.h = max(1500, target_room.bbox.h + req.delta_h)
    target_room.polygon = [
        [target_room.bbox.x, target_room.bbox.y],
        [target_room.bbox.x + target_room.bbox.w, target_room.bbox.y],
        [target_room.bbox.x + target_room.bbox.w, target_room.bbox.y + target_room.bbox.h],
        [target_room.bbox.x, target_room.bbox.y + target_room.bbox.h],
    ]
    target_room.area_actual_m2 = round((target_room.bbox.w * target_room.bbox.h) / 1_000_000.0, 2)

    # Re-run CP-SAT to resolve any overlaps caused by the edit
    res = repair_plan(plan, max_time_seconds=1.0)
    validated = validate_plan(res.plan)
    return {
        "status": res.status,
        "solve_time_ms": res.solve_time_ms,
        "conflict_core": res.conflict_core,
        "plan": validated,
    }


# Static assets directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
    return HTMLResponse("<h1>AegisPlan API running</h1>")
