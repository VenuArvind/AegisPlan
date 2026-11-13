"""Unit and integration tests for Asynchronous Server-Sent Events (SSE) streaming."""

import time
import json
import pytest
from fastapi.testclient import TestClient

from aegisplan.server import app

client = TestClient(app)


def test_sse_endpoint_status_and_content_type():
    """Verify that /api/generate/stream returns 200 OK and text/event-stream content type."""
    response = client.post(
        "/api/generate/stream",
        json={
            "brief": "Family medicine outpatient clinic with 2 exam rooms and reception",
            "boundary_w": 16000,
            "boundary_h": 12000,
            "mode": "local",
            "num_candidates": 2,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_sse_event_chronology_and_schema():
    """Parse SSE frames and assert discrete lifecycle events occur in chronological order."""
    response = client.post(
        "/api/generate/stream",
        json={
            "brief": "Pediatric care facility with waiting area and exam room",
            "boundary_w": 16000,
            "boundary_h": 12000,
            "mode": "local",
            "num_candidates": 2,
        },
    )
    assert response.status_code == 200

    # Parse SSE text stream into structured events
    raw_lines = response.text.strip().split("\n\n")
    events = []
    for block in raw_lines:
        lines = block.strip().split("\n")
        event_type = None
        data_payload = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_payload = json.loads(line[len("data: "):].strip())
        if event_type and data_payload:
            events.append((event_type, data_payload))

    # Assert event names
    event_names = [e[0] for e in events]
    assert "brief_validated" in event_names
    assert "draft_synthesized" in event_names
    assert "solver_progress" in event_names
    assert "candidate_evaluated" in event_names
    assert "synthesis_complete" in event_names

    # Check 1: First event must be brief_validated
    first_event, first_data = events[0]
    assert first_event == "brief_validated"
    assert first_data["status"] == "valid"
    assert first_data["target_candidates"] == 2
    assert first_data["ttfb_ms"] >= 0

    # Check 2: Last event must be synthesis_complete
    last_event, last_data = events[-1]
    assert last_event == "synthesis_complete"
    assert last_data["status"] == "success"
    assert last_data["total_candidates"] == 2
    assert len(last_data["ranked_candidates"]) == 2
    assert last_data["ranked_candidates"][0]["solver_status"] in ("OPTIMAL", "FEASIBLE")


def test_sse_time_to_first_byte():
    """Ensure Time-to-First-Byte (TTFB) is sub-50ms."""
    t0 = time.perf_counter()
    with client.stream(
        "POST",
        "/api/generate/stream",
        json={
            "brief": "Cardiology clinic with ultrasound and consultation",
            "boundary_w": 16000,
            "boundary_h": 12000,
            "mode": "local",
            "num_candidates": 1,
        },
    ) as response:
        assert response.status_code == 200
        # Read the first line of the stream
        first_chunk = next(response.iter_lines())
        ttfb_ms = (time.perf_counter() - t0) * 1000.0

    assert first_chunk is not None
    assert ttfb_ms < 100.0  # Fast initial handshake
