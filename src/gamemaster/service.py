"""REST API for deployment.

    uvicorn gamemaster.service:app --host 0.0.0.0 --port 8081

POST /v1/act   {"game": "connect4", "state": {...}, "time_budget": 2}  -> {"action": "3", ...}
POST /v1/act   {"rules": "...", "observation": "...", "legal_actions": [...]}
GET  /v1/games
GET  /healthz
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from academy_core import setup_logging
from gamemaster.games import get_game, list_games
from gamemaster.policy import decide

setup_logging()
_KEYS = {k.strip() for k in os.getenv("GAMEMASTER_API_KEYS", "").split(",") if k.strip()}
_SEM = asyncio.Semaphore(int(os.getenv("GAMEMASTER_MAX_CONCURRENT_REQUESTS", str(os.cpu_count() or 4))))

app = FastAPI(title="GameMaster", version="0.1.0")


class ActRequest(BaseModel):
    game: str | None = None
    state: dict[str, Any] | None = None
    rules: str | None = None
    observation: Any = None
    legal_actions: list[str] | None = None
    history: list[str] | None = None
    time_budget: float = Field(default=2.0, gt=0, le=60)
    agent: str | None = None


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if _KEYS and x_api_key not in _KEYS:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/games")
async def games() -> dict[str, Any]:
    return {name: {"players": (g := get_game(name)).num_players, "deterministic": g.deterministic, "rules": g.rules}
            for name in list_games()}


@app.post("/v1/act", dependencies=[Depends(require_key)])
async def act(req: ActRequest) -> dict[str, Any]:
    payload = req.model_dump(exclude_none=True)
    async with _SEM:  # search is CPU-bound: run in a worker thread, bounded
        try:
            result = await asyncio.to_thread(decide, payload, req.time_budget, req.agent)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result.get("action") is None:
        raise HTTPException(status_code=422, detail=result.get("error", "no action"))
    return result
