"""Remote environment adapter for games served over HTTP (rules hidden, no simulator).

Expected protocol (adapt ``_parse`` when the official API is published):
    POST {base}/reset  {"seed": int|null}   -> observation
    POST {base}/step   {"action": str}      -> observation
    observation = {"observation"|"text"|"board": ..., "legal_actions": [...], "reward": float, "done": bool}
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from gamemaster.core import Observation


class HttpEnv:
    def __init__(self, base_url: str, timeout: float = 30.0, rules: str = ""):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.rules = rules

    @staticmethod
    def _parse(data: dict[str, Any]) -> Observation:
        raw = data.get("observation", data.get("text", data.get("board", data.get("state", ""))))
        text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True)
        legal = [str(a) for a in data.get("legal_actions", data.get("actions", data.get("available_actions", [])))]
        return Observation(text=text, legal_actions=legal, reward=float(data.get("reward", 0.0) or 0.0),
                           done=bool(data.get("done", data.get("terminal", False))), info=data, key=text)

    def reset(self, seed: int | None = None) -> Observation:
        resp = self.client.post(f"{self.base_url}/reset", json={"seed": seed})
        resp.raise_for_status()
        data = resp.json()
        self.rules = data.get("rules", self.rules)
        return self._parse(data)

    def step(self, action: str) -> Observation:
        resp = self.client.post(f"{self.base_url}/step", json={"action": action})
        resp.raise_for_status()
        return self._parse(resp.json())
