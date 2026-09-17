import json

import httpx
import pytest

from gamemaster import cli
from gamemaster.agents.llm_policy import LLMPolicyAgent, SyncLLM
from gamemaster.games import get_game
from gamemaster.policy import decide
from gamemaster.prompts import messages_for_state, parse_move
from gamemaster.training.datagen import generate, load_records, split
from gamemaster.training.rewards import format_reward, make_engine_reward, teacher_reward


def test_parse_move():
    legal = ["up", "down", "left", "right"]
    assert parse_move("left", legal) == "left"
    assert parse_move("<think>hmm</think> Right.", legal) == "right"
    assert parse_move("I will go up", legal) == "up"
    assert parse_move("up or down", legal) is None
    assert parse_move("3", ["0", "3", "6"]) == "3"


def test_datagen_records_and_split(tmp_path):
    out = tmp_path / "kd.jsonl"
    stats = generate("keydoor", "planner", episodes=6, out=out, workers=1, epsilon=0.0)
    assert stats["records"] > 10
    records = load_records(out)
    r = records[0]
    assert r["completion"] in r["legal"] and r["messages"][0]["role"] == "system"
    assert get_game("keydoor").from_json(r["state"])
    train, evals = split(records, 0.2)
    assert {x["seed"] for x in train}.isdisjoint({x["seed"] for x in evals})


def test_rewards():
    legal = [["0", "3", "6"]] * 4
    completions = ["3", "I pick 6 because", "9", [{"role": "assistant", "content": "0"}]]
    assert format_reward(completions, legal) == [0.2, 0.0, -1.0, 0.2]
    policy = [{"0": 0.2, "3": 0.6, "6": 0.2}] * 4
    assert teacher_reward(completions, legal, policy) == pytest.approx([1.0, 1 / 3, 0.0, 1 / 3])

    game = get_game("connect4")
    s = game.initial_state()
    for col in [3, 0, 3, 0, 3]:
        s = game.apply(s, col)
    reward = make_engine_reward("connect4")
    moves = [game.action_to_str(s, a) for a in game.legal_actions(s)]
    scores = reward(["3", "5"], [moves, moves], [game.to_json(s)] * 2)
    assert scores[0] > scores[1]


def _mock_llm(content: str, top_logprobs=None) -> SyncLLM:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        body = json.loads(request.content)
        assert "structured_outputs" in body or body.get("max_tokens") == 1
        choice = {"message": {"content": content}}
        if top_logprobs:
            choice["logprobs"] = {"content": [{"top_logprobs": top_logprobs}]}
        return httpx.Response(200, json={"choices": [choice]})

    llm = SyncLLM(base_url="http://llm/v1")
    llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def test_llm_policy_constrained_and_priors():
    game = get_game("2048")
    s = game.initial_state(0)
    legal = game.legal_actions(s)
    agent = LLMPolicyAgent(llm=_mock_llm(legal[0]))
    assert agent.act(game, s) == legal[0]
    # an illegal answer must fall back, never be played
    bad = LLMPolicyAgent(llm=_mock_llm("jump"))
    assert bad.act(game, s) in legal and bad.stats["fallback_moves"] == 1
    tops = [{"token": a[:2], "logprob": -0.1 * i} for i, a in enumerate(legal)]
    dist = LLMPolicyAgent(llm=_mock_llm("", tops)).distribution(game, s, legal)
    assert set(dist) == set(legal)


def test_prompt_contains_rules_board_and_moves():
    game = get_game("connect4")
    messages, legal = messages_for_state(game, game.initial_state())
    assert "Connect Four" in messages[0]["content"]
    assert all(m in messages[1]["content"] for m in legal)


def test_decide_known_and_unknown_games(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "false")
    game = get_game("keydoor")
    s = game.initial_state(3)
    res = decide({"game": "keydoor", "state": game.to_json(s)}, time_budget=2)
    assert res["action"] in [game.action_to_str(s, a) for a in game.legal_actions(s)]
    assert res["source"] == "planner"
    res = decide({"rules": "secret", "observation": "???", "legal_actions": ["a", "b"]}, time_budget=1)
    assert res["action"] in {"a", "b"} and res["source"] == "fallback"


def test_cli_harness_writes_output(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "false")
    game = get_game("connect4")
    s = game.initial_state()
    for col in [3, 0, 3, 0, 3]:
        s = game.apply(s, col)
    inp = tmp_path / "state_01.json"
    inp.write_text(json.dumps({"game": "connect4", "state": game.to_json(s)}))
    assert cli.main(["--input-state", str(inp), "--output-dir", str(tmp_path), "--time-budget", "1.5"]) == 0
    out = json.loads((tmp_path / "state_01_output.json").read_text())
    assert out["action"] == "3"
