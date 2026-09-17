import random

import pytest

from gamemaster.agents import ExplorerAgent, make_agent
from gamemaster.core import GameEnv
from gamemaster.eval import evaluate, play_episode
from gamemaster.games import get_game, list_games
from gamemaster.games.game2048 import slide


@pytest.mark.parametrize("name", ["connect4", "2048", "keydoor"])
def test_random_playouts_respect_rules_and_json_roundtrip(name):
    game = get_game(name)
    rng = random.Random(0)
    for seed in range(3):
        s = game.initial_state(seed)
        steps = 0
        while not game.is_terminal(s) and steps < 400:
            assert game.from_json(game.to_json(s)) == s
            legal = game.legal_actions(s)
            assert legal
            for a in legal:
                assert game.str_to_action(s, game.action_to_str(s, a)) == a
            s = game.apply(s, rng.choice(legal), rng)
            steps += 1
        assert isinstance(game.render(s), str)
        assert len(game.returns(s)) == game.num_players


def test_connect4_win_detection():
    game = get_game("connect4")
    s = game.initial_state()
    for col in [0, 1, 0, 1, 0, 1]:
        s = game.apply(s, col)
    assert not game.is_terminal(s)
    s = game.apply(s, 0)  # four in column 0 for X
    assert game.is_terminal(s) and game.returns(s) == [1.0, -1.0]
    with pytest.raises(ValueError):
        game.apply(s, 3)


def test_2048_slide_rules():
    board = (2, 2, 4, 4, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0)
    left, gained = slide(board, "left")
    assert left[:4] == (4, 8, 0, 0) and left[8:12] == (4, 2, 0, 0) and gained == 16
    right, _ = slide(board, "right")
    assert right[:4] == (0, 0, 4, 8) and right[8:12] == (0, 0, 2, 4)


def test_mcts_takes_immediate_win_and_blocks():
    game = get_game("connect4")
    s = game.initial_state()
    for col in [3, 0, 3, 0, 3]:  # X threatens to win in column 3; O to move must block
        s = game.apply(s, col)
    agent = make_agent("mcts:iterations=400", seed=1)
    assert agent.act(game, s) == 3
    s2 = game.apply(game.apply(game.initial_state(), 0), 6)
    for col in [0, 6, 0, 6]:
        s2 = game.apply(s2, col)
    assert make_agent("mcts:iterations=300", seed=2).act(game, s2) == 0  # X completes column 0


def test_mcts_beats_random_connect4():
    report = evaluate(get_game("connect4"), ["mcts:iterations=200", "random"], episodes=4, seed=3)
    assert report["agents"]["mcts:iterations=200"]["win_rate"] >= 0.75
    assert report["illegal_moves"] == 0


def test_mcts_handles_stochastic_open_loop_2048():
    game = get_game("2048")
    agent = make_agent("mcts:iterations=40,rollout=none", seed=0)
    rng = random.Random(1)
    s = game.initial_state(1)
    for _ in range(30):  # chance changes legal moves between simulations; must never raise or go illegal
        action = agent.act(game, s)
        assert action in game.legal_actions(s)
        s = game.apply(s, action, rng)
    assert game.returns(s)[0] > 0


def test_planner_is_optimal_on_keydoor():
    game = get_game("keydoor")
    for seed in range(5):
        best = game.shortest_solution(game.initial_state(seed))
        res = play_episode(game, [make_agent("planner")], seed)
        assert res.returns[0] > 0 and res.steps == best


def test_explorer_learns_black_box_level():
    """With no simulator and no rules, the explorer solves a level and then replays it efficiently."""
    env = GameEnv(get_game("keydoor"))
    agent = ExplorerAgent(seed=0)
    results = [agent.run_episode(env, seed=7, max_steps=120) for _ in range(6)]
    solved = [r for r in results if r["return"] > 0]
    assert solved, results
    assert results[-1]["return"] > 0
    assert results[-1]["steps"] <= min(r["steps"] for r in results)


def test_registry_lists_games():
    assert {"connect4", "2048", "keydoor"} <= set(list_games())
    with pytest.raises(KeyError):
        get_game("nope")
