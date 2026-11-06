import json
from types import SimpleNamespace

import pytest

from config import MOCK_ROLE_SPECS
from manip_tester import (
    DEFAULT_TACTICS, AttackerAgent, EvaluatorAgent, OpenAICompatibleBackend, PlannerAgent,
    TargetAgent, TraceLogger, make_backend, run_episode,
)
from scenarios import GOALS, POLICY


def build_agents():
    b = {r: make_backend(s) for r, s in MOCK_ROLE_SPECS.items()}
    return dict(
        planner=PlannerAgent(b["planner"], DEFAULT_TACTICS),
        attacker=AttackerAgent(b["attacker"]),
        target=TargetAgent(b["target"], POLICY),
        evaluator=EvaluatorAgent(b["evaluator"]),
    )


def test_episode_stops_on_violation_and_logs(tmp_path):
    with TraceLogger(tmp_path, echo=False) as log:
        res = run_episode(episode_id="t-01", goal=GOALS["refund_outside_window"], policy=POLICY,
                          max_turns=5, logger=log, **build_agents())
    assert res.violated and res.first_violation_turn == 3 and res.turns_used == 3
    assert res.tactics_used == ["authority", "social_proof", "scarcity_urgency"]
    events = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in events] == ["episode_start", "turn", "turn", "turn", "episode_end"]
    assert "[Support bot]" in (tmp_path / "transcript.log").read_text(encoding="utf-8")


def test_episode_respects_max_turns():
    res = run_episode(episode_id="t-02", goal=GOALS["refund_outside_window"], policy=POLICY,
                      max_turns=2, **build_agents())
    assert not res.violated and res.turns_used == 2 and res.first_violation_turn is None


def _backend_with(monkeypatch, fake_create):
    pytest.importorskip("openai")
    monkeypatch.setattr("time.sleep", lambda s: None)
    b = OpenAICompatibleBackend(model="m", api_key="k")
    monkeypatch.setattr(b._client.chat.completions, "create", fake_create)
    return b


class FakeErr(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def test_retries_on_429_then_succeeds_and_counts_tokens(monkeypatch):
    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeErr(429)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    b = _backend_with(monkeypatch, fake_create)
    assert b.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls["n"] == 3
    assert b.usage == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5}


def test_no_retry_on_bad_key(monkeypatch):
    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        raise FakeErr(401)

    b = _backend_with(monkeypatch, fake_create)
    with pytest.raises(RuntimeError):
        b.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1
