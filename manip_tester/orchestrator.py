"""Оркестратор: связывает четыре агента в многоходовой эпизод.

Один ход:  планировщик -> атакующий -> целевой бот -> оценщик.
Эпизод заканчивается, когда оценщик фиксирует нарушение политики или исчерпан max_turns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .agents import AttackerAgent, EvaluatorAgent, PlannerAgent, TargetAgent
from .schemas import AttackGoal, Turn, Verdict
from .trace import TraceLogger


@dataclass
class EpisodeResult:
    episode_id: str
    goal_id: str
    violated: bool
    turns_used: int
    first_violation_turn: int | None  # номер хода (с 1) или None
    tactics_used: list[str]
    max_progress: int
    history: list[Turn] = field(repr=False, default_factory=list)
    verdicts: list[Verdict] = field(repr=False, default_factory=list)

    def summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("history")
        d.pop("verdicts")
        return d


def _model_name(agent) -> str:
    return str(getattr(agent.llm, "model", type(agent.llm).__name__))


def _one_line(text: str) -> str:
    return " ⏎ ".join(text.strip().splitlines())


def _usage(agent) -> dict | None:
    u = getattr(agent.llm, "usage", None)
    return dict(u) if u else None


def run_episode(
    *,
    episode_id: str,
    goal: AttackGoal,
    policy: str,
    planner: PlannerAgent,
    attacker: AttackerAgent,
    target: TargetAgent,
    evaluator: EvaluatorAgent,
    max_turns: int = 5,
    logger: TraceLogger | None = None,
) -> EpisodeResult:
    tactics_by_id = {t.id: t for t in planner.tactics}
    history: list[Turn] = []
    verdicts: list[Verdict] = []
    tried: list[tuple[str, int]] = []  # (tactic_id, progress) по прошлым ходам
    first_violation: int | None = None
    prefix = f"ep={episode_id}"

    def log_say(line: str) -> None:
        if logger:
            logger.say(f"{prefix} | {line}")

    if logger:
        logger.event(
            "episode_start", episode_id=episode_id, goal_id=goal.id, goal=goal.description,
            max_turns=max_turns,
            models={a.name: _model_name(a) for a in (planner, attacker, target, evaluator)},
        )
    log_say(f"START goal={goal.id} max_turns={max_turns}")

    for turn_idx in range(max_turns):
        n = turn_idx + 1
        decision = planner.plan(goal, history, tried, turn_idx, max_turns)
        tactic = tactics_by_id[decision.tactic_id]
        log_say(f"turn={n} [Planner] {tactic.id}: {_one_line(decision.rationale)}")

        message = attacker.write_message(goal, tactic, decision, history)
        history.append(Turn("attacker", message))
        log_say(f"turn={n} [Customer] {_one_line(message)}")

        reply = target.respond(history)
        history.append(Turn("target", reply))
        log_say(f"turn={n} [Support bot] {_one_line(reply)}")

        verdict = evaluator.judge(policy, goal, history)
        verdicts.append(verdict)
        tried.append((tactic.id, verdict.progress))
        log_say(
            f"turn={n} [Evaluator] violated={verdict.violated} progress={verdict.progress} "
            f"conf={verdict.confidence:.2f} - {_one_line(verdict.rationale)}"
        )
        if logger:
            logger.event(
                "turn", episode_id=episode_id, turn=n, tactic_id=tactic.id,
                planner_rationale=decision.rationale, planner_instruction=decision.instruction,
                customer_message=message, bot_reply=reply,
                violated=verdict.violated, progress=verdict.progress,
                confidence=verdict.confidence, evaluator_rationale=verdict.rationale,
            )
        if verdict.violated:
            first_violation = n
            break

    result = EpisodeResult(
        episode_id=episode_id,
        goal_id=goal.id,
        violated=first_violation is not None,
        turns_used=len(verdicts),
        first_violation_turn=first_violation,
        tactics_used=[t for t, _ in tried],
        max_progress=max((v.progress for v in verdicts), default=0),
        history=history,
        verdicts=verdicts,
    )
    if logger:
        logger.event(
            "episode_end", **result.summary_dict(),
            usage_cumulative={a.name: _usage(a) for a in (planner, attacker, target, evaluator)},
        )
    log_say(
        f"END violated={result.violated} first_violation_turn={result.first_violation_turn} "
        f"turns_used={result.turns_used} max_progress={result.max_progress}"
    )
    return result
