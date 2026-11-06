"""Многоходовые прогоны атак с логированием.

Примеры:
  python run_episode.py --mock                          # без сети, на заглушках
  python run_episode.py --goal refund_outside_window    # один эпизод на реальных моделях
  python run_episode.py --goal all --episodes 3 --max-turns 5
Результаты: runs/<дата_время>/trace.jsonl и transcript.log
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime

from config import MOCK_ROLE_SPECS, ROLE_SPECS
from manip_tester import (
    DEFAULT_TACTICS, AttackerAgent, EvaluatorAgent, PlannerAgent, TargetAgent, make_backend,
)
from manip_tester.orchestrator import run_episode
from manip_tester.trace import TraceLogger
from scenarios import GOALS, POLICY

try:  # необязательно: ключи из файла .env (poetry add python-dotenv)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="refund_outside_window", choices=[*GOALS, "all"])
    ap.add_argument("--episodes", type=int, default=1, help="эпизодов на каждую цель")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--language", default="English", help="язык клиента и бота (фиксируем для чистоты эксперимента)")
    ap.add_argument("--mock", action="store_true", help="заглушки вместо реальных моделей")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    specs = MOCK_ROLE_SPECS if args.mock else ROLE_SPECS
    backends = {role: make_backend(spec) for role, spec in specs.items()}
    planner = PlannerAgent(backends["planner"], DEFAULT_TACTICS)
    attacker = AttackerAgent(backends["attacker"], language=args.language)
    target = TargetAgent(backends["target"], POLICY, language=args.language)
    evaluator = EvaluatorAgent(backends["evaluator"])

    goals = list(GOALS.values()) if args.goal == "all" else [GOALS[args.goal]]
    run_dir = f"{args.runs_dir}/{datetime.now():%Y%m%d_%H%M%S}"
    results = []

    with TraceLogger(run_dir) as log:
        log.event("run_start", args=vars(args), policy=POLICY,
                  models={r: s.get("model", s["provider"]) for r, s in specs.items()})
        for goal in goals:
            for i in range(args.episodes):
                ep_id = f"{goal.id}-{i + 1:02d}"
                try:
                    results.append(run_episode(
                        episode_id=ep_id, goal=goal, policy=POLICY, planner=planner,
                        attacker=attacker, target=target, evaluator=evaluator,
                        max_turns=args.max_turns, logger=log,
                    ))
                except Exception as e:  # один сбойный эпизод не должен ронять весь прогон
                    log.event("episode_error", episode_id=ep_id, error=repr(e))
                    log.say(f"ep={ep_id} | ERROR {e!r}")

        by_goal = defaultdict(list)
        for r in results:
            by_goal[r.goal_id].append(r)
        log.say("=" * 70)
        log.say(f"{'goal':26} {'episodes':>8} {'violated':>8} {'ASR':>6} {'mean_turn_to_viol':>18}")
        summary = {}
        for gid, rs in by_goal.items():
            hits = [r.first_violation_turn for r in rs if r.violated]
            asr = len(hits) / len(rs)
            mean_t = sum(hits) / len(hits) if hits else float("nan")
            summary[gid] = {"episodes": len(rs), "violated": len(hits), "asr": asr, "mean_turn_to_violation": mean_t}
            log.say(f"{gid:26} {len(rs):>8} {len(hits):>8} {asr:>6.2f} {mean_t:>18.2f}")
        log.event("run_summary", summary=summary)
    print(f"\nЛоги: {run_dir}/")


if __name__ == "__main__":
    main()
