"""
Print token usage statistics from logs/tokens.jsonl.

Usage:
    python scripts/token_stats.py
    python scripts/token_stats.py --last 5   # last N sessions (grouped by date)
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# gpt-4o-mini pricing (per 1K tokens), USD
PRICE_INPUT_PER_1K  = 0.000150
PRICE_OUTPUT_PER_1K = 0.000600

TOKENS_PATH = Path(__file__).parent.parent / "logs" / "tokens.jsonl"


def load_records() -> list[dict]:
    if not TOKENS_PATH.exists():
        return []
    records = []
    with open(TOKENS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1000 * PRICE_INPUT_PER_1K
        + completion_tokens / 1000 * PRICE_OUTPUT_PER_1K
    )


def print_stats(records: list[dict]):
    if not records:
        print("Нет данных. Запустите сессию — токены начнут логироваться в logs/tokens.jsonl")
        return

    # ── По шагам ──────────────────────────────────────────────────────────────
    by_step: dict[str, dict] = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    for r in records:
        s = by_step[r["step"]]
        s["calls"]      += 1
        s["prompt"]     += r.get("prompt_tokens", 0)
        s["completion"] += r.get("completion_tokens", 0)
        s["total"]      += r.get("total_tokens", 0)

    print("\n── Расход по шагам (все сессии) ─────────────────────────────────────────")
    print(f"{'Шаг':<45} {'Вызовы':>7} {'Вход':>8} {'Выход':>8} {'Итого':>8} {'Стоимость':>10}")
    print("─" * 90)
    grand_prompt = grand_completion = grand_total = 0.0
    for step, s in sorted(by_step.items()):
        c = cost_usd(s["prompt"], s["completion"])
        grand_prompt     += s["prompt"]
        grand_completion += s["completion"]
        grand_total      += s["total"]
        print(f"{step:<45} {s['calls']:>7} {s['prompt']:>8} {s['completion']:>8} {s['total']:>8} {c:>9.4f}$")

    grand_cost = cost_usd(grand_prompt, grand_completion)
    print("─" * 90)
    print(f"{'ИТОГО':<45} {'':>7} {int(grand_prompt):>8} {int(grand_completion):>8} {int(grand_total):>8} {grand_cost:>9.4f}$")

    # ── По дням ───────────────────────────────────────────────────────────────
    by_day: dict[str, dict] = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
    for r in records:
        day = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d")
        by_day[day]["prompt"]     += r.get("prompt_tokens", 0)
        by_day[day]["completion"] += r.get("completion_tokens", 0)
        by_day[day]["calls"]      += 1

    print("\n── Расход по дням ────────────────────────────────────────────────────────")
    print(f"{'Дата':<15} {'Вызовов':>8} {'Вход':>8} {'Выход':>8} {'Стоимость':>10}")
    print("─" * 52)
    for day in sorted(by_day):
        d = by_day[day]
        c = cost_usd(d["prompt"], d["completion"])
        print(f"{day:<15} {d['calls']:>8} {d['prompt']:>8} {d['completion']:>8} {c:>9.4f}$")

    # ── Одна полная сессия ────────────────────────────────────────────────────
    session_steps = [
        "survey.generate_followup_questions",
        "survey.build_business_profile",
        "auditor.call_llm_direct",
        "proposal_gen.generate_proposal_text",
    ]
    session_calls: dict[str, dict] = {k: v for k, v in by_step.items() if k in session_steps}
    if session_calls:
        total_sessions = max(s["calls"] for s in session_calls.values()) or 1
        avg_p = sum(s["prompt"] for s in session_calls.values()) / total_sessions
        avg_c = sum(s["completion"] for s in session_calls.values()) / total_sessions
        avg_cost = cost_usd(avg_p, avg_c)
        print(f"\n── Средняя стоимость 1 полной сессии (≈{total_sessions} сессий) ────────────────────")
        print(f"   Вход:  {avg_p:>8.0f} токенов")
        print(f"   Выход: {avg_c:>8.0f} токенов")
        print(f"   Цена:  {avg_cost:>8.4f}$ (~{avg_cost * 90:.2f} ₽ при курсе 90)")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=0, help="Показать только последние N дней")
    args = parser.parse_args()

    records = load_records()
    if args.last and records:
        cutoff_day = sorted({datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d") for r in records})[-args.last]
        records = [r for r in records if datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d") >= cutoff_day]

    print_stats(records)


if __name__ == "__main__":
    main()
