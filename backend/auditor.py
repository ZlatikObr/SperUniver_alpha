"""Business health analysis with deep research enrichment.

Research pipeline (all pre-computed before LLM call):
  1. Zone-specific web searches (5 targeted queries, one per diagnostic zone)
  2. Open structured data: hh.ru vacancies API + CBR key rate
  3. Competitor analysis (2 targeted searches)
  4. Document-based benchmarks (when doc is uploaded, run metric-specific searches)
  5. Two-phase macro/micro research chain
"""
import json
import yaml
import requests
from datetime import datetime
from pathlib import Path

from openai import OpenAI

MODEL = "gpt-4o-mini"
YEAR = datetime.now().year

# Kept for fallback compatibility only
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Поиск открытых отраслевых данных: бенчмарки, средние показатели по рынку, "
            "тренды в отрасли."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос на русском языке"}
            },
            "required": ["query"],
        },
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_prompt(name: str) -> dict:
    path = Path("prompts") / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _web_search(query: str) -> str:
    """DuckDuckGo search — returns formatted snippet text or error string."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "Результаты не найдены."
        return "\n\n".join(
            f"**{r.get('title', '')}**\n{r.get('body', '')}" for r in results
        )
    except Exception as e:
        return f"Поиск недоступен: {e}"


# ── Improvement 1: zone-specific search queries ───────────────────────────────

def _build_zone_queries(profile: dict) -> dict:
    """Return 5 targeted search queries, one per diagnostic zone."""
    industry = profile.get("industry", "малый бизнес")
    return {
        "финансы": f"средняя рентабельность {industry} Россия {YEAR} бенчмарк",
        "маркетинг": f"стоимость привлечения клиента CAC {industry} бенчмарк {YEAR}",
        "операции": f"производительность на сотрудника {industry} норма Россия",
        "команда": f"текучесть кадров {industry} среднее по рынку Россия {YEAR}",
        "стратегия": f"темп роста рынка {industry} {YEAR} Россия прогноз",
    }


def _run_zone_searches(profile: dict) -> dict:
    """Run all 5 zone-specific searches; return {zone: snippet} dict."""
    queries = _build_zone_queries(profile)
    return {zone: _web_search(query) for zone, query in queries.items()}


# ── Improvement 2: open structured data ──────────────────────────────────────

def _fetch_hh_data(profile: dict) -> dict:
    """Fetch hiring-market data from hh.ru public API (no key required)."""
    industry = profile.get("industry", "")
    if not industry:
        return {}
    try:
        r = requests.get(
            "https://api.hh.ru/vacancies",
            params={
                "text": industry,
                "area": 113,           # Russia
                "per_page": 20,
                "only_with_salary": True,
            },
            headers={"User-Agent": "Revizor/1.0 (business-audit-tool)"},
            timeout=7,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        items = data.get("items", [])
        salaries = []
        for item in items:
            s = item.get("salary")
            if s and s.get("currency") == "RUR":
                if s.get("from"):
                    salaries.append(s["from"])
                if s.get("to"):
                    salaries.append(s["to"])
        result: dict = {"vacancy_count": data.get("found", 0)}
        if salaries:
            result["avg_salary_rub"] = int(sum(salaries) / len(salaries))
        return result
    except Exception:
        return {}


def _fetch_cbr_rate() -> str:
    """Fetch current CBR key rate snippet via web search."""
    try:
        snippet = _web_search(f"ключевая ставка ЦБ РФ {YEAR} актуальная процент")
        if "недоступен" in snippet:
            return ""
        # Return first meaningful line containing % and rate context
        for line in snippet.split("\n"):
            if "%" in line and any(w in line.lower() for w in ("ставк", "цб", "банк")):
                return line.strip()[:200]
        return snippet[:200]
    except Exception:
        return ""


def _fetch_open_data(profile: dict) -> dict:
    """Aggregate open structured data: hh.ru vacancy market + CBR key rate."""
    return {
        "hh": _fetch_hh_data(profile),
        "cbr_key_rate_snippet": _fetch_cbr_rate(),
    }


# ── Improvement 3: competitor analysis ───────────────────────────────────────

def _research_competitors(profile: dict) -> str:
    """Find top competitors in industry+region via 2 targeted web searches."""
    industry = profile.get("industry", "бизнес")
    region = profile.get("region", "Россия")
    r1 = _web_search(f"топ компании {industry} {region} лидеры рынка конкуренты")
    r2 = _web_search(f"кто лидирует на рынке {industry} Россия {YEAR}")
    return r1 + "\n\n---\n\n" + r2


# ── Improvement 4: doc + web benchmarks ──────────────────────────────────────

def _run_doc_benchmarks(profile: dict, doc_metrics: dict) -> str:
    """After extracting doc metrics, run targeted searches to contextualise them."""
    industry = profile.get("industry", "бизнес")
    metrics = doc_metrics.get("metrics", {}) if doc_metrics else {}
    queries = []
    for key in list(metrics.keys())[:5]:
        key_lower = key.lower()
        if any(w in key_lower for w in ("рентабельност", "маржа", "margin")):
            queries.append(f"рентабельность {industry} норма бенчмарк Россия {YEAR}")
            break
        if any(w in key_lower for w in ("выручка", "revenue", "оборот")):
            queries.append(f"динамика выручки {industry} средний рост Россия")
            break
    if not queries:
        queries.append(f"финансовые показатели {industry} норма Россия {YEAR}")
    return "\n\n".join(_web_search(q) for q in queries[:2])


# ── Improvement 5: two-phase macro/micro research ─────────────────────────────

def _run_phase1_macro(profile: dict) -> str:
    """Phase 1 — macro: industry trends + challenges (2 queries)."""
    industry = profile.get("industry", "малый бизнес")
    r1 = _web_search(f"рынок {industry} Россия {YEAR} тренды объём рост")
    r2 = _web_search(f"проблемы отрасли {industry} Россия {YEAR} вызовы")
    return r1 + "\n\n---\n\n" + r2


def _run_phase2_micro(profile: dict) -> str:
    """Phase 2 — micro: targeted query based on client's main pain point."""
    challenge = (
        profile.get("main_challenge")
        or profile.get("боль")
        or profile.get("challenge")
        or ""
    )
    industry = profile.get("industry", "бизнес")
    if not challenge:
        return ""
    cl = challenge.lower()
    if any(w in cl for w in ("удержани", "отток", "churn", "клиент")):
        q = f"churn rate удержание клиентов {industry} норма бенчмарк Россия"
    elif any(w in cl for w in ("масштаб", "рост", "scaling")):
        q = f"как масштабировать {industry} бизнес Россия стратегия роста"
    elif any(w in cl for w in ("персонал", "сотрудник", "найм", "кадр")):
        q = f"найм удержание персонала {industry} лучшие практики {YEAR}"
    elif any(w in cl for w in ("продаж", "выручк", "revenue")):
        q = f"увеличение продаж {industry} методы инструменты Россия"
    elif any(w in cl for w in ("процесс", "операц", "автоматиз")):
        q = f"автоматизация бизнес-процессов {industry} ROI примеры"
    elif any(w in cl for w in ("конкурент", "рынок", "доля")):
        q = f"конкурентная стратегия {industry} Россия как выделиться"
    else:
        q = f"{challenge[:60]} решения для {industry} бизнеса Россия"
    return _web_search(q)


# ── Build rich user message for LLM ──────────────────────────────────────────

def _build_user_message(
    profile: dict,
    doc_metrics: dict | None,
    research_ctx: dict | None = None,
) -> str:
    parts = ["## Профиль бизнеса\n"]
    for k, v in profile.items():
        if v:
            parts.append(f"- **{k}**: {v}")

    if doc_metrics and doc_metrics.get("summary"):
        parts.append("\n## Данные из документа\n")
        if doc_metrics.get("metrics"):
            for k, v in list(doc_metrics["metrics"].items())[:15]:
                parts.append(f"- {k}: {v}")
        parts.append(f"\nТекст документа (фрагмент):\n{doc_metrics['summary'][:2000]}")

    if research_ctx:
        parts.append("\n## Данные внешнего ресёрча\n")

        # Zone benchmarks
        if research_ctx.get("zone_benchmarks"):
            parts.append("### Отраслевые бенчмарки по зонам\n")
            for zone, data in research_ctx["zone_benchmarks"].items():
                if data and "недоступен" not in data:
                    parts.append(f"**{zone.capitalize()}:**\n{data[:400]}\n")

        # hh.ru + CBR
        od = research_ctx.get("open_data", {})
        hh = od.get("hh", {})
        if hh:
            parts.append("### hh.ru — рынок труда в отрасли")
            if hh.get("vacancy_count"):
                parts.append(f"- Активных вакансий в отрасли: {hh['vacancy_count']}")
            if hh.get("avg_salary_rub"):
                parts.append(f"- Средняя предлагаемая зарплата: {hh['avg_salary_rub']:,} ₽")
        cbr = od.get("cbr_key_rate_snippet", "")
        if cbr:
            parts.append(f"### ЦБ РФ — ключевая ставка\n{cbr}")

        # Competitors
        if research_ctx.get("competitors") and "недоступен" not in research_ctx["competitors"]:
            parts.append(f"### Конкурентное окружение\n{research_ctx['competitors'][:800]}")

        # Doc benchmarks (only when doc was uploaded)
        if research_ctx.get("doc_benchmarks") and "недоступен" not in research_ctx["doc_benchmarks"]:
            parts.append(f"### Бенчмарки для метрик из документа\n{research_ctx['doc_benchmarks'][:600]}")

        # Macro + micro
        if research_ctx.get("macro") and "недоступен" not in research_ctx["macro"]:
            parts.append(f"### Макроконтекст отрасли\n{research_ctx['macro'][:600]}")
        if research_ctx.get("micro") and "недоступен" not in research_ctx["micro"]:
            parts.append(f"### Контекст по ключевой боли клиента\n{research_ctx['micro'][:600]}")
    else:
        parts.append(
            "\n*Документ не прикреплён. Используй функцию web_search "
            "для обогащения анализа отраслевым контекстом.*"
        )

    return "\n".join(parts)


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call_llm_direct(system: str, user_content: str, client: OpenAI) -> dict | None:
    """Direct LLM call — used when research context is pre-computed."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1800,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return _extract_json_object(response.choices[0].message.content or "")


def _run_with_tools(system: str, user_content: str, client: OpenAI) -> dict | None:
    """Fallback: agentic tool-loop where LLM drives its own searches."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1800,
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",
        )
        choice = response.choices[0]
        if choice.finish_reason == "stop":
            return _extract_json_object(choice.message.content or "")
        if choice.finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls or []
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                if tc.function.name == "web_search":
                    args = json.loads(tc.function.arguments)
                    search_result = _web_search(args.get("query", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": search_result,
                    })
        else:
            break
    return None


# ── Public API (signature unchanged) ─────────────────────────────────────────

def analyze_business(
    profile: dict,
    doc_metrics: dict | None,
    client: OpenAI,
    answer_count: int = 10,
) -> dict:
    if answer_count < 5:
        return _partial_assessment(profile)

    prompt = _load_prompt("audit_v1")
    has_doc = bool(doc_metrics and doc_metrics.get("summary"))
    data_sources: list[str] = ["опрос"]

    # ── Research pipeline ──────────────────────────────────────────────────
    research_ctx: dict = {}

    # 1. Zone-specific benchmarks (always)
    research_ctx["zone_benchmarks"] = _run_zone_searches(profile)
    data_sources.append("отраслевые бенчмарки (web)")

    # 2. Open structured data: hh.ru + CBR
    research_ctx["open_data"] = _fetch_open_data(profile)
    if research_ctx["open_data"].get("hh", {}).get("vacancy_count"):
        data_sources.append("hh.ru")

    # 3. Competitor analysis
    research_ctx["competitors"] = _research_competitors(profile)

    # 4. Doc + web combined (extra benchmarks when document uploaded)
    if has_doc:
        data_sources.append("загруженный документ")
        research_ctx["doc_benchmarks"] = _run_doc_benchmarks(profile, doc_metrics)
        data_sources.append("бенчмарки по метрикам документа (web)")

    # 5. Two-phase macro/micro research
    research_ctx["macro"] = _run_phase1_macro(profile)
    research_ctx["micro"] = _run_phase2_micro(profile)

    # ── LLM call with rich context ─────────────────────────────────────────
    user_content = _build_user_message(profile, doc_metrics, research_ctx)
    result = _call_llm_direct(prompt["system"], user_content, client)

    if result is None:
        # Fallback: let LLM drive its own searches on plain message
        fallback_content = _build_user_message(profile, doc_metrics, None)
        result = _run_with_tools(prompt["system"], fallback_content, client)

    if result is None:
        return _fallback_assessment(profile)

    result["data_sources"] = data_sources
    return result


# ── Fallback assessments (unchanged) ─────────────────────────────────────────

def _partial_assessment(profile: dict) -> dict:
    return {
        "business_profile": {
            "industry": profile.get("industry", "Не указано"),
            "size": profile.get("size", "Не указано"),
            "revenue_range": profile.get("revenue_range", "Не указано"),
        },
        "health_assessment": {
            "zones": [],
            "top_risks": [],
            "overall_health": (
                "Недостаточно данных для полного анализа. "
                "Ответьте на все вопросы опроса."
            ),
        },
        "recommended_zone_ids": [],
        "data_sources": ["опрос (частичный)"],
        "disclaimer": (
            "Анализ носит рекомендательный характер. "
            "Для принятия управленческих решений рекомендуется верификация с профильным консультантом."
        ),
        "_partial": True,
    }


def _fallback_assessment(profile: dict) -> dict:
    zones = [
        {"name": z, "score": 3, "risks": ["Требует детального анализа"], "growth_points": []}
        for z in ("финансы", "операции", "маркетинг", "команда", "стратегия")
    ]
    return {
        "business_profile": {
            "industry": profile.get("industry", "Не указано"),
            "size": profile.get("size", "Не указано"),
            "revenue_range": profile.get("revenue_range", "Не указано"),
        },
        "health_assessment": {
            "zones": zones,
            "top_risks": [
                "Для точной диагностики требуется верификация данных с консультантом"
            ],
            "overall_health": (
                "Первичная диагностика проведена. "
                "Рекомендуем консультацию для углублённого анализа."
            ),
        },
        "recommended_zone_ids": ["финансы", "операции", "стратегия"],
        "data_sources": ["опрос"],
        "disclaimer": (
            "Анализ носит рекомендательный характер. "
            "Для принятия управленческих решений рекомендуется верификация с профильным консультантом."
        ),
    }
