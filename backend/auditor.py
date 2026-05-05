"""Business health analysis with deep research enrichment.

Research pipeline (all pre-computed before LLM call, run in parallel):
  1. Zone-specific web searches (5 targeted queries, one per diagnostic zone)
  2. Open structured data: hh.ru vacancies API + CBR key rate
  3. Competitor analysis (2 targeted searches)
  4. Document-based benchmarks (when doc is uploaded, run metric-specific searches)
  5. Two-phase macro/micro research chain
"""
import json
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from openai import OpenAI

from ._config import MODEL
from ._utils import extract_json_object as _extract_json_object, load_prompt as _load_prompt
from .agent_logger import log_agent_step

logger = logging.getLogger(__name__)

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
        logger.warning("DuckDuckGo search failed for query: %s", query, exc_info=True)
        log_agent_step("auditor.web_search", "error", query=query, error=e)
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
            headers={"User-Agent": "Puls/1.0 (business-audit-tool)"},
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
    except Exception as exc:
        logger.warning("hh.ru data fetch failed.", exc_info=True)
        log_agent_step("auditor.fetch_hh_data", "error", error=exc)
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
    except Exception as exc:
        logger.warning("CBR key rate fetch failed.", exc_info=True)
        log_agent_step("auditor.fetch_cbr_rate", "error", error=exc)
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
    financial_analysis = doc_metrics.get("financial_analysis", {}) if doc_metrics else {}
    queries = []
    metric_names = list(metrics.keys())
    metric_names.extend(i.get("name", "") for i in financial_analysis.get("indicators", []))
    metric_names.extend(r.get("name", "") for r in financial_analysis.get("ratios", []))
    for key in metric_names[:8]:
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
        source_files = doc_metrics.get("source_files") or [doc_metrics.get("filename")]
        source_files = [s for s in source_files if s]
        if source_files:
            parts.append(f"Файлы: {', '.join(source_files)}")
        if doc_metrics.get("metrics"):
            for k, v in list(doc_metrics["metrics"].items())[:15]:
                parts.append(f"- {k}: {v}")
        if doc_metrics.get("facts"):
            parts.append("\nФакты и цитаты из файлов:")
            for fact in doc_metrics.get("facts", [])[:12]:
                parts.append(f"- {fact}")
        financial_analysis = doc_metrics.get("financial_analysis") or {}
        if financial_analysis.get("indicators") or financial_analysis.get("ratios"):
            parts.append("\nФинансовый анализ по приложенным файлам:")
            for indicator in financial_analysis.get("indicators", [])[:8]:
                parts.append(
                    f"- {indicator.get('name')}: {indicator.get('value')} "
                    f"(факт: {indicator.get('fact', 'нет цитаты')})"
                )
            for ratio in financial_analysis.get("ratios", [])[:8]:
                parts.append(
                    f"- {ratio.get('name')}: {ratio.get('value')} — "
                    f"{ratio.get('interpretation', '')}"
                )
            for risk in financial_analysis.get("risks", [])[:5]:
                parts.append(f"- Финансовый риск: {risk}")
        parts.append(f"\nТекст документа (фрагмент):\n{doc_metrics['summary'][:3500]}")

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


def _doc_evidence_items(doc_metrics: dict | None) -> list[dict]:
    if not doc_metrics:
        return []
    items = []
    for quote in doc_metrics.get("quotes", [])[:12]:
        if isinstance(quote, dict) and quote.get("text"):
            items.append({
                "source": quote.get("source") or doc_metrics.get("filename") or "файл",
                "quote": quote["text"],
            })
    for fact in doc_metrics.get("facts", [])[:12]:
        items.append({
            "source": doc_metrics.get("filename") or "файл",
            "quote": fact,
        })
    financial_analysis = doc_metrics.get("financial_analysis") or {}
    for indicator in financial_analysis.get("indicators", [])[:8]:
        fact = indicator.get("fact") or f"{indicator.get('name')}: {indicator.get('value')}"
        items.append({
            "source": indicator.get("source") or doc_metrics.get("filename") or "файл",
            "quote": fact,
        })
    for ratio in financial_analysis.get("ratios", [])[:8]:
        items.append({
            "source": doc_metrics.get("filename") or "расчёт по файлу",
            "quote": f"{ratio.get('name')}: {ratio.get('value')} — {ratio.get('interpretation', '')}",
        })

    seen = set()
    unique = []
    for item in items:
        key = item["quote"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:20]


def _zone_keywords(zone_name: str) -> tuple[str, ...]:
    return {
        "финансы": ("выруч", "приб", "убыт", "марж", "рентабель", "расход", "денеж", "cash", "дебитор", "кредитор", "ликвид"),
        "операции": ("процесс", "операц", "ручн", "срок", "производ", "ошиб", "автомат"),
        "маркетинг": ("клиент", "продаж", "конверс", "реклама", "cac", "ltv", "отток", "жалоб"),
        "команда": ("сотрудник", "персонал", "команд", "штат", "текуч", "найм", "мотивац"),
        "стратегия": ("стратег", "рынок", "конкур", "рост", "масштаб", "план", "инвест"),
    }.get(zone_name.lower(), ())


def _select_evidence(zone_name: str, evidence_items: list[dict], limit: int = 3) -> list[dict]:
    keywords = _zone_keywords(zone_name)
    selected = []
    for item in evidence_items:
        quote = item.get("quote", "").lower()
        if any(keyword in quote for keyword in keywords):
            selected.append(item)
        if len(selected) >= limit:
            break
    if selected:
        return selected
    return evidence_items[:limit]


def _risk_to_text(risk) -> str:
    if isinstance(risk, dict):
        return risk.get("risk") or risk.get("title") or risk.get("text") or str(risk)
    return str(risk)


def _ensure_assessment_evidence(result: dict, profile: dict, doc_metrics: dict | None) -> dict:
    """Backfill explanations/evidence if the model returns the older compact schema."""
    health = result.setdefault("health_assessment", {})
    zones = health.get("zones") or []
    evidence_items = _doc_evidence_items(doc_metrics)

    for zone in zones:
        zone_name = zone.get("name", "")
        score = zone.get("score", 3)
        risks = zone.get("risks") or []
        growth = zone.get("growth_points") or []

        if not zone.get("score_explanation"):
            if risks:
                zone["score_explanation"] = (
                    f"Оценка {score}/5 поставлена из-за выявленных сигналов риска: "
                    f"{'; '.join(map(str, risks[:2]))}."
                )
            elif growth:
                zone["score_explanation"] = (
                    f"Оценка {score}/5 отражает рабочее состояние зоны; основной потенциал роста: "
                    f"{'; '.join(map(str, growth[:2]))}."
                )
            else:
                zone["score_explanation"] = (
                    f"Оценка {score}/5 основана на ответах диагностики и доступных данных по профилю "
                    f"«{profile.get('industry', 'бизнес')}»."
                )

        if not zone.get("evidence"):
            selected = _select_evidence(zone_name, evidence_items)
            zone["evidence"] = [
                {
                    "source": item.get("source", "файл"),
                    "quote": item.get("quote", ""),
                    "implication": "Факт использован как подтверждение оценки зоны.",
                }
                for item in selected
            ]

    top_risks = health.get("top_risks") or []
    if top_risks and not health.get("top_risk_details"):
        health["top_risk_details"] = []
        for i, risk in enumerate(top_risks[:3]):
            evidence = evidence_items[i:i + 1] or evidence_items[:1]
            health["top_risk_details"].append({
                "risk": _risk_to_text(risk),
                "why_critical": "Риск входит в топ приоритетов, потому что влияет на денежный поток, продажи или управляемость бизнеса.",
                "evidence": evidence,
            })

    if doc_metrics and doc_metrics.get("financial_analysis"):
        result["document_analysis"] = doc_metrics["financial_analysis"]
        result["document_facts"] = doc_metrics.get("facts", [])[:20]

    return result


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call_llm_direct(system: str, user_content: str, client: OpenAI) -> dict | None:
    """Direct LLM call — used when research context is pre-computed."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1800,
        response_format={"type": "json_object"},
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
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as exc:
                        logger.warning("Invalid web_search tool arguments.", exc_info=True)
                        log_agent_step("auditor.tool_args", "error", error=exc)
                        args = {}
                    search_result = _web_search(args.get("query", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": search_result,
                    })
        else:
            break
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_business(
    profile: dict,
    doc_metrics: dict | None,
    client: OpenAI,
    answer_count: int = 10,
    on_progress=None,           # optional callback(str) for UI progress updates
) -> dict:
    """Run full research pipeline (parallel) then call LLM for 5-zone audit."""
    log_agent_step(
        "auditor.analyze_business",
        "start",
        answer_count=answer_count,
        has_doc=bool(doc_metrics and doc_metrics.get("summary")),
        industry=profile.get("industry"),
    )

    def _notify(msg: str):
        if on_progress:
            try:
                on_progress(msg)
            except Exception as exc:
                logger.warning("Progress callback failed.", exc_info=True)
                log_agent_step("auditor.progress_callback", "error", error=exc)

    if answer_count < 5:
        return _partial_assessment(profile)

    prompt = _load_prompt("audit_v1")
    has_doc = bool(doc_metrics and doc_metrics.get("summary"))
    data_sources: list[str] = ["опрос"]

    # ── Research pipeline — parallel searches ─────────────────────────────
    _notify("🔍 Исследуем рынок: бенчмарки по 5 зонам и конкуренты...")

    zone_queries = _build_zone_queries(profile)
    zone_benchmarks: dict = {}
    hh_data: dict = {}
    cbr: str = ""
    competitors: str = ""
    macro: str = ""
    micro: str = ""
    doc_benchmarks: str = ""

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            zone_futs = {zone: ex.submit(_web_search, q) for zone, q in zone_queries.items()}
            hh_fut    = ex.submit(_fetch_hh_data, profile)
            cbr_fut   = ex.submit(_fetch_cbr_rate)
            comp_fut  = ex.submit(_research_competitors, profile)
            mac_fut   = ex.submit(_run_phase1_macro, profile)
            mic_fut   = ex.submit(_run_phase2_micro, profile)
            doc_fut   = ex.submit(_run_doc_benchmarks, profile, doc_metrics) if has_doc else None

            for zone, f in zone_futs.items():
                try:
                    zone_benchmarks[zone] = f.result(timeout=20)
                except Exception as exc:
                    logger.warning("Zone benchmark failed for %s.", zone, exc_info=True)
                    log_agent_step("auditor.zone_benchmark", "error", zone=zone, error=exc)
                    zone_benchmarks[zone] = ""

            try:
                hh_data = hh_fut.result(timeout=10)
            except Exception as exc:
                logger.warning("hh.ru future failed.", exc_info=True)
                log_agent_step("auditor.hh_future", "error", error=exc)
                hh_data = {}

            try:
                cbr = cbr_fut.result(timeout=10)
            except Exception as exc:
                logger.warning("CBR future failed.", exc_info=True)
                log_agent_step("auditor.cbr_future", "error", error=exc)
                cbr = ""

            try:
                competitors = comp_fut.result(timeout=20)
            except Exception as exc:
                logger.warning("Competitor research failed.", exc_info=True)
                log_agent_step("auditor.competitors", "error", error=exc)
                competitors = ""

            try:
                macro = mac_fut.result(timeout=20)
            except Exception as exc:
                logger.warning("Macro research failed.", exc_info=True)
                log_agent_step("auditor.macro", "error", error=exc)
                macro = ""

            try:
                micro = mic_fut.result(timeout=15)
            except Exception as exc:
                logger.warning("Micro research failed.", exc_info=True)
                log_agent_step("auditor.micro", "error", error=exc)
                micro = ""

            if doc_fut:
                try:
                    doc_benchmarks = doc_fut.result(timeout=15)
                except Exception as exc:
                    logger.warning("Document benchmark research failed.", exc_info=True)
                    log_agent_step("auditor.doc_benchmarks", "error", error=exc)
                    doc_benchmarks = ""

    except Exception as exc:
        logger.warning("Research pipeline failed; continuing with empty context.", exc_info=True)
        log_agent_step("auditor.research_pipeline", "error", error=exc)

    # Build data_sources list
    data_sources.append("отраслевые бенчмарки (web)")
    if hh_data.get("vacancy_count"):
        data_sources.append("hh.ru")
    if has_doc:
        data_sources.append("загруженный документ")
        data_sources.append("бенчмарки по метрикам документа (web)")

    research_ctx: dict = {
        "zone_benchmarks": zone_benchmarks,
        "open_data": {"hh": hh_data, "cbr_key_rate_snippet": cbr},
        "competitors": competitors,
        "macro": macro,
        "micro": micro,
    }
    if has_doc:
        research_ctx["doc_benchmarks"] = doc_benchmarks

    # ── LLM call with rich pre-computed context ────────────────────────────
    _notify("🤖 AI строит диагностику по 5 зонам...")

    user_content = _build_user_message(profile, doc_metrics, research_ctx)
    result = _call_llm_direct(prompt["system"], user_content, client)

    if result is None:
        # Fallback: let LLM drive its own searches
        fallback_content = _build_user_message(profile, doc_metrics, None)
        result = _run_with_tools(prompt["system"], fallback_content, client)

    if result is None:
        log_agent_step("auditor.analyze_business", "fallback")
        return _ensure_assessment_evidence(_fallback_assessment(profile), profile, doc_metrics)

    llm_sources = result.get("data_sources", [])
    result["data_sources"] = list(dict.fromkeys([*llm_sources, *data_sources]))
    result = _ensure_assessment_evidence(result, profile, doc_metrics)
    log_agent_step("auditor.analyze_business", "success", data_sources=result["data_sources"])
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
