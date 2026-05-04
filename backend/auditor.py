"""Business health analysis via Claude with optional web_search enrichment."""
import json
import yaml
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Поиск открытых отраслевых данных: бенчмарки, средние показатели по рынку, "
        "тренды в отрасли. Используй для обогащения анализа при отсутствии документа клиента."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Поисковый запрос на русском языке"
            }
        },
        "required": ["query"]
    }
}


def _load_prompt(name: str) -> dict:
    path = Path("prompts") / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _web_search(query: str) -> str:
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


def analyze_business(
    profile: dict,
    doc_metrics: dict | None,
    client: anthropic.Anthropic,
    answer_count: int = 10,
) -> dict:
    """
    Run 5-zone business health assessment.

    Returns the health_assessment JSON as defined in audit_v1.yaml.
    Falls back to a minimal structure on parse errors.
    """
    if answer_count < 5:
        return _partial_assessment(profile)

    prompt = _load_prompt("audit_v1")
    user_content = _build_user_message(profile, doc_metrics)

    use_search = doc_metrics is None or not doc_metrics.get("summary")

    if use_search:
        result = _run_with_tools(prompt["system"], user_content, client)
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            system=prompt["system"],
            messages=[{"role": "user", "content": user_content}],
        )
        result = _extract_json(response.content[0].text)

    if result is None:
        return _fallback_assessment(profile)

    if "data_sources" not in result:
        result["data_sources"] = ["опрос"]
    if doc_metrics and doc_metrics.get("summary"):
        result["data_sources"] = list(set(result.get("data_sources", []) + ["загруженный документ"]))

    return result


def _build_user_message(profile: dict, doc_metrics: dict | None) -> str:
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
    else:
        parts.append("\n*Документ не прикреплён. Используй web_search для обогащения отраслевым контекстом.*")

    return "\n".join(parts)


def _run_with_tools(system: str, user_content: str, client: anthropic.Anthropic) -> dict | None:
    messages = [{"role": "user", "content": user_content}]
    search_results = []

    for _ in range(5):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            system=system,
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return _extract_json(block.text)
            return None

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "web_search":
                    query = block.input.get("query", "")
                    search_text = _web_search(query)
                    search_results.append(f"Запрос: {query}\nРезультат: {search_text[:800]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": search_text,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return None


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
            "overall_health": "Недостаточно данных для полного анализа. Ответьте на все вопросы опроса."
        },
        "recommended_zone_ids": [],
        "data_sources": ["опрос (частичный)"],
        "disclaimer": "Анализ носит рекомендательный характер. Для принятия управленческих решений рекомендуется верификация с профильным консультантом.",
        "_partial": True,
    }


def _fallback_assessment(profile: dict) -> dict:
    zones = [
        {"name": "финансы", "score": 3, "risks": ["Требует детального анализа"], "growth_points": []},
        {"name": "операции", "score": 3, "risks": ["Требует детального анализа"], "growth_points": []},
        {"name": "маркетинг", "score": 3, "risks": ["Требует детального анализа"], "growth_points": []},
        {"name": "команда", "score": 3, "risks": ["Требует детального анализа"], "growth_points": []},
        {"name": "стратегия", "score": 3, "risks": ["Требует детального анализа"], "growth_points": []},
    ]
    return {
        "business_profile": {
            "industry": profile.get("industry", "Не указано"),
            "size": profile.get("size", "Не указано"),
            "revenue_range": profile.get("revenue_range", "Не указано"),
        },
        "health_assessment": {
            "zones": zones,
            "top_risks": ["Для точной диагностики требуется верификация данных с консультантом"],
            "overall_health": "Первичная диагностика проведена. Рекомендуем консультацию для углублённого анализа."
        },
        "recommended_zone_ids": ["финансы", "операции", "стратегия"],
        "data_sources": ["опрос"],
        "disclaimer": "Анализ носит рекомендательный характер. Для принятия управленческих решений рекомендуется верификация с профильным консультантом.",
    }
