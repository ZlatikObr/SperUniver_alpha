"""Тесты модуля backend.auditor — построение запросов, fallback, защита от сетевых ошибок."""
from unittest.mock import MagicMock

import pytest

from backend import auditor


# ── Happy path ────────────────────────────────────────────────────────────────

def test_build_zone_queries_covers_all_five_zones():
    """Happy path: запросы строятся по всем 5 диагностическим зонам и содержат отрасль."""
    profile = {"industry": "IT и технологии"}
    queries = auditor._build_zone_queries(profile)

    expected_zones = {"финансы", "операции", "маркетинг", "команда", "стратегия"}
    assert set(queries.keys()) == expected_zones

    for zone, q in queries.items():
        assert "IT" in q, f"Запрос для {zone} должен включать отрасль"
        assert str(auditor.YEAR) in q or "Россия" in q


def test_partial_assessment_when_too_few_answers():
    """Happy path: меньше 5 ответов — возвращается _partial_assessment с флагом _partial."""
    profile = {"industry": "IT", "size": "11–50", "revenue_range": "100 млн"}
    fake_client = MagicMock()  # не должен вызываться

    result = auditor.analyze_business(profile, None, fake_client, answer_count=3)

    assert result.get("_partial") is True
    assert result["health_assessment"]["zones"] == []
    fake_client.chat.completions.create.assert_not_called()


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_extract_json_object_handles_nested_objects():
    """Edge case: извлекаемый JSON-объект с вложенными структурами парсится корректно."""
    raw = 'preface {"a": {"b": [1, 2, 3]}, "c": "text"} suffix'
    result = auditor._extract_json_object(raw)

    assert result == {"a": {"b": [1, 2, 3]}, "c": "text"}


def test_run_phase2_micro_returns_empty_when_no_challenge():
    """Edge case: без main_challenge микро-фаза возвращает пустую строку, а не падает."""
    profile = {"industry": "IT"}
    assert auditor._run_phase2_micro(profile) == ""


def test_fetch_hh_handles_network_failure(monkeypatch):
    """Edge case: при сетевой ошибке к hh.ru возвращается пустой dict, а не exception."""
    def boom(*a, **kw):
        raise ConnectionError("network is unreachable")

    monkeypatch.setattr(auditor.requests, "get", boom)

    result = auditor._fetch_hh_data({"industry": "IT"})
    assert result == {}


# ── Negative ──────────────────────────────────────────────────────────────────

def test_analyze_business_returns_fallback_when_llm_returns_garbage(monkeypatch):
    """Негатив: LLM вернул не-JSON и tool-loop тоже не помог → _fallback_assessment."""
    # Глушим весь сетевой ресёрч и YAML
    monkeypatch.setattr(auditor, "_load_prompt", lambda name: {"system": "x"})
    monkeypatch.setattr(auditor, "_web_search", lambda q: "поиск недоступен")
    monkeypatch.setattr(auditor, "_fetch_hh_data", lambda p: {})
    monkeypatch.setattr(auditor, "_fetch_cbr_rate", lambda: "")

    fake_client = MagicMock()
    # Любой ответ — мусор без JSON
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            finish_reason="stop",
            message=MagicMock(content="К сожалению, не могу обработать запрос", tool_calls=None),
        )]
    )

    profile = {
        "industry": "Розничная торговля",
        "region": "Москва",
        "main_challenge": "падают продажи",
    }
    result = auditor.analyze_business(profile, None, fake_client, answer_count=10)

    # Должен сработать fallback с 5 зонами по умолчанию
    assert "health_assessment" in result
    zones = result["health_assessment"]["zones"]
    assert len(zones) == 5
    zone_names = {z["name"] for z in zones}
    assert zone_names == {"финансы", "операции", "маркетинг", "команда", "стратегия"}
