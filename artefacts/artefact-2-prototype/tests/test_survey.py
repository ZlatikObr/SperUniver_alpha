"""Тесты модуля backend.survey — базовые вопросы, парсинг JSON, fallback-поведение LLM."""
from unittest.mock import MagicMock

import pytest

from backend import survey


# ── Happy path ────────────────────────────────────────────────────────────────

def test_get_base_questions_returns_required_fields():
    """Happy path: базовые вопросы содержат все обязательные поля и валидные типы."""
    questions = survey.get_base_questions()

    assert isinstance(questions, list)
    assert len(questions) >= 10, "Должно быть минимум 10 базовых вопросов"

    required_keys = {"id", "text", "type"}
    valid_types = {"select", "text", "textarea"}
    seen_ids = set()

    for q in questions:
        assert required_keys.issubset(q.keys()), f"У вопроса {q} нет обязательных полей"
        assert q["type"] in valid_types, f"Невалидный type: {q['type']}"
        assert q["id"] not in seen_ids, f"Дублированный id: {q['id']}"
        seen_ids.add(q["id"])

        if q["type"] == "select":
            assert q.get("options"), f"select-вопрос {q['id']} без options"
            assert len(q["options"]) >= 2

    # Проверяем наличие критичных вопросов
    assert "industry" in seen_ids
    assert "main_challenge" in seen_ids


def test_extract_json_object_with_surrounding_noise():
    """Happy path: парсер извлекает JSON, даже если LLM добавил пояснения вокруг."""
    raw = """
    Конечно, вот профиль:

    ```json
    {"industry": "IT", "size": "51–200", "revenue_range": "100–500 млн ₽"}
    ```

    Надеюсь, это поможет!
    """
    result = survey._extract_json_object(raw)

    assert result is not None
    assert result["industry"] == "IT"
    assert result["size"] == "51–200"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_extract_json_array_invalid_returns_none():
    """Edge case: при битом JSON парсер возвращает None, а не падает."""
    bad_inputs = [
        "Просто текст без скобок",
        "[не валидный json]",
        "[1, 2, 3,]",          # trailing comma
        "[{broken: true}]",    # unquoted key
    ]
    for raw in bad_inputs:
        assert survey._extract_json_array(raw) is None, f"Должен вернуть None для: {raw!r}"


def test_build_business_profile_fallback_when_llm_fails(monkeypatch):
    """Edge case: если LLM вернул мусор — собираем профиль из ответов опроса."""
    base_answers = {
        "industry": "Розничная торговля",
        "region": "Москва",
        "employees": "11–50",
        "revenue": "30–100 млн ₽",
        "main_challenge": "Падают продажи",
        "company_age": "3–7 лет",
        "profitability": "Небольшая прибыль (до 5%)",
        "client_satisfaction": "Средний уровень, стабильно",
        "team_challenge": "Высокая текучесть кадров",
        "process_maturity": "Есть отдельные инструкции, но не систематизированы",
    }
    followup_answers = {"followup_1": "За 3 месяца"}

    # Мокируем YAML-промпт, чтобы не зависеть от файловой системы
    monkeypatch.setattr(
        survey, "_load_prompt",
        lambda name: {"system": "x", "profile_system": "x"},
    )

    # LLM возвращает мусор — JSON не извлечётся
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Извините, не смог сформировать ответ"))]
    )

    profile = survey.build_business_profile(base_answers, followup_answers, fake_client)

    # Должен сработать fallback из base_answers
    assert profile["industry"] == "Розничная торговля"
    assert profile["main_challenge"] == "Падают продажи"
    assert profile["region"] == "Москва"


# ── Negative ──────────────────────────────────────────────────────────────────

def test_generate_followup_uses_fallback_on_garbage_llm(monkeypatch):
    """Негатив: если LLM возвращает не-JSON, используем дефолтные fallback-вопросы."""
    monkeypatch.setattr(survey, "_load_prompt", lambda name: {"system": "x"})

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="error 500: bad gateway"))]
    )

    result = survey.generate_followup_questions({"industry": "IT"}, fake_client)

    assert isinstance(result, list)
    assert len(result) == 3, "Fallback должен вернуть ровно 3 вопроса"
    for q in result:
        assert "id" in q and "text" in q and "type" in q
