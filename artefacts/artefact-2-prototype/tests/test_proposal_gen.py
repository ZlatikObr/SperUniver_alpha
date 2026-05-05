"""Тесты модуля backend.proposal_gen — сборка КП, fallback, рендер HTML."""
from unittest.mock import MagicMock

import pytest

from backend import proposal_gen as pg


# ── Happy path ────────────────────────────────────────────────────────────────

def test_build_services_section_includes_all_selected():
    """Happy path: блок услуг ВСЕГДА содержит все выбранные позиции (защита от LLM-обрезки)."""
    services = [
        {
            "id": "s1", "name": "Финансовый аудит", "zone": "финансы",
            "description": "Описание 1", "expected_effect": "Эффект 1",
            "price_range": "100к", "duration": "2 нед", "roi_estimate": "300%",
        },
        {
            "id": "s2", "name": "Картирование процессов", "zone": "операции",
            "description": "Описание 2", "expected_effect": "Эффект 2",
            "price_range": "200к", "duration": "4 нед", "roi_estimate": "400%",
        },
    ]
    section = pg._build_services_section(services)

    assert "Финансовый аудит" in section
    assert "Картирование процессов" in section
    assert "300%" in section and "400%" in section
    # Проверяем, что услуги собраны таблицей
    assert "| Услуга | Задача | Методология |" in section
    assert section.count("| Финансовый аудит |") == 1


def test_fallback_proposal_is_complete_without_llm():
    """Happy path: _fallback_proposal даёт валидный КП без вызова LLM."""
    profile = {"industry": "IT", "region": "Москва", "revenue_range": "100–500 млн ₽"}
    health = {
        "zones": [
            {"name": "финансы", "score": 2, "risks": ["низкая маржа"], "growth_points": []},
            {"name": "операции", "score": 4, "risks": [], "growth_points": ["автоматизация"]},
        ],
    }
    services = [{
        "id": "s1", "name": "Аудит", "zone": "финансы",
        "description": "d", "expected_effect": "e",
        "price_range": "p", "duration": "d", "roi_estimate": "r",
    }]
    result = pg._fallback_proposal(profile, health, services)

    assert "Коммерческое предложение" in result
    assert "IT" in result
    assert "Аудит" in result
    assert "Финансы" in result.replace("ё", "е") or "финансы" in result.lower()
    assert "Предлагаемые услуги" in result


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_build_services_section_empty_list():
    """Edge case: пустой список услуг возвращает пустую строку, а не падает."""
    assert pg._build_services_section([]) == ""


def test_build_services_section_missing_fields_use_dashes():
    """Edge case: услуга без необязательных полей рендерится с прочерками."""
    services = [{"id": "s1", "name": "Услуга-минимум"}]
    section = pg._build_services_section(services)

    assert "Услуга-минимум" in section
    assert "—" in section  # дефолтные прочерки для отсутствующих полей


def test_render_html_with_minimal_inputs():
    """Edge case: render_html работает на минимальных данных без assessment."""
    html = pg.render_html("# Заголовок\n\nТекст.", {"industry": "IT"})

    assert "<!DOCTYPE html>" in html
    assert "ПУЛЬС" in html
    assert "IT" in html


def test_simple_md_to_html_handles_nested_lists_and_bold():
    """Edge case: простой markdown→html парсер обрабатывает заголовки, списки, **жирный**."""
    md = "# H1\n## H2\n- пункт **важный**\n- второй\n\nПараграф **bold**."
    html = pg._simple_md_to_html(md)

    assert "<h1>H1</h1>" in html
    assert "<h2>H2</h2>" in html
    assert "<ul>" in html and "</ul>" in html
    assert "<strong>важный</strong>" in html
    assert "<strong>bold</strong>" in html


def test_simple_md_to_html_renders_tables():
    """Edge case: fallback markdown-рендерер сохраняет табличный блок услуг."""
    md = "## Услуги\n\n| Услуга | ROI |\n|---|---:|\n| Аудит | 300% |"
    html = pg._simple_md_to_html(md)

    assert "<table>" in html
    assert "<th>Услуга</th>" in html
    assert "<td>Аудит</td>" in html


def test_risk_html_wraps_long_generated_text():
    """Edge case: длинный AI-текст риска не должен вылезать из карточки."""
    long_risk = "ОченьДлинныйРискБезПробелов" * 12
    html = pg._risk_html([long_risk])

    assert long_risk in html
    assert "min-width:0" in html
    assert "overflow-wrap:anywhere" in html
    assert "word-break:break-word" in html


# ── Negative ──────────────────────────────────────────────────────────────────

def test_generate_proposal_text_falls_back_when_llm_raises(monkeypatch):
    """Негатив: если LLM-вызов падает, генератор использует _fallback_narrative + полный список услуг."""
    monkeypatch.setattr(pg, "_load_prompt", lambda name: {"system": "x"})

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("API down")

    profile = {"industry": "IT", "region": "СПб", "revenue_range": "100 млн", "main_challenge": "масштабирование"}
    assessment = {"health_assessment": {"zones": [{"name": "финансы", "score": 3, "risks": ["x"]}], "top_risks": []}}
    services = [{
        "id": "s1", "name": "Критичная услуга", "zone": "финансы",
        "description": "d", "expected_effect": "e",
        "price_range": "p", "duration": "d", "roi_estimate": "r",
    }]

    result = pg.generate_proposal_text(profile, assessment, services, fake_client)

    # Главное: услуга НЕ потеряна, даже когда LLM умер
    assert "Критичная услуга" in result
    assert "Коммерческое предложение" in result


def test_render_pdf_returns_none_on_invalid_html(monkeypatch):
    """Негатив: render_pdf возвращает None при ошибке (например, weasyprint недоступен)."""
    # Мокируем импорт weasyprint так, чтобы он бросал
    import sys
    monkeypatch.setitem(sys.modules, "weasyprint", None)

    # При None в sys.modules `from weasyprint import ...` вызовет ImportError
    result = pg.render_pdf("<html></html>")
    assert result is None
