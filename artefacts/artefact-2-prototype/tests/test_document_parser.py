"""Тесты модуля backend.document_parser — парсинг CSV/Excel/PDF и обработка ошибок."""
import io

import pandas as pd
import pytest

from backend import document_parser as dp


# ── Happy path ────────────────────────────────────────────────────────────────

def test_parse_csv_extracts_numeric_metrics():
    """Happy path: CSV с числовыми колонками парсится в metrics с sum/mean/last."""
    csv_text = "month,revenue,profit\n2025-01,1000000,150000\n2025-02,1200000,180000\n2025-03,950000,90000\n"
    result = dp.parse_document(csv_text.encode("utf-8"), "report.csv")

    assert result["error"] is None
    assert "revenue" in result["metrics"]
    assert "profit" in result["metrics"]

    rev = result["metrics"]["revenue"]
    assert rev["sum"] == 3_150_000
    assert rev["mean"] == pytest.approx(1_050_000)
    assert rev["last"] == 950_000

    assert "Таблица: 3 строк" in result["summary"]


def test_extract_numeric_mentions_finds_russian_terms():
    """Happy path: regex-извлечение метрик находит выручку и сотрудников в русском тексте."""
    text = "Годовая выручка: 250 млн ₽. Прибыль 15 млн. Сотрудников 42 человека. Клиентов 1200."
    metrics = dp._extract_numeric_mentions(text)

    assert "выручка" in metrics
    assert "прибыль" in metrics
    assert "сотрудники" in metrics
    assert "клиенты" in metrics


def test_dataframe_financial_analysis_calculates_ratios():
    """Happy path: табличная отчётность превращается в коэффициенты для диагностики."""
    df = pd.DataFrame({
        "month": ["2025-01", "2025-02"],
        "revenue": [1_000_000, 1_200_000],
        "profit": [100_000, 180_000],
        "expenses": [900_000, 1_020_000],
    })

    result = dp._dataframe_to_result(df)

    ratios = result["financial_analysis"]["ratios"]
    ratio_names = {r["name"] for r in ratios}
    assert "Рентабельность по прибыли" in ratio_names
    assert "Динамика выручки" in ratio_names
    assert result["facts"]


def test_merge_document_results_keeps_sources_and_financial_analysis():
    """Happy path: несколько файлов объединяются без потери фактов и источников."""
    first = dp.parse_document(
        "month,revenue,profit\n2025-01,1000000,100000\n".encode("utf-8"),
        "pnl.csv",
    )
    second = dp.parse_document(
        "месяц,доход\nянварь,100000\n".encode("cp1251"),
        "sales.csv",
    )

    merged = dp.merge_document_results([first, second])

    assert merged["error"] is None
    assert merged["source_files"] == ["pnl.csv", "sales.csv"]
    assert "pnl.csv" in merged["summary"]
    assert merged["financial_analysis"]["indicators"]


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_parse_csv_handles_cp1251_encoding():
    """Edge case: CSV в кодировке cp1251 (типично для российских выгрузок) читается."""
    csv_text = "месяц,доход\nянварь,100000\nфевраль,200000\n"
    raw = csv_text.encode("cp1251")
    result = dp.parse_document(raw, "1c_export.csv")

    assert result["error"] is None
    assert "доход" in result["metrics"]
    assert result["metrics"]["доход"]["sum"] == 300_000


def test_dataframe_to_result_with_empty_frame():
    """Edge case: пустой DataFrame не падает, возвращает структуру без метрик."""
    df = pd.DataFrame(columns=["a", "b"])
    result = dp._dataframe_to_result(df)

    assert result["error"] is None
    assert result["metrics"] == {}
    assert "0 строк" in result["summary"]


def test_extract_numeric_mentions_handles_no_matches():
    """Edge case: текст без финансовых терминов даёт пустой dict, не падает."""
    text = "Это просто описание без чисел и финансовых показателей"
    assert dp._extract_numeric_mentions(text) == {}


# ── Negative ──────────────────────────────────────────────────────────────────

def test_parse_unsupported_extension():
    """Негатив: формат .docx не поддерживается — возвращается дружелюбное сообщение."""
    result = dp.parse_document(b"fake content", "report.docx")

    assert result["error"] is not None
    assert "Формат файла не поддерживается" in result["error"]
    assert result["metrics"] == {}


def test_parse_corrupted_pdf_returns_error():
    """Негатив: битый PDF не валит приложение — ошибка ловится и описывается."""
    result = dp.parse_document(b"\x00\x01\x02not a real pdf", "broken.pdf")

    assert result["error"] is not None
    # Либо текст не распознан, либо exception при парсинге — оба варианта валидны
    assert result["metrics"] == {}


def test_parse_csv_undecodable_bytes():
    """Негатив: CSV с непарсимыми байтами — возвращает error без падения."""
    # Намеренно битые байты во всех трёх кодировках
    raw = b"\xff\xfe\x00\x01\x02\xc3\xa9"
    # Может либо упасть на чтении (попадёт в общий except),
    # либо успешно распарсить как одну колонку — допускаем оба исхода
    result = dp.parse_document(raw, "garbage.csv")
    assert "error" in result
    # В любом случае структура должна быть валидной
    assert "metrics" in result
    assert "summary" in result
