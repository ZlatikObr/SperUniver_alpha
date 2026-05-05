"""Extracts key financial/operational metrics from uploaded PDF, CSV, or Excel files."""
import io

from .agent_logger import log_agent_step


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls"}
UNSUPPORTED_MSG = (
    "Формат файла не поддерживается. Загрузите PDF, CSV или Excel (.xlsx/.xls). "
    "Анализ продолжится на основе ответов опроса."
)


def parse_document(file_bytes: bytes, filename: str) -> dict:
    """
    Returns dict with keys:
      - metrics: dict of extracted key-value pairs
      - summary: human-readable text of extracted content
      - error: str | None
    """
    ext = _get_ext(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        log_agent_step("document_parser.parse_document", "unsupported", filename=filename, extension=ext)
        return {"metrics": {}, "summary": "", "error": UNSUPPORTED_MSG}

    try:
        if ext == ".pdf":
            result = _parse_pdf(file_bytes)
        elif ext == ".csv":
            result = _parse_csv(file_bytes)
        else:
            result = _parse_excel(file_bytes)
        log_agent_step(
            "document_parser.parse_document",
            "success" if not result.get("error") else "error",
            filename=filename,
            extension=ext,
            metric_count=len(result.get("metrics", {})),
            error=result.get("error"),
        )
        return result
    except Exception as e:
        log_agent_step("document_parser.parse_document", "error", filename=filename, extension=ext, error=e)
        return {
            "metrics": {},
            "summary": "",
            "error": f"Не удалось обработать файл: {e}. Анализ продолжится на основе ответов опроса."
        }


def _get_ext(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return f".{parts[-1].lower()}" if len(parts) > 1 else ""


def _parse_pdf(data: bytes) -> dict:
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:10]:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    full_text = "\n".join(text_parts)
    if not full_text.strip():
        return {
            "metrics": {},
            "summary": "",
            "error": "PDF не содержит распознаваемого текста (возможно, скан без OCR). Анализ продолжится по опросу."
        }

    metrics = _extract_numeric_mentions(full_text)
    return {
        "metrics": metrics,
        "summary": full_text[:4000],
        "error": None,
    }


def _parse_csv(data: bytes) -> dict:
    import pandas as pd

    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
            break
        except Exception as exc:
            log_agent_step("document_parser.parse_csv", "encoding_failed", encoding=encoding, error=exc)
            continue
    else:
        return {"metrics": {}, "summary": "", "error": "Не удалось прочитать CSV-файл."}

    return _dataframe_to_result(df)


def _parse_excel(data: bytes) -> dict:
    import pandas as pd

    df = pd.read_excel(io.BytesIO(data))
    return _dataframe_to_result(df)


def _dataframe_to_result(df) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics = {}
    for col in numeric_cols[:20]:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            metrics[col] = {
                "sum": float(col_data.sum()),
                "mean": float(col_data.mean()),
                "last": float(col_data.iloc[-1]),
            }

    summary = f"Таблица: {df.shape[0]} строк, {df.shape[1]} столбцов.\nКолонки: {', '.join(df.columns.tolist()[:30])}\n"
    if not df.empty:
        summary += f"\nПервые строки:\n{df.head(5).to_string()}"

    return {"metrics": metrics, "summary": summary[:4000], "error": None}


def _extract_numeric_mentions(text: str) -> dict:
    """Rough extraction of labeled numbers from text (revenue, profit, etc.)."""
    import re

    number = r"(?:\d{1,3}(?:[\s.]\d{3})+|\d+)(?:[,.]\d+)?"
    patterns = {
        "выручка": rf"выручк[аеи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "прибыль": rf"прибыл[ьи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "сотрудники": r"(сотрудник|работник|персонал|штат)[а-я]*\s*[:\-–]?\s*([\d]+)",
        "клиенты": r"клиент[а-я]*\s*[:\-–]?\s*([\d]+)",
    }

    metrics = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[label] = match.group(0).strip()

    return metrics
