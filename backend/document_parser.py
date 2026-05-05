"""Extracts facts and financial metrics from uploaded PDF, CSV, or Excel files."""
import io
import re
from collections import defaultdict

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
      - facts: short factual statements for the audit prompt
      - quotes: cited facts with source file names
      - financial_analysis: calculated indicators/ratios/risks
      - error: str | None
    """
    ext = _get_ext(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        log_agent_step("document_parser.parse_document", "unsupported", filename=filename, extension=ext)
        return _empty_result(error=UNSUPPORTED_MSG, filename=filename)

    try:
        if ext == ".pdf":
            result = _parse_pdf(file_bytes)
        elif ext == ".csv":
            result = _parse_csv(file_bytes)
        else:
            result = _parse_excel(file_bytes)
        result = _attach_filename(result, filename)
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
        return _empty_result(
            error=f"Не удалось обработать файл: {e}. Анализ продолжится на основе ответов опроса.",
            filename=filename,
        )


def merge_document_results(results: list[dict]) -> dict:
    """Merge several parsed uploads into one audit-ready document context."""
    valid = [r for r in results if r and not r.get("error")]
    errors = [f"{r.get('filename', 'файл')}: {r.get('error')}" for r in results if r and r.get("error")]
    if not valid:
        return _empty_result(error="\n".join(errors) if errors else None)

    merged_metrics = {}
    summaries = []
    facts = []
    quotes = []
    analyses = []
    source_files = []

    for result in valid:
        filename = result.get("filename", "файл")
        source_files.append(filename)
        if result.get("summary"):
            summaries.append(f"### {filename}\n{result['summary']}")
        for key, value in result.get("metrics", {}).items():
            metric_key = key if key not in merged_metrics else f"{filename}: {key}"
            merged_metrics[metric_key] = value
        facts.extend(result.get("facts", []))
        quotes.extend(result.get("quotes", []))
        if result.get("financial_analysis"):
            analyses.append(result["financial_analysis"])

    return {
        "filename": ", ".join(source_files),
        "source_files": source_files,
        "metrics": merged_metrics,
        "summary": "\n\n".join(summaries)[:12000],
        "facts": _dedupe_texts(facts)[:20],
        "quotes": quotes[:20],
        "financial_analysis": _merge_financial_analyses(analyses),
        "errors": errors,
        "error": None,
    }


def _get_ext(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return f".{parts[-1].lower()}" if len(parts) > 1 else ""


def _empty_result(error: str | None = None, filename: str | None = None) -> dict:
    return {
        "filename": filename or "",
        "metrics": {},
        "summary": "",
        "facts": [],
        "quotes": [],
        "financial_analysis": {
            "indicators": [],
            "ratios": [],
            "risks": [],
            "summary": "",
        },
        "error": error,
    }


def _attach_filename(result: dict, filename: str) -> dict:
    result["filename"] = filename
    for quote in result.get("quotes", []):
        if isinstance(quote, dict):
            if not quote.get("source") or quote.get("source") == "файл":
                quote["source"] = filename
    return result


def _parse_pdf(data: bytes) -> dict:
    import pdfplumber

    text_parts = []
    table_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_number, page in enumerate(pdf.pages[:10], 1):
            text = page.extract_text()
            if text:
                text_parts.append(f"[стр. {page_number}] {text}")
            for table in page.extract_tables() or []:
                table_text = _table_to_text(table)
                if table_text:
                    table_parts.append(f"[стр. {page_number}, таблица]\n{table_text}")

    full_text = "\n".join(text_parts)
    table_text = "\n".join(table_parts)
    combined = "\n\n".join(part for part in (full_text, table_text) if part.strip())
    if not combined.strip():
        return {
            "metrics": {},
            "summary": "",
            "facts": [],
            "quotes": [],
            "financial_analysis": _empty_result()["financial_analysis"],
            "error": "PDF не содержит распознаваемого текста (возможно, скан без OCR). Анализ продолжится по опросу."
        }

    metrics = _extract_numeric_mentions(combined)
    facts = _extract_key_facts(combined)
    return {
        "metrics": metrics,
        "summary": combined[:8000],
        "facts": facts,
        "quotes": _facts_to_quotes(facts),
        "financial_analysis": _analyze_text_financials(combined),
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

    financial_analysis = _analyze_dataframe_financials(df)
    facts = _dataframe_facts(df, metrics, financial_analysis)
    return {
        "metrics": metrics,
        "summary": summary[:8000],
        "facts": facts,
        "quotes": _facts_to_quotes(facts),
        "financial_analysis": financial_analysis,
        "error": None,
    }


def _extract_numeric_mentions(text: str) -> dict:
    """Rough extraction of labeled numbers from text (revenue, profit, etc.)."""
    number = r"(?:\d{1,3}(?:[\s.]\d{3})+|\d+)(?:[,.]\d+)?"
    patterns = {
        "выручка": rf"выручк[аеи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "чистая прибыль": rf"чист[а-я\s]+прибыл[ьи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "прибыль": rf"прибыл[ьи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "расходы": rf"(расход[а-я]*|затрат[а-я]*|opex)\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "себестоимость": rf"себестоимост[ьи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "денежный поток": rf"(денежн[а-я\s]+поток|cash\s*flow|ддс)\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "дебиторская задолженность": rf"дебиторск[а-я\s]+задолженност[ьи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "кредиторская задолженность": rf"кредиторск[а-я\s]+задолженност[ьи]\s*[:\-–]?\s*({number})\s*(млн|тыс|руб|₽)?",
        "сотрудники": r"(сотрудник|работник|персонал|штат)[а-я]*\s*[:\-–]?\s*([\d]+)",
        "клиенты": r"клиент[а-я]*\s*[:\-–]?\s*([\d]+)",
    }

    metrics = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[label] = match.group(0).strip()

    return metrics


# ── Fact extraction ───────────────────────────────────────────────────────────

FINANCIAL_KEYWORDS = (
    "выруч", "доход", "оборот", "приб", "убыт", "марж", "рентабель",
    "расход", "затрат", "себесто", "денеж", "cash", "ликвид", "дебитор",
    "кредитор", "долг", "ebitda", "клиент", "сотрудник", "конверси",
)


def _table_to_text(table: list[list[object]]) -> str:
    rows = []
    for row in table[:30]:
        cells = [str(cell).strip() for cell in row if cell not in (None, "")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_key_facts(text: str, limit: int = 12) -> list[str]:
    candidates = []
    for chunk in re.split(r"[\n\r]+|(?<=[.!?])\s+", text):
        clean = re.sub(r"\s+", " ", chunk).strip(" -;")
        if len(clean) < 12:
            continue
        lower = clean.lower()
        has_keyword = any(keyword in lower for keyword in FINANCIAL_KEYWORDS)
        has_number = bool(re.search(r"\d|₽|%", clean))
        if has_keyword and has_number:
            candidates.append(clean[:360])
    return _dedupe_texts(candidates)[:limit]


def _facts_to_quotes(facts: list[str]) -> list[dict]:
    return [{"source": "файл", "text": fact} for fact in facts[:12]]


def _dedupe_texts(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = re.sub(r"\s+", " ", str(value)).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(str(value).strip())
    return result


# ── Financial analysis ───────────────────────────────────────────────────────

TEXT_FINANCIAL_PATTERNS = {
    "выручка": r"выручк[аеи]|доход[ыа]?|оборот",
    "чистая прибыль": r"чист[а-я\s]+прибыл[ьи]|net\s*profit|net\s*income",
    "прибыль": r"прибыл[ьи]|profit",
    "операционная прибыль": r"операционн[а-я\s]+прибыл[ьи]|ebit",
    "расходы": r"расход[а-я]*|затрат[а-я]*|opex",
    "себестоимость": r"себестоимост[ьи]|cost\s*of\s*sales|cogs",
    "денежный поток": r"денежн[а-я\s]+поток|cash\s*flow|ддс",
    "дебиторская задолженность": r"дебиторск[а-я\s]+задолженност[ьи]|accounts\s*receivable",
    "кредиторская задолженность": r"кредиторск[а-я\s]+задолженност[ьи]|accounts\s*payable",
}

DATAFRAME_ALIASES = {
    "выручка": ("выруч", "доход", "оборот", "revenue", "sales"),
    "прибыль": ("приб", "profit", "net_income", "net income", "income"),
    "расходы": ("расход", "затрат", "expense", "cost", "opex"),
    "денежные средства": ("денеж", "остаток", "cash", "ддс"),
    "дебиторская задолженность": ("дебитор", "receivable", "ar"),
    "кредиторская задолженность": ("кредитор", "payable", "ap"),
}


def _analyze_text_financials(text: str) -> dict:
    mentions = _extract_structured_financial_mentions(text)
    indicators = []
    values = {}

    for label, items in mentions.items():
        if not items:
            continue
        item = items[0]
        values[label] = item["amount"]
        indicators.append({
            "name": _display_metric_name(label),
            "value": _format_amount(item["amount"]),
            "source": "PDF/текст",
            "fact": item["raw"],
        })

    return _build_financial_analysis(indicators, values)


def _extract_structured_financial_mentions(text: str) -> dict[str, list[dict]]:
    amount_re = (
        r"(?P<amount>-?(?:\d{1,3}(?:[\s.]\d{3})+|\d+)(?:[,.]\d+)?)"
        r"\s*(?P<unit>млн|миллион(?:а|ов)?|тыс|тысяч(?:а|и)?|руб(?:\.|лей|ля)?|₽)?"
    )
    result: dict[str, list[dict]] = defaultdict(list)

    for label, label_pattern in TEXT_FINANCIAL_PATTERNS.items():
        pattern = rf"(?P<label>{label_pattern})[^\n\r\d-]{{0,50}}{amount_re}"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            amount = _normalize_amount(match.group("amount"), match.group("unit"))
            if amount is None:
                continue
            raw = re.sub(r"\s+", " ", match.group(0)).strip()
            result[label].append({"amount": amount, "raw": raw})
            break
    return dict(result)


def _analyze_dataframe_financials(df) -> dict:
    import pandas as pd

    series_by_label = {}
    for label, aliases in DATAFRAME_ALIASES.items():
        col = _find_column(df, aliases)
        if col is None:
            continue
        series = _to_numeric_series(df[col])
        if not series.empty:
            series_by_label[label] = series

    indicators = []
    values = {}
    for label, series in series_by_label.items():
        total = float(series.sum())
        latest = float(series.iloc[-1])
        values[label] = total
        indicators.append({
            "name": f"{_display_metric_name(label)} за период",
            "value": _format_amount(total),
            "source": "таблица",
            "fact": f"Сумма по колонке «{_find_column(df, DATAFRAME_ALIASES[label])}»: {_format_amount(total)}; последнее значение: {_format_amount(latest)}",
        })

    analysis = _build_financial_analysis(indicators, values)

    revenue = series_by_label.get("выручка")
    if revenue is not None and len(revenue) >= 2:
        first = float(revenue.iloc[0])
        last = float(revenue.iloc[-1])
        if first:
            growth = (last - first) / abs(first) * 100
            analysis["ratios"].append({
                "name": "Динамика выручки",
                "value": _format_percent(growth),
                "interpretation": _growth_interpretation(growth),
            })
            if growth < 0:
                analysis["risks"].append("Выручка снижается по последним строкам отчётности; требуется анализ причин падения спроса, цен или объёмов.")

    profit = series_by_label.get("прибыль")
    if revenue is not None and profit is not None and len(revenue) and len(profit):
        last_revenue = float(revenue.iloc[-1])
        last_profit = float(profit.iloc[-1])
        if last_revenue:
            latest_margin = last_profit / last_revenue * 100
            analysis["ratios"].append({
                "name": "Маржинальность последнего периода",
                "value": _format_percent(latest_margin),
                "interpretation": _margin_interpretation(latest_margin),
            })

    analysis["summary"] = _financial_summary(analysis)
    return analysis


def _build_financial_analysis(indicators: list[dict], values: dict[str, float]) -> dict:
    ratios = []
    risks = []

    revenue = values.get("выручка")
    profit = values.get("чистая прибыль", values.get("прибыль"))
    expenses = values.get("расходы")
    cogs = values.get("себестоимость")
    receivables = values.get("дебиторская задолженность")
    payables = values.get("кредиторская задолженность")

    if revenue and profit is not None:
        margin = profit / revenue * 100
        ratios.append({
            "name": "Рентабельность по прибыли",
            "value": _format_percent(margin),
            "interpretation": _margin_interpretation(margin),
        })
        if margin < 0:
            risks.append("Бизнес показывает отрицательную прибыльность: операционная модель не покрывает расходы.")
        elif margin < 5:
            risks.append("Рентабельность ниже 5%: запас прочности к росту затрат и просадке продаж минимален.")

    cost_base = expenses if expenses is not None else cogs
    if revenue and cost_base is not None:
        cost_ratio = cost_base / revenue * 100
        ratios.append({
            "name": "Доля расходов в выручке",
            "value": _format_percent(cost_ratio),
            "interpretation": "Высокая нагрузка на выручку." if cost_ratio > 80 else "Расходы требуют регулярного план-факт контроля.",
        })
        if cost_ratio > 90:
            risks.append("Расходы съедают более 90% выручки; высок риск кассовых разрывов и убытка.")

    if receivables is not None and payables is not None:
        balance = receivables - payables
        ratios.append({
            "name": "Разница дебиторской и кредиторской задолженности",
            "value": _format_amount(balance),
            "interpretation": "Проверьте сроки оплаты и кассовый цикл, чтобы не финансировать клиентов за счёт оборотных средств.",
        })

    analysis = {
        "indicators": indicators[:12],
        "ratios": ratios[:8],
        "risks": _dedupe_texts(risks)[:8],
        "summary": "",
    }
    analysis["summary"] = _financial_summary(analysis)
    return analysis


def _merge_financial_analyses(analyses: list[dict]) -> dict:
    indicators = []
    ratios = []
    risks = []
    for analysis in analyses:
        indicators.extend(analysis.get("indicators", []))
        ratios.extend(analysis.get("ratios", []))
        risks.extend(analysis.get("risks", []))
    result = {
        "indicators": indicators[:20],
        "ratios": ratios[:12],
        "risks": _dedupe_texts(risks)[:12],
        "summary": "",
    }
    result["summary"] = _financial_summary(result)
    return result


def _dataframe_facts(df, metrics: dict, financial_analysis: dict) -> list[str]:
    facts = [
        f"Таблица содержит {df.shape[0]} строк и {df.shape[1]} столбцов: {', '.join(map(str, df.columns.tolist()[:12]))}."
    ]
    for col, values in list(metrics.items())[:8]:
        facts.append(
            f"По колонке «{col}»: сумма {_format_amount(values['sum'])}, среднее {_format_amount(values['mean'])}, последнее значение {_format_amount(values['last'])}."
        )
    for ratio in financial_analysis.get("ratios", [])[:4]:
        facts.append(f"{ratio['name']}: {ratio['value']} ({ratio.get('interpretation', '')}).")
    return facts[:12]


def _find_column(df, aliases: tuple[str, ...]) -> str | None:
    for col in df.columns:
        name = str(col).lower().replace("_", " ")
        if any(alias in name for alias in aliases):
            return col
    return None


def _to_numeric_series(series):
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        return series.dropna().astype(float)

    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").dropna().astype(float)


def _normalize_amount(raw_amount: str, unit: str | None) -> float | None:
    raw = raw_amount.replace(" ", "")
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", raw):
        raw = raw.replace(".", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    unit_l = (unit or "").lower()
    if unit_l.startswith("млн") or unit_l.startswith("миллион"):
        value *= 1_000_000
    elif unit_l.startswith("тыс") or unit_l.startswith("тысяч"):
        value *= 1_000
    return value


def _format_amount(value: float) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(float(value))
    if absolute >= 1_000_000_000:
        return f"{sign}{absolute / 1_000_000_000:.1f} млрд ₽".replace(".", ",")
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.1f} млн ₽".replace(".", ",")
    if absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.1f} тыс. ₽".replace(".", ",")
    return f"{sign}{absolute:.0f} ₽".replace(".", ",")


def _format_percent(value: float) -> str:
    return f"{value:.1f}%".replace(".", ",")


def _display_metric_name(label: str) -> str:
    return {
        "выручка": "Выручка",
        "чистая прибыль": "Чистая прибыль",
        "прибыль": "Прибыль",
        "операционная прибыль": "Операционная прибыль",
        "расходы": "Расходы",
        "себестоимость": "Себестоимость",
        "денежный поток": "Денежный поток",
        "денежные средства": "Денежные средства",
        "дебиторская задолженность": "Дебиторская задолженность",
        "кредиторская задолженность": "Кредиторская задолженность",
    }.get(label, label.capitalize())


def _margin_interpretation(margin: float) -> str:
    if margin < 0:
        return "убыток, нужен срочный разбор структуры расходов и цен"
    if margin < 5:
        return "очень низкий запас прочности"
    if margin < 15:
        return "умеренная прибыльность, требуется контроль маржи"
    return "здоровая прибыльность при сохранении текущей структуры затрат"


def _growth_interpretation(growth: float) -> str:
    if growth < 0:
        return "отрицательная динамика"
    if growth < 10:
        return "слабый рост"
    return "положительная динамика"


def _financial_summary(analysis: dict) -> str:
    parts = []
    if analysis.get("ratios"):
        ratios_text = "; ".join(f"{r['name']} — {r['value']}" for r in analysis["ratios"][:3])
        parts.append(f"Рассчитаны ключевые коэффициенты: {ratios_text}.")
    if analysis.get("risks"):
        parts.append("Финансовые риски: " + " ".join(analysis["risks"][:2]))
    if not parts and analysis.get("indicators"):
        parts.append("Из отчётности извлечены базовые финансовые показатели; для глубокой интерпретации нужны детализация периодов и статей затрат.")
    return " ".join(parts)
