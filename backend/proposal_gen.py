"""Generates proposal text, renders HTML with charts, exports PDF."""
import io
import base64
import html as html_lib
import logging
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from openai import OpenAI

from ._config import MODEL
from ._utils import load_prompt as _load_prompt
from .agent_logger import log_agent_step

logger = logging.getLogger(__name__)

SCORE_COLORS = {1: "#E53935", 2: "#FB8C00", 3: "#FDD835", 4: "#43A047", 5: "#1E88E5"}
ZONE_RU = {
    "финансы": "Финансы", "операции": "Операции",
    "маркетинг": "Маркетинг", "команда": "Команда", "стратегия": "Стратегия",
}


def _build_services_section(selected_services: list[dict]) -> str:
    """
    Build the services block programmatically — 100% guaranteed to include
    every service the user selected, regardless of LLM token limits.
    """
    if not selected_services:
        return ""

    lines = [
        "## Предлагаемые услуги",
        "",
        "| Услуга | Задача | Методология | Ожидаемый результат | ROI / срок |",
        "|---|---|---|---|---|",
    ]
    for svc in selected_services:
        name   = svc.get("name", "—")
        desc   = svc.get("description", "—")
        effect = svc.get("expected_effect", "—")
        dur    = svc.get("duration", "—")
        roi    = svc.get("roi_estimate", "—")
        price  = svc.get("price_range", "—")
        methodology = _service_methodology(svc)

        lines.append(
            "| "
            f"{_md_cell(name)} | "
            f"{_md_cell(desc)} | "
            f"{_md_cell(methodology)} | "
            f"{_md_cell(effect)} | "
            f"{_md_cell(f'{roi}; {dur}; {price}')} |"
        )

    return "\n".join(lines)


def _md_cell(value: object) -> str:
    return str(value or "—").replace("|", "\\|").replace("\n", " ").strip()


def _service_methodology(svc: dict) -> str:
    zone = (svc.get("zone") or "").lower()
    tags = ", ".join(svc.get("tags", [])[:4])
    base = {
        "финансы": "Анализ P&L, cash-flow, баланса, план-факт отклонений и unit-экономики.",
        "операции": "Картирование AS-IS/TO-BE, поиск узких мест, KPI и дорожная карта внедрения.",
        "маркетинг": "Аудит воронки, каналов, CAC/LTV, конверсий и конкурентного позиционирования.",
        "команда": "Интервью, оргдиагностика, оценка ролей, KPI, мотивации и управленческих ритмов.",
        "стратегия": "SWOT/PESTLE, сценарное моделирование, приоритизация инициатив и план реализации.",
    }.get(zone, "Диагностика текущего состояния, расчёт эффекта и план внедрения.")
    return f"{base} Фокус: {tags}." if tags else base


def _build_key_findings_section(assessment: dict) -> str:
    health = assessment.get("health_assessment", {})
    top_risks = health.get("top_risks", []) or []
    details = health.get("top_risk_details", []) or []
    if not top_risks and not details:
        return ""

    lines = ["## Ключевые выводы диагностики", ""]
    risks_count = max(len(top_risks), len(details))
    for idx in range(min(risks_count, 3)):
        detail = details[idx] if idx < len(details) and isinstance(details[idx], dict) else {}
        risk = detail.get("risk") or (top_risks[idx] if idx < len(top_risks) else "")
        if isinstance(risk, dict):
            risk = risk.get("risk") or risk.get("title") or str(risk)
        why = detail.get("why_critical") or "Риск влияет на устойчивость, денежный поток и управляемость бизнеса."
        evidence = _format_evidence(detail.get("evidence", []))
        lines.append(f"### {idx + 1}. {risk}")
        lines.append(f"{why}")
        if evidence:
            lines.append(f"**Факты/цитаты:** {evidence}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_financial_analysis_section(assessment: dict) -> str:
    analysis = assessment.get("document_analysis") or {}
    if not any(analysis.get(key) for key in ("indicators", "ratios", "risks", "summary")):
        return ""

    lines = ["## Финансовый анализ по приложенным файлам", ""]
    if analysis.get("summary"):
        lines.append(analysis["summary"])
        lines.append("")

    if analysis.get("indicators"):
        lines.extend([
            "| Показатель | Значение | Основание |",
            "|---|---:|---|",
        ])
        for item in analysis["indicators"][:10]:
            lines.append(
                f"| {_md_cell(item.get('name'))} | {_md_cell(item.get('value'))} | {_md_cell(item.get('fact', item.get('source', 'файл')))} |"
            )
        lines.append("")

    if analysis.get("ratios"):
        lines.extend([
            "| Коэффициент | Значение | Интерпретация |",
            "|---|---:|---|",
        ])
        for item in analysis["ratios"][:8]:
            lines.append(
                f"| {_md_cell(item.get('name'))} | {_md_cell(item.get('value'))} | {_md_cell(item.get('interpretation'))} |"
            )
        lines.append("")

    if analysis.get("risks"):
        lines.append("**Финансовые риски:**")
        for risk in analysis["risks"][:6]:
            lines.append(f"- {risk}")

    return "\n".join(lines).strip()


def _build_zone_evidence_section(assessment: dict) -> str:
    zones = assessment.get("health_assessment", {}).get("zones", []) or []
    if not zones:
        return ""

    lines = [
        "## Почему такие оценки по зонам",
        "",
        "| Зона | Оценка | Объяснение | Факты/цитаты |",
        "|---|---:|---|---|",
    ]
    for zone in zones:
        name = ZONE_RU.get(zone.get("name"), str(zone.get("name", "—")).capitalize())
        score = zone.get("score", "—")
        explanation = zone.get("score_explanation") or _default_zone_explanation(zone)
        evidence = _format_evidence(zone.get("evidence", [])) or "Данные диагностики без отдельной цитаты."
        lines.append(f"| {_md_cell(name)} | {_md_cell(f'{score}/5')} | {_md_cell(explanation)} | {_md_cell(evidence)} |")
    return "\n".join(lines)


def _default_zone_explanation(zone: dict) -> str:
    score = zone.get("score", "—")
    risks = zone.get("risks", [])
    growth = zone.get("growth_points", [])
    if risks:
        return f"Оценка {score}/5 связана с рисками: {'; '.join(map(str, risks[:2]))}."
    if growth:
        return f"Оценка {score}/5: зона стабильна, но есть точки роста: {'; '.join(map(str, growth[:2]))}."
    return f"Оценка {score}/5 поставлена по результатам диагностики."


def _format_evidence(evidence_items: list) -> str:
    if not evidence_items:
        return ""
    formatted = []
    for item in evidence_items[:3]:
        if isinstance(item, dict):
            source = item.get("source", "данные")
            quote = item.get("quote") or item.get("fact") or item.get("text") or ""
            if quote:
                formatted.append(f"{source}: «{quote}»")
        elif item:
            formatted.append(str(item))
    return "; ".join(formatted)


def generate_proposal_text(
    profile: dict,
    assessment: dict,
    selected_services: list[dict],
    client: OpenAI,
) -> str:
    """
    Strategy: LLM writes only the narrative (intro, diagnosis, recommendations,
    conclusion). The services section is assembled by code so ALL selected
    services are always present — LLM cannot omit or summarise them.
    """
    log_agent_step(
        "proposal_gen.generate_proposal_text",
        "start",
        service_count=len(selected_services),
        industry=profile.get("industry"),
    )
    try:
        prompt = _load_prompt("proposal_v1")
    except Exception as exc:
        logger.warning("Failed to load proposal prompt; using fallback prompt.", exc_info=True)
        log_agent_step("proposal_gen.generate_proposal_text", "prompt_fallback", error=exc)
        prompt = {"system": "Ты — бизнес-консультант. Составь профессиональное КП в формате Markdown."}

    health = assessment.get("health_assessment", {})

    zones_text = ""
    for z in health.get("zones", []):
        score = z.get("score", "?")
        risks = "; ".join(z.get("risks", []))
        zones_text += f"- **{z['name'].capitalize()}** (оценка {score}/5): {risks or 'без критических рисков'}\n"

    top_risks = "\n".join(f"- {r}" for r in health.get("top_risks", []))
    n_services = len(selected_services)
    zone_evidence_text = ""
    for z in health.get("zones", []):
        evidence = _format_evidence(z.get("evidence", []))
        explanation = z.get("score_explanation", "")
        if evidence or explanation:
            zone_evidence_text += (
                f"- **{z.get('name', '').capitalize()}**: {explanation} "
                f"Факты: {evidence}\n"
            )
    financial_context = ""
    doc_analysis = assessment.get("document_analysis") or {}
    if doc_analysis.get("summary"):
        financial_context += doc_analysis["summary"] + "\n"
    for ratio in doc_analysis.get("ratios", [])[:6]:
        financial_context += f"- {ratio.get('name')}: {ratio.get('value')} — {ratio.get('interpretation', '')}\n"

    # LLM prompt: narrative only — no services enumeration
    user_message = f"""
## Профиль клиента
Отрасль: {profile.get('industry', 'Не указано')}
Регион: {profile.get('region', 'Не указано')}
Размер: {profile.get('size', 'Не указано')} сотрудников
Выручка: {profile.get('revenue_range', 'Не указано')}
Основная проблема: {profile.get('main_challenge', 'Не указано')}

## Результаты диагностики по зонам
{zones_text}

## Топ-3 риска
{top_risks}

## Объяснения и доказательства по зонам
{zone_evidence_text or "Нет отдельных доказательств."}

## Финансовый контекст из приложенных файлов
{financial_context or "Файл не приложен или финансовые коэффициенты не рассчитаны."}

Клиент выбрал {n_services} услуг (они будут добавлены в КП отдельной таблицей автоматически — тебе перечислять их НЕ нужно).

Сформируй в Markdown только следующие разделы КП:
1. Заголовок и краткое резюме (2–3 абзаца о ситуации клиента и ценности работы с нами)
2. Стратегические рекомендации (3–5 пунктов, с опорой на факты диагностики)
3. Общий ожидаемый эффект (с числовой логикой, без выдуманных исходных данных)
4. Следующие шаги (как начать работу)
5. Почему мы (короткий блок доверия)

НЕ добавляй разделы «Ключевые выводы диагностики», «Финансовый анализ», «Почему такие оценки по зонам» и список услуг — они будут вставлены автоматически.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2500,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": user_message},
            ],
        )
        narrative = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM proposal generation failed; using fallback narrative.", exc_info=True)
        log_agent_step("proposal_gen.generate_proposal_text", "llm_error", error=exc)
        narrative = ""

    if not narrative:
        narrative = _fallback_narrative(profile, health)

    # Append deterministic sections built by code — always complete
    findings_section = _build_key_findings_section(assessment)
    financial_section = _build_financial_analysis_section(assessment)
    zone_section = _build_zone_evidence_section(assessment)
    services_section = _build_services_section(selected_services)

    result = "\n\n".join(
        section for section in (
            narrative,
            findings_section,
            financial_section,
            zone_section,
            services_section,
        )
        if section
    )
    log_agent_step("proposal_gen.generate_proposal_text", "success", output_chars=len(result))
    return result


def _fallback_narrative(profile: dict, health: dict) -> str:
    """Minimal narrative when LLM call fails — services appended separately by caller."""
    lines = [
        "# Коммерческое предложение",
        "",
        f"**Отрасль:** {profile.get('industry', '—')}  ",
        f"**Регион:** {profile.get('region', '—')}  ",
        f"**Выручка:** {profile.get('revenue_range', '—')}",
        "",
        "## Диагностика по зонам",
    ]
    for z in health.get("zones", []):
        lines.append(f"- **{z['name'].capitalize()}**: {z.get('score', '?')}/5")
    lines.append("")
    lines.append("---")
    lines.append("*Анализ носит рекомендательный характер.*")
    return "\n".join(lines)


def _fallback_proposal(profile: dict, health: dict, services: list[dict]) -> str:
    """Full fallback КП (used from app.py except-block)."""
    narrative = _fallback_narrative(profile, health)
    services_section = _build_services_section(services)
    assessment = {"health_assessment": health}
    return "\n\n".join(
        section for section in (
            narrative,
            _build_key_findings_section(assessment),
            _build_zone_evidence_section(assessment),
            services_section,
        )
        if section
    )


# ─── Chart generators ─────────────────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _radar_chart(zones: list[dict]) -> str:
    if not zones:
        return ""
    try:
        labels = [ZONE_RU.get(z["name"], z["name"].capitalize()) for z in zones]
        scores = [z.get("score", 3) for z in zones]
        N = len(labels)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores_plot = scores + [scores[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#FAFAFA")

        ax.plot(angles, scores_plot, "o-", linewidth=2, color="#FF5600")
        ax.fill(angles, scores_plot, alpha=0.15, color="#FF5600")

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, fontfamily="DejaVu Sans")
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#7b7b78")
        ax.tick_params(pad=8)
        ax.spines["polar"].set_color("#E0DED8")
        ax.grid(color="#E0DED8", linewidth=0.8)

        return _fig_to_base64(fig)
    except Exception as exc:
        logger.warning("Radar chart generation failed.", exc_info=True)
        log_agent_step("proposal_gen.radar_chart", "error", error=exc)
        return ""


def _bar_chart(zones: list[dict]) -> str:
    if not zones:
        return ""
    try:
        labels = [ZONE_RU.get(z["name"], z["name"].capitalize()) for z in zones]
        scores = [z.get("score", 3) for z in zones]
        colors = [SCORE_COLORS.get(s, "#7b7b78") for s in scores]

        fig, ax = plt.subplots(figsize=(6, 2.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        bars = ax.barh(labels, scores, color=colors, height=0.55, zorder=2)
        ax.set_xlim(0, 5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.xaxis.grid(True, color="#E0DED8", linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)

        for bar, score in zip(bars, scores):
            ax.text(
                score + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{score}/5", va="center", fontsize=10, fontweight="bold",
                color="#1E1E1E", fontfamily="DejaVu Sans",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E0DED8")
        ax.spines["bottom"].set_color("#E0DED8")
        ax.tick_params(colors="#1E1E1E", labelsize=10)

        legend_items = [
            mpatches.Patch(color="#E53935", label="Критично (1)"),
            mpatches.Patch(color="#FB8C00", label="Проблемно (2)"),
            mpatches.Patch(color="#FDD835", label="Внимание (3)"),
            mpatches.Patch(color="#43A047", label="Хорошо (4)"),
            mpatches.Patch(color="#1E88E5", label="Отлично (5)"),
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=7,
                  framealpha=0.8, edgecolor="#E0DED8")

        plt.tight_layout()
        return _fig_to_base64(fig)
    except Exception as exc:
        logger.warning("Bar chart generation failed.", exc_info=True)
        log_agent_step("proposal_gen.bar_chart", "error", error=exc)
        return ""


def _risk_chart(top_risks: list[str]) -> str:
    if not top_risks:
        return ""
    try:
        labels = [f"Риск {i+1}" for i in range(len(top_risks[:5]))]
        weights = list(range(len(labels), 0, -1))

        fig, ax = plt.subplots(figsize=(6, max(1.8, len(labels) * 0.6)))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        colors = ["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5"][:len(labels)]
        bars = ax.barh(labels[::-1], weights[::-1], color=colors[::-1], height=0.5, zorder=2)
        ax.set_xlim(0, max(weights) + 0.5)
        ax.xaxis.grid(True, color="#E0DED8", linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)
        ax.set_xticks([])

        for bar, label in zip(bars, labels[::-1]):
            ax.text(0.15, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=10,
                    color="white", fontweight="500", fontfamily="DejaVu Sans")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#E0DED8")
        ax.set_yticklabels([])
        ax.tick_params(left=False)

        plt.tight_layout()
        return _fig_to_base64(fig)
    except Exception as exc:
        logger.warning("Risk chart generation failed.", exc_info=True)
        log_agent_step("proposal_gen.risk_chart", "error", error=exc)
        return ""


def _risk_display_text(risk) -> str:
    if isinstance(risk, dict):
        return risk.get("risk") or risk.get("title") or risk.get("text") or str(risk)
    return str(risk)


def _risk_list_html(top_risks: list) -> str:
    if not top_risks:
        return ""
    items = []
    for i, risk in enumerate(top_risks[:5], 1):
        items.append(
            "<li>"
            f"<span class='risk-num'>{i}</span>"
            f"<span>{html_lib.escape(_risk_display_text(risk))}</span>"
            "</li>"
        )
    return f"<ol class='risk-list'>{''.join(items)}</ol>"


def _zone_explanations_html(zones: list[dict]) -> str:
    rows = []
    for zone in zones:
        explanation = zone.get("score_explanation") or _default_zone_explanation(zone)
        evidence = _format_evidence(zone.get("evidence", []))
        if not explanation and not evidence:
            continue
        name = ZONE_RU.get(zone.get("name"), str(zone.get("name", "—")).capitalize())
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(name)}</td>"
            f"<td>{html_lib.escape(str(zone.get('score', '—')))}/5</td>"
            f"<td>{html_lib.escape(explanation)}</td>"
            f"<td>{html_lib.escape(evidence or 'Данные диагностики')}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<div class='zone-evidence'>"
        "<h3>Почему такие оценки</h3>"
        "<table><thead><tr><th>Зона</th><th>Оценка</th><th>Объяснение</th><th>Факты/цитаты</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</div>"
    )


# ─── HTML renderer ────────────────────────────────────────────────────────────

def render_html(proposal_markdown: str, profile: dict, assessment: dict = None) -> str:
    try:
        import markdown as md_lib
        body_html = md_lib.markdown(proposal_markdown, extensions=["tables", "fenced_code"])
    except Exception as exc:
        logger.warning("Markdown package rendering failed; using simple renderer.", exc_info=True)
        log_agent_step("proposal_gen.render_html", "markdown_fallback", error=exc)
        body_html = _simple_md_to_html(proposal_markdown)

    company_info = f"{profile.get('industry', '')} · {profile.get('region', '')} · {profile.get('revenue_range', '')}"
    today = date.today().strftime("%d.%m.%Y")

    # Generate charts
    charts_html = ""
    if assessment:
        zones = assessment.get("health_assessment", {}).get("zones", [])
        top_risks = assessment.get("health_assessment", {}).get("top_risks", [])

        radar_b64 = _radar_chart(zones)
        bar_b64 = _bar_chart(zones)
        risk_b64 = _risk_chart(top_risks)
        risk_list_html = _risk_list_html(top_risks)
        zone_explanations = _zone_explanations_html(zones)

        if radar_b64 and bar_b64:
            charts_html = f"""
<div class="charts-section">
  <h2>Визуализация диагностики</h2>
  <div class="charts-grid">
    <div class="chart-card">
      <h3>Радар здоровья бизнеса</h3>
      <img src="data:image/png;base64,{radar_b64}" alt="Radar chart" style="width:100%;max-width:320px;display:block;margin:0 auto;">
    </div>
    <div class="chart-card">
      <h3>Оценка по зонам</h3>
      <img src="data:image/png;base64,{bar_b64}" alt="Bar chart" style="width:100%;display:block;">
    </div>
  </div>
  {"<div class='chart-card chart-full'><h3>Приоритет рисков</h3><img src='data:image/png;base64," + risk_b64 + "' alt='Risk chart' style='width:100%;display:block;'>" + risk_list_html + "</div>" if risk_b64 else ""}
  {zone_explanations}
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>КП — Пульс</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', Arial, sans-serif; font-size: 14px; color: #1E1E1E; background: #fff; }}
  .wrapper {{ max-width: 820px; margin: 0 auto; padding: 48px 56px; }}

  /* Header */
  .header {{ display: flex; justify-content: space-between; align-items: flex-start;
             border-bottom: 2px solid #FF5600; padding-bottom: 24px; margin-bottom: 36px; }}
  .brand {{ font-size: 22px; font-weight: 800; letter-spacing: 0; color: #1E1E1E; }}
  .brand-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background: #FF5600; margin-left: 4px; margin-bottom: 2px; }}
  .brand-sub {{ font-size: 11px; color: #7b7b78; letter-spacing: 0.06em; margin-top: 3px; }}
  .header-meta {{ text-align: right; font-size: 12px; color: #7b7b78; line-height: 1.8; }}

  /* Body */
  h2 {{ font-size: 18px; font-weight: 700; color: #1E1E1E;
        margin: 32px 0 14px; padding-left: 12px;
        border-left: 3px solid #FF5600; }}
  h3 {{ font-size: 14px; font-weight: 600; color: #1E1E1E; margin: 18px 0 8px; }}
  p {{ line-height: 1.75; margin-bottom: 10px; color: #1E1E1E; }}
  ul, ol {{ padding-left: 20px; margin-bottom: 14px; }}
  li {{ line-height: 1.75; margin-bottom: 4px; }}
  strong {{ color: #1E1E1E; font-weight: 600; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 12px; table-layout: fixed; }}
  th {{ background: #1E1E1E; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #E0DED8; vertical-align: top; overflow-wrap: anywhere; }}
  tr:nth-child(even) td {{ background: #F7F7F5; }}

  /* Charts */
  .charts-section {{ margin: 32px 0; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }}
  .chart-card {{ background: #F7F7F5; border: 1px solid #E0DED8; border-radius: 8px; padding: 20px; }}
  .chart-card h3 {{ font-size: 13px; color: #7b7b78; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 14px; }}
  .chart-full {{ grid-column: 1 / -1; }}
  .risk-list {{ list-style: none; padding-left: 0; margin: 12px 0 0; }}
  .risk-list li {{ display: flex; gap: 10px; align-items: flex-start; margin-bottom: 8px;
                   color: #1E1E1E; line-height: 1.55; }}
  .risk-num {{ flex: 0 0 auto; min-width: 22px; height: 22px; border-radius: 4px;
               background: #FF5600; color: #fff; font-size: 11px; font-weight: 700;
               display: inline-flex; align-items: center; justify-content: center; }}
  .zone-evidence {{ margin-top: 18px; }}

  /* Highlight box */
  .highlight {{ background: #FFF3EE; border-left: 3px solid #FF5600;
                border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 16px 0;
                font-size: 13px; line-height: 1.7; }}

  /* Footer */
  .footer {{ margin-top: 48px; padding-top: 18px; border-top: 1px solid #E0DED8;
             font-size: 11px; color: #7b7b78; font-style: italic; }}

  @media print {{
    body {{ font-size: 12px; }}
    .wrapper {{ padding: 24px 28px; }}
    .charts-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div>
      <div class="brand">ПУЛЬС <span class="brand-dot"></span></div>
      <div class="brand-sub">AI-ДИАГНОСТИКА БИЗНЕСА</div>
    </div>
    <div class="header-meta">
      {company_info}<br>
      Дата: {today}
    </div>
  </div>

  {charts_html}

  {body_html}

  <div class="footer">
    Анализ носит рекомендательный характер и подготовлен на основе предоставленных данных.
    Для принятия управленческих решений рекомендуется верификация с профильным консультантом.
    Подготовлено: Пульс AI · {today}
  </div>
</div>
</body>
</html>"""


def render_pdf(html: str) -> bytes | None:
    try:
        from weasyprint import HTML as WeasyprintHTML
        return WeasyprintHTML(string=html).write_pdf()
    except Exception as exc:
        logger.warning("WeasyPrint PDF rendering failed.", exc_info=True)
        log_agent_step("proposal_gen.render_pdf", "weasyprint_error", error=exc)

    import sys
    if "weasyprint" in sys.modules and sys.modules["weasyprint"] is None:
        return None

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(40, 800, "ПУЛЬС: PDF fallback. Скачайте HTML для полной версии с графиками.")
        pdf.showPage()
        pdf.save()
        return buffer.getvalue()
    except Exception as exc:
        logger.warning("ReportLab PDF fallback failed.", exc_info=True)
        log_agent_step("proposal_gen.render_pdf", "reportlab_error", error=exc)
        return None


def _simple_md_to_html(text: str) -> str:
    import re
    lines = text.split("\n")
    html_parts = []
    in_list = False
    i = 0

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            close_list()
            headers = _split_md_table_row(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_md_table_row(lines[i]))
                i += 1
            html_parts.append("<table><thead><tr>")
            for header in headers:
                html_parts.append(f"<th>{_inline_md_to_html(header)}</th>")
            html_parts.append("</tr></thead><tbody>")
            for row in rows:
                html_parts.append("<tr>")
                for cell in row[:len(headers)]:
                    html_parts.append(f"<td>{_inline_md_to_html(cell)}</td>")
                html_parts.append("</tr>")
            html_parts.append("</tbody></table>")
            continue

        if line.startswith("### "):
            close_list()
            html_parts.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            close_list()
            html_parts.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            close_list()
            html_parts.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            item = _inline_md_to_html(line[2:])
            html_parts.append(f"<li>{item}</li>")
        elif line.strip() == "":
            close_list()
            html_parts.append("<br>")
        else:
            close_list()
            html_parts.append(f"<p>{_inline_md_to_html(line)}</p>")
        i += 1

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def _split_md_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    cells = []
    current = []
    escaped = False
    for char in stripped:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = False
    cells.append("".join(current).strip())
    return cells


def _inline_md_to_html(text: str) -> str:
    import re

    escaped = html_lib.escape(str(text or ""))
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
