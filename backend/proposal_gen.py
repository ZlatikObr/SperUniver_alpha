"""Generates proposal text via Claude and renders it as a downloadable HTML."""
import yaml
from pathlib import Path
from datetime import date

import anthropic

MODEL = "claude-sonnet-4-6"


def _load_prompt(name: str) -> dict:
    path = Path("prompts") / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_proposal_text(
    profile: dict,
    assessment: dict,
    selected_services: list[dict],
    client: anthropic.Anthropic,
) -> str:
    prompt = _load_prompt("proposal_v1")

    health = assessment.get("health_assessment", {})
    zones_text = ""
    for z in health.get("zones", []):
        score = z.get("score", "?")
        risks = "; ".join(z.get("risks", []))
        zones_text += f"- **{z['name'].capitalize()}** (оценка {score}/5): {risks or 'без критических рисков'}\n"

    services_text = ""
    for svc in selected_services:
        services_text += (
            f"### {svc['name']}\n"
            f"Описание: {svc['description']}\n"
            f"Ожидаемый эффект: {svc['expected_effect']}\n"
            f"Стоимость: {svc['price_range']} | Срок: {svc['duration']} | ROI: {svc['roi_estimate']}\n\n"
        )

    top_risks = "\n".join(f"- {r}" for r in health.get("top_risks", []))

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

## Выбранные клиентом услуги
{services_text}

Сформируй профессиональное КП в формате Markdown.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=prompt["system"],
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def render_html(proposal_markdown: str, profile: dict) -> str:
    """Convert markdown proposal to a standalone HTML file with print-ready styles."""
    try:
        import markdown as md_lib
        body_html = md_lib.markdown(
            proposal_markdown,
            extensions=["tables", "fenced_code"]
        )
    except ImportError:
        body_html = _simple_md_to_html(proposal_markdown)

    company_info = f"{profile.get('industry', '')} · {profile.get('region', '')} · {profile.get('revenue_range', '')}"
    today = date.today().strftime("%d.%m.%Y")

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>КП — Ревизор</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', Arial, sans-serif; font-size: 14px; color: #1a1a2e; background: #fff; }}
  .wrapper {{ max-width: 800px; margin: 0 auto; padding: 40px 48px; }}
  .header {{ border-bottom: 3px solid #2d6a4f; padding-bottom: 20px; margin-bottom: 32px; }}
  .header h1 {{ font-size: 26px; font-weight: 700; color: #1b4332; }}
  .header .meta {{ font-size: 13px; color: #6c757d; margin-top: 6px; }}
  h2 {{ font-size: 18px; font-weight: 600; color: #1b4332; margin: 28px 0 12px; border-left: 4px solid #2d6a4f; padding-left: 10px; }}
  h3 {{ font-size: 15px; font-weight: 600; color: #2d6a4f; margin: 20px 0 8px; }}
  p {{ line-height: 1.7; margin-bottom: 10px; }}
  ul, ol {{ padding-left: 22px; margin-bottom: 12px; }}
  li {{ line-height: 1.7; margin-bottom: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
  th {{ background: #1b4332; color: #fff; padding: 8px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #dee2e6; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  strong {{ color: #1b4332; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d; font-style: italic; }}
  @media print {{
    .no-print {{ display: none; }}
    body {{ font-size: 12px; }}
    .wrapper {{ padding: 20px; }}
  }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>Коммерческое предложение</h1>
    <div class="meta">{company_info} &nbsp;|&nbsp; Дата: {today} &nbsp;|&nbsp; Подготовлено: Ревизор AI</div>
  </div>

  {body_html}

  <div class="footer">
    Анализ носит рекомендательный характер и подготовлен на основе предоставленных данных.
    Для принятия управленческих решений рекомендуется верификация с профильным консультантом.
  </div>
</div>
</body>
</html>"""


def _simple_md_to_html(text: str) -> str:
    """Minimal markdown → HTML conversion without external dependencies."""
    import re

    lines = text.split("\n")
    html_parts = []
    in_list = False

    for line in lines:
        if line.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            item = line[2:]
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            html_parts.append(f"<li>{item}</li>")
        elif line.strip() == "":
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append("<br>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            html_parts.append(f"<p>{line}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)
