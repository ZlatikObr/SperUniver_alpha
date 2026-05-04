"""Ревизор — AI-инструмент первичного бизнес-скрининга."""
import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure backend is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

import anthropic
from backend.survey import get_base_questions, generate_followup_questions, build_business_profile
from backend.document_parser import parse_document
from backend.auditor import analyze_business
from backend.catalog import load_catalog, filter_services, get_services_by_ids
from backend.proposal_gen import generate_proposal_text, render_html

# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ревизор — бизнес-диагностика",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SCORE_COLORS = {1: "#dc3545", 2: "#fd7e14", 3: "#ffc107", 4: "#20c997", 5: "#198754"}
SCORE_LABELS = {
    1: "Критический уровень",
    2: "Серьёзные проблемы",
    3: "Требует внимания",
    4: "Удовлетворительно",
    5: "Сильная зона",
}

STEPS = [
    "welcome",
    "survey_base",
    "survey_followup",
    "document",
    "analyzing",
    "diagnostics",
    "catalog",
    "generating",
    "proposal",
]


# ─── Session helpers ──────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "step": "welcome",
        "base_answers": {},
        "followup_questions": [],
        "followup_answers": {},
        "doc_result": None,
        "business_profile": {},
        "assessment": {},
        "catalog_services": [],
        "selected_ids": [],
        "proposal_markdown": "",
        "proposal_html": "",
        "start_ts": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go(step: str):
    st.session_state.step = step
    st.rerun()


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("ANTHROPIC_API_KEY не задан. Добавьте его в файл .env")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _header(subtitle: str = ""):
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 4px;">
            <span style="font-size:32px; font-weight:800; color:#1b4332; letter-spacing:-1px;">🔍 Ревизор</span>
            <p style="color:#6c757d; font-size:14px; margin-top:4px;">AI-диагностика бизнеса · от первого контакта до КП за 20 минут</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"<p style='text-align:center; color:#495057; font-size:15px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()


def _progress(step: str):
    step_labels = {
        "survey_base": "Опрос",
        "survey_followup": "Уточнения",
        "document": "Документы",
        "analyzing": "Анализ",
        "diagnostics": "Диагностика",
        "catalog": "Услуги",
        "generating": "КП",
        "proposal": "КП готово",
    }
    visible = ["survey_base", "survey_followup", "document", "analyzing", "diagnostics", "catalog", "generating", "proposal"]
    current_idx = visible.index(step) if step in visible else 0
    total = len(visible)

    progress_val = (current_idx + 1) / total
    label = step_labels.get(step, "")
    st.progress(progress_val, text=f"Шаг {current_idx + 1} из {total}: {label}")


def _zone_badge(name: str, score: int) -> str:
    color = SCORE_COLORS.get(score, "#6c757d")
    label = SCORE_LABELS.get(score, "")
    return (
        f'<span style="display:inline-block; background:{color}20; border:1px solid {color}; '
        f'color:{color}; border-radius:6px; padding:3px 10px; font-size:13px; font-weight:600;">'
        f'{name.capitalize()} — {score}/5 · {label}</span>'
    )


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_welcome():
    _header()
    st.markdown(
        """
        <div style="background:#f8f9fa; border-radius:12px; padding:28px 32px; margin-bottom:20px;">
        <h3 style="color:#1b4332; margin-bottom:12px;">Что делает Ревизор?</h3>
        <ul style="line-height:2; color:#343a40;">
          <li>🎯 Проводит адаптивный опрос из 10–15 вопросов</li>
          <li>📄 Извлекает метрики из вашей отчётности (PDF, Excel, CSV)</li>
          <li>🧠 Анализирует бизнес по 5 зонам: финансы, операции, маркетинг, команда, стратегия</li>
          <li>📋 Генерирует готовое КП с обоснованием и ROI-оценкой</li>
        </ul>
        <p style="color:#6c757d; font-size:13px; margin-top:12px;">⏱ Среднее время: 15–20 минут · Результат: диагностика + КП в PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Начать диагностику →", use_container_width=True, type="primary"):
            st.session_state.start_ts = time.time()
            _go("survey_base")

    st.markdown(
        "<p style='text-align:center; font-size:12px; color:#adb5bd; margin-top:20px;'>"
        "Анализ носит рекомендательный характер. Данные сессии не сохраняются.</p>",
        unsafe_allow_html=True,
    )


def page_survey_base():
    _header("Шаг 1 — Расскажите о вашем бизнесе")
    _progress("survey_base")

    questions = get_base_questions()

    with st.form("base_form"):
        answers = {}
        for q in questions:
            label = q["text"]
            qid = q["id"]
            qtype = q["type"]

            if qtype == "select":
                answers[qid] = st.selectbox(label, q["options"], key=f"base_{qid}")
            elif qtype == "textarea":
                answers[qid] = st.text_area(label, placeholder=q.get("placeholder", ""), key=f"base_{qid}", height=100)
            else:
                answers[qid] = st.text_input(label, placeholder=q.get("placeholder", ""), key=f"base_{qid}")

        submitted = st.form_submit_button("Продолжить →", use_container_width=True, type="primary")

    if submitted:
        missing = [q["text"] for q in questions if q["type"] != "textarea" and not answers.get(q["id"])]
        if answers.get("main_challenge", "").strip() == "":
            missing.append("Опишите главную проблему")
        if missing:
            st.warning("Пожалуйста, ответьте на все вопросы.")
        else:
            st.session_state.base_answers = answers
            with st.spinner("Формирую уточняющие вопросы..."):
                client = _get_client()
                followup = generate_followup_questions(answers, client)
            st.session_state.followup_questions = followup
            _go("survey_followup")


def page_survey_followup():
    _header("Шаг 2 — Уточняющие вопросы")
    _progress("survey_followup")

    questions = st.session_state.followup_questions
    if not questions:
        _go("document")
        return

    with st.form("followup_form"):
        answers = {}
        for q in questions:
            qid = q.get("id", "q")
            label = q.get("text", "")
            qtype = q.get("type", "text")

            if qtype == "select" and q.get("options"):
                answers[qid] = st.selectbox(label, q["options"], key=f"fu_{qid}")
            else:
                answers[qid] = st.text_area(label, key=f"fu_{qid}", height=80)

        col1, col2 = st.columns(2)
        with col1:
            skip = st.form_submit_button("Пропустить", use_container_width=True)
        with col2:
            submitted = st.form_submit_button("Продолжить →", use_container_width=True, type="primary")

    if submitted or skip:
        st.session_state.followup_answers = answers if submitted else {}
        _go("document")


def page_document():
    _header("Шаг 3 — Документы (необязательно)")
    _progress("document")

    st.markdown(
        "Прикрепите финансовую или управленческую отчётность — это улучшит качество диагностики. "
        "Поддерживаются: **PDF, CSV, Excel (.xlsx/.xls)**."
    )
    st.caption("Документ используется только в рамках текущей сессии и не сохраняется.")

    uploaded = st.file_uploader(
        "Загрузить документ",
        type=["pdf", "csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        skip = st.button("Пропустить — анализировать без документа", use_container_width=True)
    with col2:
        proceed = st.button("Продолжить с документом →", use_container_width=True, type="primary", disabled=uploaded is None)

    if skip:
        st.session_state.doc_result = None
        _run_analysis()

    if proceed and uploaded:
        with st.spinner("Обрабатываю документ..."):
            doc_result = parse_document(uploaded.read(), uploaded.name)
        if doc_result.get("error"):
            st.warning(doc_result["error"])
            if st.button("Продолжить без документа"):
                st.session_state.doc_result = None
                _run_analysis()
        else:
            st.session_state.doc_result = doc_result
            _run_analysis()


def _run_analysis():
    _go("analyzing")


def page_analyzing():
    _header("Анализирую ваш бизнес...")
    _progress("analyzing")

    with st.spinner("Claude анализирует профиль по 5 зонам. Это займёт до 3 минут..."):
        client = _get_client()

        profile = build_business_profile(
            st.session_state.base_answers,
            st.session_state.followup_answers,
            client,
        )
        st.session_state.business_profile = profile

        answer_count = len(st.session_state.base_answers) + len(st.session_state.followup_answers)
        assessment = analyze_business(
            profile,
            st.session_state.doc_result,
            client,
            answer_count=answer_count,
        )
        st.session_state.assessment = assessment

        zones = assessment.get("health_assessment", {}).get("zones", [])
        risk_zone_ids = [z["name"] for z in zones if z.get("score", 5) <= 3]
        if not risk_zone_ids:
            risk_zone_ids = assessment.get("recommended_zone_ids", ["финансы", "стратегия"])

        catalog = load_catalog()
        filtered = filter_services(catalog, risk_zone_ids, profile.get("industry", ""))
        st.session_state.catalog_services = filtered

    _go("diagnostics")


def page_diagnostics():
    _header("Диагностика бизнеса")
    _progress("diagnostics")

    assessment = st.session_state.assessment
    health = assessment.get("health_assessment", {})
    profile = st.session_state.business_profile

    bp = assessment.get("business_profile", {})
    st.markdown(
        f"**Профиль:** {bp.get('industry', profile.get('industry', '—'))} · "
        f"{profile.get('region', '—')} · {bp.get('revenue_range', profile.get('revenue_range', '—'))}",
    )

    if assessment.get("_partial"):
        st.warning("⚠️ Частичная диагностика: ответов на вопросы недостаточно для полного анализа.")

    sources = assessment.get("data_sources", [])
    st.caption(f"Источники данных: {', '.join(sources)}")

    st.markdown("### Оценка по зонам")
    zones = health.get("zones", [])
    if zones:
        for zone in zones:
            score = zone.get("score", 3)
            color = SCORE_COLORS.get(score, "#6c757d")
            with st.expander(f"{zone['name'].capitalize()} — {score}/5  {SCORE_LABELS.get(score, '')}"):
                risks = zone.get("risks", [])
                growth = zone.get("growth_points", [])
                if risks:
                    st.markdown("**Риски:**")
                    for r in risks:
                        st.markdown(f"- 🔴 {r}")
                if growth:
                    st.markdown("**Точки роста:**")
                    for g in growth:
                        st.markdown(f"- 🟢 {g}")
    else:
        st.info("Зоны не определены. Рекомендуем пройти опрос полностью.")

    top_risks = health.get("top_risks", [])
    if top_risks:
        st.markdown("### Топ-3 приоритетных риска")
        for i, risk in enumerate(top_risks[:3], 1):
            st.markdown(f"**{i}.** {risk}")

    overall = health.get("overall_health", "")
    if overall:
        st.info(f"**Вывод:** {overall}")

    st.caption(assessment.get("disclaimer", ""))
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Вернуться к опросу", use_container_width=True):
            _go("welcome")
    with col2:
        if st.button("Перейти к витрине услуг →", use_container_width=True, type="primary"):
            _go("catalog")


def page_catalog():
    _header("Витрина услуг")
    _progress("catalog")

    services = st.session_state.catalog_services
    if not services:
        st.warning("Услуги не найдены. Возможно, каталог пуст.")
        if st.button("Вернуться к диагностике"):
            _go("diagnostics")
        return

    st.markdown(
        f"На основе диагностики подобраны **{len(services)} услуги**, релевантные вашим зонам риска. "
        "Выберите те, которые хотите включить в КП."
    )

    selected = []
    for svc in services:
        zone_color = {"финансы": "#0d6efd", "операции": "#6f42c1", "маркетинг": "#d63384",
                      "команда": "#fd7e14", "стратегия": "#198754"}.get(svc.get("zone", ""), "#6c757d")

        with st.container():
            col1, col2 = st.columns([0.08, 0.92])
            with col1:
                checked = st.checkbox("", key=f"svc_{svc['id']}", value=True)
            with col2:
                st.markdown(
                    f"<span style='background:{zone_color}20; color:{zone_color}; border:1px solid {zone_color}; "
                    f"border-radius:4px; padding:1px 8px; font-size:12px;'>{svc.get('zone', '').capitalize()}</span> "
                    f"**{svc['name']}**",
                    unsafe_allow_html=True,
                )
                st.markdown(f"_{svc['description']}_")
                cols = st.columns(3)
                cols[0].metric("Стоимость", svc.get("price_range", "—"))
                cols[1].metric("Срок", svc.get("duration", "—"))
                cols[2].metric("ROI", svc.get("roi_estimate", "—"))
            if checked:
                selected.append(svc["id"])
            st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Диагностика", use_container_width=True):
            _go("diagnostics")
    with col2:
        if st.button(f"Сгенерировать КП ({len(selected)} услуги) →", use_container_width=True, type="primary",
                     disabled=len(selected) == 0):
            st.session_state.selected_ids = selected
            _go("generating")


def page_generating():
    _header("Генерирую коммерческое предложение...")
    _progress("generating")

    catalog = load_catalog()
    selected_services = get_services_by_ids(catalog, st.session_state.selected_ids)

    with st.spinner("Claude формирует текст КП с ROI-обоснованием. До 2 минут..."):
        client = _get_client()
        proposal_md = generate_proposal_text(
            st.session_state.business_profile,
            st.session_state.assessment,
            selected_services,
            client,
        )
        st.session_state.proposal_markdown = proposal_md
        st.session_state.proposal_html = render_html(proposal_md, st.session_state.business_profile)

    _go("proposal")


def page_proposal():
    _header("Коммерческое предложение готово")
    _progress("proposal")

    if st.session_state.start_ts:
        elapsed = int(time.time() - st.session_state.start_ts)
        mins, secs = divmod(elapsed, 60)
        st.success(f"✅ КП сформировано за {mins} мин {secs} сек")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇ Скачать КП (HTML → PDF через браузер)",
            data=st.session_state.proposal_html.encode("utf-8"),
            file_name="revisor_kp.html",
            mime="text/html",
            use_container_width=True,
            type="primary",
        )
    with col2:
        st.download_button(
            label="⬇ Скачать КП (Markdown)",
            data=st.session_state.proposal_markdown.encode("utf-8"),
            file_name="revisor_kp.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.caption("💡 Для PDF: откройте HTML-файл в браузере → Печать (Ctrl+P) → Сохранить как PDF")

    with st.expander("Предпросмотр КП", expanded=True):
        st.markdown(st.session_state.proposal_markdown)

    st.divider()
    if st.button("🔄 Начать новую диагностику", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _go("welcome")


# ─── Router ───────────────────────────────────────────────────────────────────

def main():
    _init_state()

    step = st.session_state.step
    router = {
        "welcome": page_welcome,
        "survey_base": page_survey_base,
        "survey_followup": page_survey_followup,
        "document": page_document,
        "analyzing": page_analyzing,
        "diagnostics": page_diagnostics,
        "catalog": page_catalog,
        "generating": page_generating,
        "proposal": page_proposal,
    }

    page_fn = router.get(step, page_welcome)
    page_fn()


if __name__ == "__main__":
    main()
