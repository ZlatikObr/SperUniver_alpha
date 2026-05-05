"""Пульс — AI-диагностика бизнеса."""
import os
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from backend.survey import get_base_questions, generate_followup_questions, build_business_profile
from backend.document_parser import parse_document
from backend.auditor import analyze_business
from backend.catalog import load_catalog, filter_services, get_services_by_ids
from backend.proposal_gen import generate_proposal_text, render_html, render_pdf

# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Пульс — диагностика бизнеса",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SCORE_COLORS = {1: "#E53935", 2: "#FB8C00", 3: "#FDD835", 4: "#43A047", 5: "#1E88E5"}
SCORE_LABELS = {1: "Критично", 2: "Проблемно", 3: "Требует внимания", 4: "Удовлетворительно", 5: "Сильная зона"}
ZONE_ICONS  = {"финансы": "₽", "операции": "⚙", "маркетинг": "◈", "команда": "◉", "стратегия": "◎"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── App background + dot grid ── */
.stApp {
    background-color: #F7F7F5 !important;
    background-image: radial-gradient(circle, #D8D6D0 1px, transparent 1px) !important;
    background-size: 28px 28px !important;
}

/* ── Left & right decorative bars ── */
.stApp::before, .stApp::after {
    content: '';
    position: fixed;
    top: 0;
    bottom: 0;
    width: 4px;
    z-index: 9999;
    pointer-events: none;
    background: linear-gradient(to bottom, #FF5600 0%, #FF5600 30%, transparent 100%);
}
.stApp::before { left: 0; }
.stApp::after  { right: 0; }

/* ── Content card ── */
.main .block-container {
    max-width: 680px !important;
    padding: 2.5rem 2rem 5rem !important;
    background: rgba(247, 247, 245, 0.92) !important;
    animation: fadeInUp 0.4s ease-out both;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,86,0,0.5); }
    50%       { box-shadow: 0 0 0 7px rgba(255,86,0,0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

/* ── Text colors ── */
.stApp, .main, .block-container,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
p, li, label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stExpander"] p,
[data-testid="stExpander"] li {
    color: #1E1E1E !important;
}
.stCaption, [data-testid="stCaptionContainer"] p { color: #7b7b78 !important; }

/* ── Primary buttons ── */
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #FF5600 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.15s, transform 0.1s !important;
    box-shadow: 0 2px 8px rgba(255,86,0,0.25) !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #E04E00 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(255,86,0,0.3) !important;
}

/* ── Secondary buttons ── */
.stButton > button[kind="secondary"],
.stFormSubmitButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"] {
    background: #fff !important;
    color: #1E1E1E !important;
    border: 1.5px solid #E0DED8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button[kind="secondary"]:hover {
    border-color: #FF5600 !important;
    color: #FF5600 !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #fff !important;
    border: 1.5px solid #E0DED8 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    color: #1E1E1E !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #FF5600 !important;
    box-shadow: 0 0 0 3px rgba(255,86,0,0.08) !important;
    outline: none !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #fff !important;
    border: 1.5px solid #E0DED8 !important;
    border-radius: 8px !important;
    color: #1E1E1E !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #1E1E1E !important;
}

/* ── Fix file uploader button doubling ── */
[data-testid="stFileUploaderDropzone"] input[type="file"] {
    opacity: 0 !important;
    position: absolute !important;
    width: 100% !important;
    height: 100% !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"] input[type="file"]::-webkit-file-upload-button {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] input[type="file"]::file-selector-button {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #fff !important;
    border: 2px dashed #E0DED8 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
    position: relative !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #FF5600 !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: #FF5600 !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #E0DED8 !important;
    border-radius: 4px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #fff !important;
    border: 1px solid #E0DED8 !important;
    border-radius: 10px !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #fff !important;
    border: 1px solid #E0DED8 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #7b7b78 !important; }
[data-testid="stMetricValue"] { font-size: 15px !important; font-weight: 700 !important; color: #1E1E1E !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 13px !important; }

hr { border: none !important; border-top: 1px solid #E0DED8 !important; margin: 1.5rem 0 !important; }
</style>
"""


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


def _get_client() -> OpenAI:
    # Streamlit Cloud secrets take priority, then .env
    api_key = st.secrets.get("AITUNNEL_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = os.getenv("AITUNNEL_API_KEY", "")
    if not api_key:
        st.error("AITUNNEL_API_KEY не задан. Добавьте в Streamlit Cloud → Settings → Secrets.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.aitunnel.ru/v1/")


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _logo():
    st.markdown(
        """
        <div style="margin-bottom:2rem;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:22px;font-weight:800;letter-spacing:-0.5px;color:#1E1E1E;">ПУЛЬС</span>
            <span style="width:8px;height:8px;border-radius:50%;background:#FF5600;display:inline-block;
                         animation:pulse-dot 2s infinite;"></span>
          </div>
          <div style="font-size:11px;color:#7b7b78;letter-spacing:0.06em;margin-top:2px;">AI-ДИАГНОСТИКА БИЗНЕСА</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _progress(step: str):
    visible = ["survey_base","survey_followup","document","analyzing","diagnostics","catalog","generating","proposal"]
    labels  = ["Опрос","Уточнения","Данные","Анализ","Диагностика","Услуги","КП","Готово"]
    if step not in visible:
        return
    current = visible.index(step)
    pills = ""
    for i, label in enumerate(labels):
        if i < current:
            bg, color, weight = "#FF5600", "#fff", "600"
        elif i == current:
            bg, color, weight = "#1E1E1E", "#fff", "700"
        else:
            bg, color, weight = "#E0DED8", "#7b7b78", "400"
        pills += (
            f'<div style="background:{bg};color:{color};border-radius:20px;'
            f'padding:4px 12px;font-size:11px;font-weight:{weight};white-space:nowrap;">'
            f'{label}</div>'
        )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:2rem;">{pills}</div>',
        unsafe_allow_html=True,
    )


def _section(title: str):
    st.markdown(
        f'<p style="font-size:11px;font-weight:700;letter-spacing:0.08em;color:#7b7b78;'
        f'text-transform:uppercase;margin:1.5rem 0 0.75rem;">{title}</p>',
        unsafe_allow_html=True,
    )


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_welcome():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()

    st.markdown(
        """
        <div style="margin-bottom:2.5rem;">
          <h1 style="font-size:38px;font-weight:800;line-height:1.1;color:#1E1E1E;letter-spacing:-1.5px;margin-bottom:14px;">
            Диагностика бизнеса<br>за&nbsp;20&nbsp;минут
          </h1>
          <p style="font-size:16px;color:#7b7b78;line-height:1.65;max-width:480px;">
            Ответьте на вопросы — получите анализ по 5 зонам и готовое коммерческое предложение с ROI-обоснованием.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:2.5rem;">
          <div style="background:#fff;border:1px solid #E0DED8;border-radius:12px;padding:22px;
                      transition:box-shadow 0.2s;animation:fadeInUp 0.4s 0.05s both;">
            <div style="font-size:22px;margin-bottom:10px;">◉</div>
            <div style="font-size:13px;font-weight:600;color:#1E1E1E;margin-bottom:5px;">Адаптивный опрос</div>
            <div style="font-size:12px;color:#7b7b78;line-height:1.5;">10–13 вопросов, уточняющихся под ваш бизнес</div>
          </div>
          <div style="background:#fff;border:1px solid #E0DED8;border-radius:12px;padding:22px;
                      animation:fadeInUp 0.4s 0.1s both;">
            <div style="font-size:22px;margin-bottom:10px;">◈</div>
            <div style="font-size:13px;font-weight:600;color:#1E1E1E;margin-bottom:5px;">5-зонный анализ</div>
            <div style="font-size:12px;color:#7b7b78;line-height:1.5;">Финансы, операции, маркетинг, команда, стратегия</div>
          </div>
          <div style="background:#fff;border:1px solid #E0DED8;border-radius:12px;padding:22px;
                      animation:fadeInUp 0.4s 0.15s both;">
            <div style="font-size:22px;margin-bottom:10px;">◎</div>
            <div style="font-size:13px;font-weight:600;color:#1E1E1E;margin-bottom:5px;">Графики и диаграммы</div>
            <div style="font-size:12px;color:#7b7b78;line-height:1.5;">Радар, баровые графики, приоритизация рисков</div>
          </div>
          <div style="background:#FF5600;border-radius:12px;padding:22px;
                      animation:fadeInUp 0.4s 0.2s both;">
            <div style="font-size:22px;margin-bottom:10px;color:#fff;">↓</div>
            <div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:5px;">Готовое КП в PDF</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.75);line-height:1.5;">Скачать напрямую — без лишних шагов</div>
          </div>
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
        '<p style="text-align:center;font-size:11px;color:#C3C2BD;margin-top:16px;">'
        'Данные сессии не сохраняются · Анализ носит рекомендательный характер</p>',
        unsafe_allow_html=True,
    )


def page_survey_base():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("survey_base")

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">О вашем бизнесе</h2>'
        '<p style="font-size:13px;color:#7b7b78;margin-bottom:1.5rem;">Заполните все поля — это займёт 3–5 минут</p>',
        unsafe_allow_html=True,
    )

    questions = get_base_questions()
    with st.form("base_form"):
        answers = {}
        for q in questions:
            label = q["text"]
            qid   = q["id"]
            qtype = q["type"]
            if qtype == "select":
                answers[qid] = st.selectbox(label, q["options"], key=f"base_{qid}")
            elif qtype == "textarea":
                answers[qid] = st.text_area(label, placeholder=q.get("placeholder", ""), key=f"base_{qid}", height=90)
            else:
                answers[qid] = st.text_input(label, placeholder=q.get("placeholder", ""), key=f"base_{qid}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Продолжить →", use_container_width=True, type="primary")

    if submitted:
        if answers.get("main_challenge", "").strip() == "":
            st.warning("Пожалуйста, опишите главную проблему.")
        else:
            st.session_state.base_answers = answers
            with st.spinner("Формирую уточняющие вопросы..."):
                followup = generate_followup_questions(answers, _get_client())
            st.session_state.followup_questions = followup
            _go("survey_followup")


def page_survey_followup():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("survey_followup")

    questions = st.session_state.followup_questions
    if not questions:
        _go("document")
        return

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">Уточняющие вопросы</h2>'
        '<p style="font-size:13px;color:#7b7b78;margin-bottom:1.5rem;">Сформированы на основе ваших ответов</p>',
        unsafe_allow_html=True,
    )

    with st.form("followup_form"):
        answers = {}
        for q in questions:
            qid   = q.get("id", "q")
            label = q.get("text", "")
            qtype = q.get("type", "text")
            if qtype == "select" and q.get("options"):
                answers[qid] = st.selectbox(label, q["options"], key=f"fu_{qid}")
            else:
                answers[qid] = st.text_area(label, key=f"fu_{qid}", height=80)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            skip = st.form_submit_button("Пропустить", use_container_width=True)
        with col2:
            submitted = st.form_submit_button("Продолжить →", use_container_width=True, type="primary")

    if submitted or skip:
        st.session_state.followup_answers = answers if submitted else {}
        _go("document")


def page_document():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("document")

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">Финансовые данные</h2>'
        '<p style="font-size:13px;color:#7b7b78;margin-bottom:1.5rem;">'
        'Прикрепите отчётность для более точного анализа — PDF, CSV или Excel.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "file",
        type=["pdf", "csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        skip = st.button("Пропустить", use_container_width=True)
    with col2:
        proceed = st.button("Продолжить →", use_container_width=True, type="primary", disabled=uploaded is None)

    st.markdown(
        '<p style="font-size:11px;color:#C3C2BD;margin-top:10px;">'
        'Файл используется только в рамках сессии и не сохраняется</p>',
        unsafe_allow_html=True,
    )

    if skip:
        st.session_state.doc_result = None
        _go("analyzing")

    if proceed and uploaded:
        with st.spinner("Обрабатываю документ..."):
            doc_result = parse_document(uploaded.read(), uploaded.name)
        if doc_result.get("error"):
            st.warning(doc_result["error"])
            if st.button("Продолжить без документа"):
                st.session_state.doc_result = None
                _go("analyzing")
        else:
            st.session_state.doc_result = doc_result
            _go("analyzing")


def page_analyzing():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("analyzing")

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">Анализирую бизнес</h2>'
        '<p style="font-size:13px;color:#7b7b78;margin-bottom:1.5rem;">~30–60 секунд</p>',
        unsafe_allow_html=True,
    )

    client = _get_client()

    # Используем st.empty() — надёжнее st.status() в разных версиях Streamlit
    progress_placeholder = st.empty()

    def show_step(msg: str):
        progress_placeholder.markdown(
            f'<div style="background:#fff;border:1px solid #E0DED8;border-radius:8px;'
            f'padding:12px 18px;font-size:13px;color:#1E1E1E;">{msg}</div>',
            unsafe_allow_html=True,
        )

    show_step("🧠 Строю профиль бизнеса...")
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
        on_progress=show_step,
    )
    st.session_state.assessment = assessment

    show_step("📋 Формирую витрину услуг...")
    zones         = assessment.get("health_assessment", {}).get("zones", [])
    risk_zone_ids = [z["name"] for z in zones if z.get("score", 5) <= 3]
    if not risk_zone_ids:
        risk_zone_ids = assessment.get("recommended_zone_ids", ["финансы", "стратегия"])

    catalog  = load_catalog()
    filtered = filter_services(catalog, risk_zone_ids, profile.get("industry", ""))
    st.session_state.catalog_services = filtered

    show_step("✅ Анализ завершён!")

    _go("diagnostics")


def page_diagnostics():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("diagnostics")

    assessment = st.session_state.assessment
    health     = assessment.get("health_assessment", {})
    profile    = st.session_state.business_profile
    bp         = assessment.get("business_profile", {})

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">Результаты диагностики</h2>',
        unsafe_allow_html=True,
    )

    industry = bp.get("industry", profile.get("industry", "—"))
    region   = profile.get("region", "—")
    revenue  = bp.get("revenue_range", profile.get("revenue_range", "—"))
    sources  = ", ".join(assessment.get("data_sources", ["опрос"]))
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.5rem;">'
        f'<span style="background:#fff;border:1px solid #E0DED8;border-radius:20px;padding:4px 14px;font-size:12px;color:#1E1E1E;">{industry}</span>'
        f'<span style="background:#fff;border:1px solid #E0DED8;border-radius:20px;padding:4px 14px;font-size:12px;color:#1E1E1E;">{region}</span>'
        f'<span style="background:#fff;border:1px solid #E0DED8;border-radius:20px;padding:4px 14px;font-size:12px;color:#1E1E1E;">{revenue}</span>'
        f'<span style="background:#FFF3EE;border:1px solid #FFD5C0;border-radius:20px;padding:4px 14px;font-size:12px;color:#FF5600;">Источник: {sources}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if assessment.get("_partial"):
        st.warning("Частичная диагностика: для полного анализа ответьте на все вопросы.")

    _section("ОЦЕНКА ПО ЗОНАМ")
    zones = health.get("zones", [])
    if zones:
        for zone in zones:
            score  = zone.get("score", 3)
            color  = SCORE_COLORS.get(score, "#7b7b78")
            label  = SCORE_LABELS.get(score, "")
            icon   = ZONE_ICONS.get(zone["name"], "◉")
            risks  = zone.get("risks", [])
            growth = zone.get("growth_points", [])

            with st.expander(f"{icon}  {zone['name'].capitalize()}  ·  {score}/5  ·  {label}"):
                st.markdown(
                    f'<div style="height:4px;background:#E0DED8;border-radius:4px;margin-bottom:14px;">'
                    f'<div style="height:4px;width:{score*20}%;background:{color};border-radius:4px;"></div></div>',
                    unsafe_allow_html=True,
                )
                if risks:
                    st.markdown(
                        '<p style="font-size:11px;font-weight:700;color:#7b7b78;letter-spacing:0.06em;margin-bottom:6px;">РИСКИ</p>',
                        unsafe_allow_html=True,
                    )
                    for r in risks:
                        st.markdown(f"— {r}")
                if growth:
                    st.markdown(
                        '<p style="font-size:11px;font-weight:700;color:#7b7b78;letter-spacing:0.06em;margin:12px 0 6px;">ТОЧКИ РОСТА</p>',
                        unsafe_allow_html=True,
                    )
                    for g in growth:
                        st.markdown(f"+ {g}")
    else:
        st.info("Зоны не определены. Пройдите опрос полностью.")

    top_risks = health.get("top_risks", [])
    if top_risks:
        _section("ПРИОРИТЕТНЫЕ РИСКИ")
        for i, risk in enumerate(top_risks[:3], 1):
            st.markdown(
                f'<div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;">'
                f'<span style="background:#FF5600;color:#fff;border-radius:4px;padding:1px 8px;'
                f'font-size:12px;font-weight:700;min-width:22px;text-align:center;">{i}</span>'
                f'<span style="font-size:14px;color:#1E1E1E;line-height:1.5;">{risk}</span></div>',
                unsafe_allow_html=True,
            )

    overall = health.get("overall_health", "")
    if overall:
        st.markdown(
            f'<div style="background:#FFF3EE;border-left:3px solid #FF5600;border-radius:0 8px 8px 0;'
            f'padding:14px 18px;margin:1.5rem 0;font-size:14px;color:#1E1E1E;line-height:1.6;">'
            f'{overall}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<p style="font-size:11px;color:#C3C2BD;margin-bottom:1.5rem;">{assessment.get("disclaimer","")}</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Начать заново", use_container_width=True):
            _go("welcome")
    with col2:
        if st.button("Перейти к услугам →", use_container_width=True, type="primary"):
            _go("catalog")


def page_catalog():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("catalog")

    services = st.session_state.catalog_services
    if not services:
        st.warning("Услуги не найдены.")
        if st.button("← Диагностика"):
            _go("diagnostics")
        return

    st.markdown(
        f'<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">Рекомендованные услуги</h2>'
        f'<p style="font-size:13px;color:#7b7b78;margin-bottom:1.5rem;">'
        f'Подобрано {len(services)} услуг по результатам диагностики.</p>',
        unsafe_allow_html=True,
    )

    ZONE_COLORS = {
        "финансы": "#1E88E5", "операции": "#8E24AA", "маркетинг": "#D81B60",
        "команда": "#FB8C00", "стратегия": "#43A047",
    }

    selected = []
    for svc in services:
        zone       = svc.get("zone", "")
        zone_color = ZONE_COLORS.get(zone, "#7b7b78")

        col1, col2 = st.columns([0.06, 0.94])
        with col1:
            checked = st.checkbox("", key=f"svc_{svc['id']}", value=True)
        with col2:
            st.markdown(
                f'<div style="background:#fff;border:1px solid #E0DED8;border-radius:10px;'
                f'padding:16px 20px;margin-bottom:2px;transition:box-shadow 0.2s;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="background:{zone_color}15;color:{zone_color};border:1px solid {zone_color}30;'
                f'border-radius:4px;padding:1px 8px;font-size:11px;font-weight:600;">{zone.capitalize()}</span>'
                f'<span style="font-size:14px;font-weight:600;color:#1E1E1E;">{svc["name"]}</span>'
                f'</div>'
                f'<p style="font-size:13px;color:#7b7b78;margin-bottom:10px;line-height:1.5;">{svc["description"]}</p>'
                f'<div style="display:flex;gap:20px;font-size:12px;">'
                f'<span><b style="color:#1E1E1E;">{svc.get("price_range","—")}</b>'
                f' <span style="color:#7b7b78;">стоимость</span></span>'
                f'<span><b style="color:#1E1E1E;">{svc.get("duration","—")}</b>'
                f' <span style="color:#7b7b78;">срок</span></span>'
                f'<span><b style="color:#FF5600;">{svc.get("roi_estimate","—")}</b>'
                f' <span style="color:#7b7b78;">ROI</span></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        if checked:
            selected.append(svc["id"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Диагностика", use_container_width=True):
            _go("diagnostics")
    with col2:
        if st.button(
            f"Сформировать КП · {len(selected)} услуги →",
            use_container_width=True, type="primary",
            disabled=len(selected) == 0,
        ):
            st.session_state.selected_ids = selected
            _go("generating")


def page_generating():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("generating")

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:4px;">Формирую КП</h2>'
        '<p style="font-size:13px;color:#7b7b78;margin-bottom:1.5rem;">До 2 минут</p>',
        unsafe_allow_html=True,
    )

    catalog           = load_catalog()
    selected_services = get_services_by_ids(catalog, st.session_state.selected_ids)

    with st.spinner("AI составляет коммерческое предложение с графиками и ROI-обоснованием..."):
        client      = _get_client()
        proposal_md = generate_proposal_text(
            st.session_state.business_profile,
            st.session_state.assessment,
            selected_services,
            client,
        )
        st.session_state.proposal_markdown = proposal_md
        st.session_state.proposal_html = render_html(
            proposal_md,
            st.session_state.business_profile,
            st.session_state.assessment,   # ← передаём для графиков
        )

    _go("proposal")


def page_proposal():
    st.markdown(CSS, unsafe_allow_html=True)
    _logo()
    _progress("proposal")

    if st.session_state.start_ts:
        elapsed = int(time.time() - st.session_state.start_ts)
        mins, secs = divmod(elapsed, 60)
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:8px;background:#F0FFF4;'
            f'border:1px solid #C6F6D5;border-radius:8px;padding:8px 16px;margin-bottom:1.5rem;">'
            f'<span style="color:#276749;font-size:13px;font-weight:600;">КП готово за {mins} мин {secs} сек</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1E1E1E;margin-bottom:1rem;">Коммерческое предложение</h2>',
        unsafe_allow_html=True,
    )

    # Generate PDF
    with st.spinner("Генерирую PDF..."):
        pdf_bytes = render_pdf(st.session_state.proposal_html)

    col1, col2 = st.columns(2)
    with col1:
        if pdf_bytes:
            st.download_button(
                label="↓ Скачать PDF",
                data=pdf_bytes,
                file_name="puls_kp.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.download_button(
                label="↓ Скачать HTML",
                data=st.session_state.proposal_html.encode("utf-8"),
                file_name="puls_kp.html",
                mime="text/html",
                use_container_width=True,
                type="primary",
            )
    with col2:
        st.download_button(
            label="↓ Скачать Markdown",
            data=st.session_state.proposal_markdown.encode("utf-8"),
            file_name="puls_kp.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if pdf_bytes:
        st.caption("PDF сгенерирован автоматически · содержит графики и диаграммы")
    else:
        st.caption("Откройте HTML в браузере → Ctrl+P → Сохранить как PDF")

    # HTML preview
    st.markdown("---")
    _section("ПРЕДПРОСМОТР КП")
    components.html(st.session_state.proposal_html, height=700, scrolling=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("Начать новую диагностику →", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _go("welcome")


# ─── Router ───────────────────────────────────────────────────────────────────

def main():
    _init_state()
    router = {
        "welcome":         page_welcome,
        "survey_base":     page_survey_base,
        "survey_followup": page_survey_followup,
        "document":        page_document,
        "analyzing":       page_analyzing,
        "diagnostics":     page_diagnostics,
        "catalog":         page_catalog,
        "generating":      page_generating,
        "proposal":        page_proposal,
    }
    router.get(st.session_state.step, page_welcome)()


if __name__ == "__main__":
    main()
