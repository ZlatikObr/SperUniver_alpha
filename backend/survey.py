from openai import OpenAI

from ._config import MODEL
from ._utils import (
    extract_json_array as _extract_json_array,
    extract_json_object as _extract_json_object,
    load_prompt as _load_prompt,
)
from .agent_logger import log_agent_step

BASE_QUESTIONS = [
    {
        "id": "industry",
        "text": "В какой отрасли работает ваш бизнес?",
        "type": "select",
        "options": [
            "Розничная торговля", "Производство", "IT и технологии",
            "Строительство", "Логистика и транспорт", "Финансы и страхование",
            "Образование", "Здравоохранение", "Общественное питание",
            "Профессиональные услуги (юристы, бухгалтеры, консультанты)", "Другое"
        ]
    },
    {
        "id": "region",
        "text": "В каком регионе преимущественно работает бизнес?",
        "type": "text",
        "placeholder": "Например: Москва, Санкт-Петербург, Екатеринбург"
    },
    {
        "id": "employees",
        "text": "Сколько человек работает в компании?",
        "type": "select",
        "options": ["1–10", "11–50", "51–200", "201–500", "500+"]
    },
    {
        "id": "revenue",
        "text": "Годовая выручка компании (ориентировочно)?",
        "type": "select",
        "options": [
            "До 30 млн ₽", "30–100 млн ₽", "100–500 млн ₽",
            "500 млн – 2 млрд ₽", "Более 2 млрд ₽"
        ]
    },
    {
        "id": "main_challenge",
        "text": "Опишите главную проблему или задачу. Что вас беспокоит больше всего?",
        "type": "textarea",
        "placeholder": "Например: падают продажи, растут издержки, сложно масштабироваться..."
    },
    {
        "id": "company_age",
        "text": "Сколько лет существует компания?",
        "type": "select",
        "options": ["Менее 1 года", "1–3 года", "3–7 лет", "7–15 лет", "Более 15 лет"]
    },
    {
        "id": "profitability",
        "text": "Как вы оцениваете текущую прибыльность бизнеса?",
        "type": "select",
        "options": [
            "Работаем в убыток", "Выходим в ноль", "Небольшая прибыль (до 5%)",
            "Хорошая прибыль (5–15%)", "Высокая прибыль (>15%)"
        ]
    },
    {
        "id": "client_satisfaction",
        "text": "Как вы оцениваете уровень удовлетворённости ваших клиентов?",
        "type": "select",
        "options": [
            "Много жалоб, высокий отток", "Есть проблемы с удержанием",
            "Средний уровень, стабильно", "Клиенты довольны, редкие жалобы",
            "Высокая лояльность, активные рекомендации"
        ]
    },
    {
        "id": "team_challenge",
        "text": "Какие основные сложности с командой вы испытываете?",
        "type": "select",
        "options": [
            "Сложно найти нужных специалистов", "Высокая текучесть кадров",
            "Низкая производительность", "Конфликты и слабое взаимодействие",
            "Отсутствие чёткой системы управления", "С командой всё в порядке"
        ]
    },
    {
        "id": "process_maturity",
        "text": "Насколько формализованы бизнес-процессы в компании?",
        "type": "select",
        "options": [
            "Всё держится в головах сотрудников",
            "Есть отдельные инструкции, но не систематизированы",
            "Процессы описаны, но не всегда соблюдаются",
            "Чёткие регламенты, регулярный контроль",
            "Автоматизированные и оцифрованные процессы"
        ]
    }
]


def get_base_questions() -> list[dict]:
    return BASE_QUESTIONS


def generate_followup_questions(answers: dict, client: OpenAI) -> list[dict]:
    log_agent_step("survey.generate_followup_questions", "start", answer_count=len(answers))
    prompt = _load_prompt("survey_v1")
    answers_text = "\n".join(f"— {k}: {v}" for k, v in answers.items())

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=700,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {
                    "role": "user",
                    "content": (
                        "Ответы клиента на базовые вопросы:\n"
                        f"{answers_text}\n\n"
                        "Сгенерируй ровно 3 уточняющих вопроса в формате JSON-массива."
                    )
                }
            ]
        )

        raw = response.choices[0].message.content or ""
        result = _extract_json_array(raw)
        if result:
            log_agent_step("survey.generate_followup_questions", "success", question_count=len(result))
            return result
    except Exception as exc:
        log_agent_step("survey.generate_followup_questions", "error", error=exc)

    log_agent_step("survey.generate_followup_questions", "fallback")
    return [
        {"id": "followup_1", "text": "Какой результат вы ожидаете получить от работы с консультантом?", "type": "text"},
        {"id": "followup_2", "text": "Какие попытки решить проблему вы уже предпринимали?", "type": "text"},
        {"id": "followup_3", "text": "В какие сроки вы планируете решить текущую проблему?", "type": "select",
         "options": ["1–3 месяца", "3–6 месяцев", "6–12 месяцев", "Более года"]},
    ]


def build_business_profile(base_answers: dict, followup_answers: dict, client: OpenAI) -> dict:
    log_agent_step(
        "survey.build_business_profile",
        "start",
        base_answer_count=len(base_answers),
        followup_answer_count=len(followup_answers),
    )
    all_answers = {**base_answers, **followup_answers}
    answers_text = "\n".join(f"— {k}: {v}" for k, v in all_answers.items())

    prompt = _load_prompt("survey_v1")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt["profile_system"]},
                {
                    "role": "user",
                    "content": f"Создай структурированный JSON-профиль бизнеса на основе ответов:\n\n{answers_text}"
                }
            ]
        )

        raw = response.choices[0].message.content or ""
        result = _extract_json_object(raw)
        if result:
            log_agent_step("survey.build_business_profile", "success", fields=list(result.keys()))
            return result
    except Exception as exc:
        log_agent_step("survey.build_business_profile", "error", error=exc)

    log_agent_step("survey.build_business_profile", "fallback")
    return {
        "industry": base_answers.get("industry", "Не указано"),
        "region": base_answers.get("region", "Не указано"),
        "size": base_answers.get("employees", "Не указано"),
        "revenue_range": base_answers.get("revenue", "Не указано"),
        "company_age": base_answers.get("company_age", "Не указано"),
        "profitability": base_answers.get("profitability", "Не указано"),
        "main_challenge": base_answers.get("main_challenge", ""),
        "client_satisfaction": base_answers.get("client_satisfaction", ""),
        "team_challenge": base_answers.get("team_challenge", ""),
        "process_maturity": base_answers.get("process_maturity", ""),
        "additional_context": str(followup_answers),
    }
