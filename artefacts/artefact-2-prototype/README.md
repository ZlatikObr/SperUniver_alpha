# Артефакт 2 — Прототип AI-сервиса

**Ответственный:** Данила Шакин  
**Срок:** Дни 2–3 интенсива.

Прототип реализован в корне репозитория, чтобы Streamlit Cloud и локальный запуск работали без дополнительных путей и копирования модулей. Эта папка хранит описание артефакта, тесты и будущие E2E-фикстуры.

## Текущая структура

```
repo/
├── pyproject.toml          ← Python 3.10+, метаданные проекта, pytest-настройки
├── requirements.txt        ← зависимости для быстрого запуска
├── app.py                  ← Streamlit UI: опрос → документ → аудит → витрина → КП
├── backend/
│   ├── survey.py           ← адаптивный опрос, профиль бизнеса
│   ├── document_parser.py  ← PDF / CSV / Excel → метрики
│   ├── auditor.py          ← AI-анализ профиля, open-data research, оценка по зонам
│   ├── catalog.py          ← фильтрация витрины услуг
│   ├── proposal_gen.py     ← генерация КП, HTML/PDF
│   ├── _config.py          ← единая конфигурация модели
│   ├── _utils.py           ← YAML/JSON helpers с кэшем
│   └── agent_logger.py     ← JSONL-лог шагов агента
├── prompts/                ← YAML-промпты
├── data/
│   └── services_catalog.json
├── logs/                   ← runtime JSONL-логи, `agent.jsonl` игнорируется git
└── artefacts/artefact-2-prototype/
    ├── README.md
    ├── data/test_cases/    ← место для ручных E2E-кейсов и файловых фикстур
    └── tests/              ← offline pytest-набор из PR #2
```

## Быстрый старт

```bash
pip install -r requirements.txt
streamlit run app.py
```

Требуется **Python 3.10+**. Для работы LLM нужен ключ `AITUNNEL_API_KEY`; тесты запускаются офлайн и ключа не требуют.

## Чек-лист готовности

- [x] `streamlit run app.py` — запускается из корня репозитория
- [x] E2E сценарий: опрос → загрузка документа → аудит → витрина → корзина → КП
- [x] Fallback без документа: open-data research обогащает профиль
- [x] JSONL-лог шагов агента через `backend/agent_logger.py`
- [x] Промпты в YAML, не хардкод в коде
- [x] 5+ тестов: добавлен offline pytest-набор в `tests/`
- [x] README: запуск ≤2 команды
