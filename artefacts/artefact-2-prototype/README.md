# Артефакт 2 — Прototип AI-сервиса

**Ответственный:** Данила Шакин  
**Срок:** Дни 2–3 интенсива.

## Будущая структура

```
artefact-2-prototype/
├── README.md               ← как запустить (≤2 команды)
├── pyproject.toml
├── app.py                  ← Streamlit UI (опрос → витрина → КП)
├── agent/
│   ├── orchestrator.py     ← агент-оркестратор
│   ├── survey.py           ← адаптивный опрос, профиль бизнеса
│   ├── document_parser.py  ← PDF / CSV / Excel → метрики
│   ├── auditor.py          ← AI-анализ профиля, оценка по зонам
│   ├── catalog.py          ← фильтрация витрины услуг
│   └── proposal_gen.py     ← генерация КП (PDF)
├── prompts/                ← YAML-промпты (версионированы)
│   ├── survey_v1.yaml
│   ├── audit_v1.yaml
│   └── proposal_v1.yaml
├── data/
│   ├── services_catalog.json
│   └── test_cases/
├── logs/                   ← JSONL-логи шагов агента
└── tests/                  ← 5+ тест-сценариев
```

## Чек-лист готовности

- [ ] `streamlit run app.py` — запускается без ручной магии
- [ ] E2E сценарий: опрос → загрузка документа → аудит → витрина → корзина → КП
- [ ] Fallback без документа: web_search tool обогащает профиль
- [ ] JSONL-лог каждого шага агента
- [ ] Промпты в YAML, не хардкод в коде
- [ ] 5+ тестов (happy path x2, edge case x2, негатив x1)
- [ ] README: запуск ≤2 команды
