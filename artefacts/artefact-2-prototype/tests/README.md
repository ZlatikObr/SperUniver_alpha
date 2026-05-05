# Тесты прототипа «Пульс»

Покрытие: **33 теста** на 5 backend-модулей: `survey`, `document_parser`, `catalog`, `auditor`, `proposal_gen`.
Все тесты офлайн: OpenAI-клиент и сетевой research мокируются через `unittest.mock` и `monkeypatch`.

## Запуск

```bash
pip install -r requirements.txt pytest
pytest artefacts/artefact-2-prototype/tests/ -v
```

Требуется **Python 3.10+**. API-ключ `AITUNNEL_API_KEY` для тестов не нужен.

## Состав

Фактически реализовано: **33 теста** = happy path, edge cases, негативные сценарии и защитные контрактные проверки.

| Файл | Кол-во | Что покрыто |
|---|---:|---|
| `test_survey.py` | 5 | Базовые вопросы, извлечение JSON, fallback профиля и уточняющих вопросов |
| `test_document_parser.py` | 8 | CSV/PDF/Excel-парсинг, cp1251, пустые таблицы, битые и неподдерживаемые файлы |
| `test_catalog.py` | 6 | Валидность каталога, фильтрация, padding, выбор по id |
| `test_proposal_gen.py` | 8 | Полный список услуг в КП, fallback без LLM, HTML, markdown, PDF-fallback |
| `test_auditor.py` | 6 | 5 зон диагностики, partial-режим, nested JSON, hh.ru/network fallback, полный fallback аудита |

Минимальное требование артефакта «5+ тестов (happy×2 / edge×2 / негатив×1)» перевыполнено.
