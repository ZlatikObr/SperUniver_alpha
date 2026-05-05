# Тесты прототипа «Пульс»

Покрытие: **33 теста** на 5 backend-модулей (`survey`, `document_parser`, `catalog`, `auditor`, `proposal_gen`).
Все тесты **офлайн** — не требуют API-ключа, не делают сетевых вызовов (OpenAI и сетевой ресёрч мокируются).

## Запуск

Из корня репозитория:

```bash
pip install -r requirements.txt pytest
pytest artefacts/artefact-2-prototype/tests/ -v
```

> ⚠️ Требуется **Python 3.10+** — production-код использует синтаксис PEP 604 (`dict | None`).
> `conftest.py` сам добавляет корень репо в `sys.path`, чтобы был импортируем `backend`.

## Категории (по требованию из README артефакта)

Требование артефакта: ≥5 тестов = happy×2 + edge×2 + негатив×1.
Фактически реализовано: **33 теста** = happy×9 + edge×11 + негатив×8 (+ 5 защитно-вспомогательных).

### `test_survey.py` — адаптивный опрос и парсинг JSON

| # | Тест | Категория | Что проверяет / зачем |
|---|------|-----------|------------------------|
| 1 | `test_get_base_questions_returns_required_fields` | happy | У всех 10+ базовых вопросов есть `id`, `text`, `type`; нет дубликатов id; `select`-вопросы имеют непустой `options`. Гарантирует, что UI не упадёт на отрисовке формы. |
| 2 | `test_extract_json_object_with_surrounding_noise` | happy | Парсер вытаскивает JSON, даже если LLM добавил поясняющий текст и markdown-обёртку — типичная реальность для `gpt-4o-mini`. |
| 3 | `test_extract_json_array_invalid_returns_none` | edge | На битом/невалидном JSON парсер возвращает `None`, а не бросает `JSONDecodeError`. Защищает оркестрацию от падения. |
| 4 | `test_build_business_profile_fallback_when_llm_fails` | edge | Если LLM вернул мусор, профиль собирается из ответов опроса — пользователь не теряет введённые данные. |
| 5 | `test_generate_followup_uses_fallback_on_garbage_llm` | негатив | На некорректном ответе LLM используются 3 дефолтных уточняющих вопроса вместо краша опроса. |

### `test_document_parser.py` — парсинг загруженных документов

| # | Тест | Категория | Что проверяет / зачем |
|---|------|-----------|------------------------|
| 6 | `test_parse_csv_extracts_numeric_metrics` | happy | CSV → корректные `sum`/`mean`/`last` по числовым колонкам. Базовый сценарий аудита. |
| 7 | `test_extract_numeric_mentions_finds_russian_terms` | happy | Regex находит выручку/прибыль/сотрудников/клиентов в русском тексте — критично для PDF-отчётов 1С. |
| 8 | `test_parse_csv_handles_cp1251_encoding` | edge | CSV в `cp1251` (стандарт 1С/Excel-RU) читается. Без этого теряем половину российских пользователей. |
| 9 | `test_dataframe_to_result_with_empty_frame` | edge | Пустой DataFrame не валит парсер. |
| 10 | `test_extract_numeric_mentions_handles_no_matches` | edge | Текст без финансовых терминов → `{}`, не исключение. |
| 11 | `test_parse_unsupported_extension` | негатив | `.docx` → дружелюбное сообщение, флоу продолжается без документа. |
| 12 | `test_parse_corrupted_pdf_returns_error` | негатив | Битые байты в PDF → `error`, а не Streamlit-краш. |
| 13 | `test_parse_csv_undecodable_bytes` | негатив | CSV с непарсимой бинарной мусоркой не валит UI. |

### `test_catalog.py` — витрина услуг

| # | Тест | Категория | Что проверяет / зачем |
|---|------|-----------|------------------------|
| 14 | `test_load_catalog_returns_valid_services` | happy | `services_catalog.json` валиден: уникальные id, все обязательные поля. Защита от регрессии при ручных правках каталога. |
| 15 | `test_filter_services_by_finance_zone` | happy | Фильтр по зоне «финансы» возвращает релевантные услуги. |
| 16 | `test_filter_services_pads_when_zone_unknown` | edge | Padding-логика: даже при незнакомой зоне витрина не пустая (минимум 3 услуги). |
| 17 | `test_filter_services_industry_specific_priority` | edge | Услуги с `industries=["all"]` всегда в результате — иначе клиенты редких отраслей получат пустой каталог. |
| 18 | `test_get_services_by_unknown_ids_returns_empty` | негатив | Запрос несуществующих id → `[]`, не KeyError. |

### `test_proposal_gen.py` — генерация КП

| # | Тест | Категория | Что проверяет / зачем |
|---|------|-----------|------------------------|
| 19 | `test_build_services_section_includes_all_selected` | happy | **Все** выбранные услуги попадают в КП. Это фикс из коммита `78fc738` — LLM раньше обрезал список, теперь блок строится кодом. |
| 20 | `test_fallback_proposal_is_complete_without_llm` | happy | `_fallback_proposal` даёт валидный КП без обращения к LLM — страховка на случай сбоя API. |
| 21 | `test_build_services_section_empty_list` | edge | Пустой список → пустая строка, не TypeError. |
| 22 | `test_build_services_section_missing_fields_use_dashes` | edge | Услуга без `description`/`roi` рендерится с `—`, не валит сборку. |
| 23 | `test_render_html_with_minimal_inputs` | edge | HTML рендерится без `assessment` (только профиль + текст). |
| 24 | `test_simple_md_to_html_handles_nested_lists_and_bold` | edge | Самописный markdown→html парсер (фоллбек, когда `markdown` пакет недоступен) корректно обрабатывает заголовки, списки, `**bold**`. |
| 25 | `test_generate_proposal_text_falls_back_when_llm_raises` | негатив | Падение LLM → пользователь всё равно получает КП с **полным** списком выбранных услуг. |
| 26 | `test_render_pdf_returns_none_on_invalid_html` | негатив | Недоступен weasyprint → `None`, app.py переключится на скачивание HTML. |

### `test_auditor.py` — диагностика по 5 зонам

| # | Тест | Категория | Что проверяет / зачем |
|---|------|-----------|------------------------|
| 27 | `test_build_zone_queries_covers_all_five_zones` | happy | Запросы строятся по всем 5 зонам и параметризуются отраслью. Контракт ресёрч-пайплайна. |
| 28 | `test_partial_assessment_when_too_few_answers` | happy | <5 ответов → быстрый partial без вызова LLM (экономия токенов и времени). |
| 29 | `test_extract_json_object_handles_nested_objects` | edge | Парсер вытаскивает вложенные структуры (зоны → риски → массивы). |
| 30 | `test_run_phase2_micro_returns_empty_when_no_challenge` | edge | Без `main_challenge` микро-фаза не делает лишних запросов. |
| 31 | `test_fetch_hh_handles_network_failure` | edge | hh.ru недоступен → `{}`, диагностика идёт без него. |
| 32 | `test_analyze_business_returns_fallback_when_llm_returns_garbage` | негатив | Тройная страховка: web_search мокирован, LLM вернул мусор → итог = валидный `_fallback_assessment` с 5 зонами. Гарантирует, что E2E-флоу всегда доходит до шага «витрина услуг». |

> Нумерация выше сквозная — итого **32 проверочных функции** в 5 файлах. Минимальное требование «5+ тестов (happy×2 / edge×2 / негатив×1)» перевыполнено.
