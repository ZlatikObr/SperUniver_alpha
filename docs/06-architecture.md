# Техническая архитектура — финальное ТЗ

> Цель: 38–40 баллов из 40 на питче МФТИ. Демо стоит ~0 ₽. Прототип запускается одной командой. Все артефакты сквозно связаны.

---

## 1. Архитектурные принципы (нерушимые)

1. **LLM используется только там, где он нужен.** Pandas-агрегации, статистика, аномалии — без LLM. Это снижает cost-per-task в 5–10×, ускоряет демо и упрощает тестирование.
2. **Каждый модуль = одна функция = один контракт (Pydantic).** Никакого LangChain, никакого AutoGen, никакого «магического» оркестратора.
3. **Промпты — версионированные YAML-файлы.** Никакого хардкода в коде.
4. **Каждый запуск пишет JSONL-лог.** Эти логи — источник правды для Артефакта 3 (экономика).
5. **Ничего, что не прибивает гвоздями к 4 критериям МФТИ, в MVP не идёт.**

---

## 2. Стек (фиксируем — без альтернатив)

| Слой | Технология | Почему именно это |
|---|---|---|
| Frontend | **Streamlit** 1.40+ | За 6 часов даёт дашборд того же качества, что Next.js за 2 дня. Бесплатный hosting в Streamlit Community Cloud. |
| Язык | Python 3.11 | Алсу и Данила в нём сильнее всего. |
| LLM-основа | **Claude Sonnet 4.6** | Качество для основной цепочки. |
| LLM-валидатор | **Claude Haiku 4.5** | Парсинг и санитайзинг CSV — экономия в 10× на токенах. |
| Анализ данных | pandas 2.x + numpy | Без альтернатив. |
| RAG | keyword search по 10 кейсам в `cases_db.json` | Embedding-RAG для 10 документов — оверкилл. Все 10 кейсов влезают в один промпт (~5к токенов). |
| PDF | **WeasyPrint** | Open-source, Python, без внешних API, бесплатно. |
| Тесты | pytest 8.x | Стандарт. |
| Хостинг демо | **Streamlit Community Cloud** (бесплатно) + резерв: локально + ngrok | План B обязателен. |
| API-обёртка | официальный `anthropic` Python SDK | Без надстроек. |
| Конфиг-форматы | YAML (промпты, бенчмарки), JSON (кейсы), CSV (данные) | Читаемо в репо, легко версионировать. |

**Что мы НЕ используем (и почему):**
- ❌ LangChain / LlamaIndex — медленный debug, лишние токены, чёрный ящик.
- ❌ Embedding-RAG (Chroma/Pinecone/etc.) — для 10 кейсов это пушка по воробьям.
- ❌ Next.js / React — лишний день работы без выигрыша в баллах.
- ❌ Docker — запуск через `make demo`, контейнер не нужен.
- ❌ База данных — никакого state, всё в файлах.
- ❌ Платный hosting — Streamlit Cloud + GitHub бесплатны.

---

## 3. Архитектурная диаграмма (data flow)

```
                        ┌──────────────────────┐
                        │       USER           │
                        │  (директор сети)     │
                        └─────────┬────────────┘
                                  │ form + CSV upload
                                  ▼
                        ┌──────────────────────┐
                        │   Streamlit UI       │
                        │      (app.py)        │
                        └─────────┬────────────┘
                                  │ raw_input
                                  ▼
                        ┌──────────────────────┐
                        │   Validator          │  ← Haiku
                        │ (src/validator.py)   │     (только если CSV сломан)
                        └─────────┬────────────┘
                                  │ ValidatedDataset
                                  ▼
                        ┌──────────────────────┐
                        │   Analyzer           │  ← pure pandas
                        │ (src/analyzer.py)    │     (без LLM)
                        └─────────┬────────────┘
                                  │ AnalysisResult
                                  ▼
                        ┌──────────────────────┐
                        │   CaseMatcher        │  ← Sonnet
                        │ (src/case_matcher.py)│     + cases_db.json
                        └─────────┬────────────┘
                                  │ MatchedCases (топ-3)
                                  ▼
                        ┌──────────────────────┐
                        │ InitiativeGenerator  │  ← Sonnet
                        │ (src/initiative_gen) │     + benchmarks.yaml
                        └─────────┬────────────┘
                                  │ Initiatives (3 шт)
                                  ▼
                        ┌──────────────────────┐
                        │   ROICalculator      │  ← pure pandas + формулы
                        │ (src/roi_calc.py)    │     (без LLM)
                        └─────────┬────────────┘
                                  │ ROIReport
                                  ▼
                        ┌──────────────────────┐
                        │   Reporter           │  ← Sonnet
                        │ (src/reporter.py)    │     (markdown)
                        └─────────┬────────────┘
                                  │ MarkdownReport
                                  ▼
                        ┌──────────────────────┐
                        │   PDFRenderer        │  ← WeasyPrint
                        │ (src/pdf_renderer)   │     (без LLM)
                        └─────────┬────────────┘
                                  │ PDF + dashboard
                                  ▼
                        ┌──────────────────────┐
                        │       USER           │
                        │ видит дашборд + PDF  │
                        └──────────────────────┘

      ─ ─ ─ Каждый шаг → JSONL-лог в logs/{run_id}.jsonl ─ ─ ─
```

**Сквозной контракт:** на входе и выходе каждого модуля — Pydantic-модель. Если контракт нарушен, пайплайн падает с понятной ошибкой, а не «галлюцинирует».

---

## 4. Контракты модулей (Pydantic-схемы)

```python
# src/schemas.py

from pydantic import BaseModel, Field
from datetime import date
from enum import Enum

class WriteoffReason(str, Enum):
    SPOILAGE = "порча"
    EXPIRED = "истёк_срок"
    DAMAGED = "повреждение_упаковки"
    MISGRADING = "пересортица"
    OTHER = "иное"

class WriteoffRow(BaseModel):
    date: date
    store_id: str
    sku: str
    category: str
    reason: WriteoffReason
    quantity: float
    cost_per_unit: float
    total_loss: float

class NetworkProfile(BaseModel):
    name: str
    format: str  # «у дома», «мини-маркет», ...
    store_count: int = Field(ge=1, le=200)
    annual_revenue_rub: float
    avg_writeoff_pct_self_reported: float | None = None

class ValidatedDataset(BaseModel):
    profile: NetworkProfile
    rows: list[WriteoffRow]
    period_months: int
    total_loss_rub: float
    quality_score: float  # 0..1, насколько чистые данные

class AnalysisResult(BaseModel):
    top_categories: list[dict]  # [{category, loss_rub, share_pct}, ...]
    top_stores: list[dict]
    seasonality: dict  # {month: loss_rub}
    anomalies: list[str]  # human-readable инсайты
    overall_loss_pct_revenue: float

class MatchedCase(BaseModel):
    case_id: str
    company: str
    relevance_score: float  # 0..1
    why_relevant: str

class Initiative(BaseModel):
    title: str
    description: str
    technology: str  # «динамическое ценообразование», «CV-детекция свежести», ...
    based_on_cases: list[str]
    investment_rub: float
    payback_months: int
    expected_loss_reduction_pct: float
    confidence: float  # 0..1

class ROIReport(BaseModel):
    network_profile: NetworkProfile
    initiatives: list[Initiative]
    annual_savings_total_rub: float
    total_investment_rub: float
    blended_roi_x: float
    blended_payback_months: int
    assumptions: list[str]

class MarkdownReport(BaseModel):
    content: str  # markdown
    metadata: dict  # generated_at, run_id, model_used
```

---

## 5. Финальная структура репо

```
SperUniver_alpha/
├── README.md                         ← обзор + статус + команда (есть)
├── pyproject.toml                    ← зависимости + entry points (новый)
├── Makefile                          ← make demo / test / lint (новый)
├── .gitignore                        ← (есть)
├── .env.example                      ← ANTHROPIC_API_KEY=... (новый)
├── app.py                            ← Streamlit entry point (новый)
├── src/
│   ├── __init__.py
│   ├── schemas.py                    ← Pydantic модели всех контрактов
│   ├── validator.py                  ← парсинг + Haiku-санитайзер CSV
│   ├── analyzer.py                   ← pandas: статистика, аномалии
│   ├── case_matcher.py               ← поиск по cases_db.json + Sonnet
│   ├── initiative_generator.py       ← выбор и адаптация инициатив
│   ├── roi_calculator.py             ← формулы ROI (без LLM)
│   ├── reporter.py                   ← markdown отчёт через Sonnet
│   ├── pdf_renderer.py               ← weasyprint
│   ├── orchestrator.py               ← главный пайплайн
│   ├── llm_client.py                 ← обёртка над Anthropic SDK + retry
│   └── logger.py                     ← JSONL-логирование
├── prompts/
│   ├── validator.yaml
│   ├── case_matcher.yaml
│   ├── initiative_generator.yaml
│   └── reporter.yaml
├── configs/
│   ├── benchmarks.yaml               ← общие бенчмарки рынка
│   └── settings.yaml                 ← модели, токены, limits
├── data/
│   ├── cases_db.json                 ← 10 кейсов
│   ├── synthetic_writeoffs.csv       ← основной демо-датасет
│   └── edge_cases/
│       ├── small.csv
│       ├── dirty.csv
│       ├── empty.csv
│       └── wrong_columns.csv
├── tests/
│   ├── test_validator.py
│   ├── test_analyzer.py
│   ├── test_orchestrator.py
│   └── fixtures/
├── docs/                             ← теория (есть)
│   ├── 00-brief.md
│   ├── 01-target-audience.md
│   ├── 02-market-research.md
│   ├── 03-mvp-scope.md
│   ├── 04-criteria-mfti.md
│   ├── 05-data-strategy.md
│   └── 06-architecture.md            ← этот файл
├── artefacts/
│   ├── artefact-1-spec/
│   │   ├── spec.md                   ← полная спецификация
│   │   ├── user_stories.md
│   │   ├── ai_policy.md              ← обязательный раздел!
│   │   └── architecture.png          ← диаграмма (от Маши)
│   ├── artefact-2-prototype/
│   │   └── README.md
│   └── artefact-3-economics/
│       ├── economics.md              ← полный расчёт
│       ├── cost_per_task.csv         ← из реальных логов
│       └── scenarios.md              ← base/optimistic
├── pitch/
│   └── pitch.md                      ← скрипт питча
└── logs/                             ← в .gitignore
```

---

## 6. Приоритизация фич (MoSCoW)

### Must — обязательно работает на демо

| ID | Фича | Кто | День |
|---|---|---|---|
| M1 | Загрузка CSV/Excel + базовая валидация | Данила | 2 |
| M2 | Анализ через pandas (топ-категории, точки, аномалии) | Данила | 2 |
| M3 | AI-выбор 3 инициатив на основе БД кейсов | Данила | 2 |
| M4 | ROI-калькулятор с числами из данных клиента | Данила | 3 |
| M5 | Markdown-отчёт + PDF-экспорт | Данила | 3 |
| M6 | JSONL-логирование шагов | Данила | 3 |
| M7 | README с одной командой запуска (`make demo`) | Данила | 3 |
| M8 | Spec / User Stories / AI Policy / DoD | Маша | 1 |
| M9 | cases_db.json с 10 кейсами | Алсу | 1 |
| M10 | synthetic_writeoffs.csv | Алсу | 1 |
| M11 | Артефакт 3 с реальными числами из логов | Злата | 3 |
| M12 | 5+ тест-сценариев (включая negative) | Даня | 2–3 |
| M13 | Скрипт питча по PDLC-логике | Злата | 3 |

### Should — даёт «вау» в демо

| ID | Фича | Кто | День |
|---|---|---|---|
| S1 | Streamlit прогресс-бар по шагам | Данила | 3 |
| S2 | 2–3 dashboard-визуализации (plotly или streamlit charts) | Данила | 3 |
| S3 | Демо-режим «нажать кнопку → готовый прогон за 30 сек» | Данила | 3 |
| S4 | Сравнительная таблица «без AI vs c AI» в PDF | Данила | 3 |

### Could — если останется час

| ID | Фича |
|---|---|
| C1 | Конструктор сценариев («что если уменьшим на 5%?») |
| C2 | Sharing — выгрузка отчёта в email-friendly markdown |

### Won't — явно вне MVP

| ID | Что |
|---|---|
| W1 | Реальная интеграция с 1С API |
| W2 | Авторизация / личный кабинет |
| W3 | Биллинг / оплата |
| W4 | CV-модели для распознавания свежести |
| W5 | Real-time мониторинг |

---

## 7. План реализации по дням

### День 1 (Спецификация + данные + каркас)

| Кто | Задачи |
|---|---|
| **Маша** | Артефакт 1: User Stories, AC, AI Policy, DoD, диаграмма архитектуры |
| **Алсу** | `cases_db.json` (10 кейсов), `synthetic_writeoffs.csv` (10к строк) |
| **Данила** | Каркас репо: `pyproject.toml`, `Makefile`, `schemas.py`, `llm_client.py`, `logger.py`, `.env.example` |
| **Злата** | Skeleton питча, синхронизация ТЗ с командой |
| **Даня** | Подготовить 4 edge-case датасета |

### День 2 (Прототип — backend)

| Кто | Задачи |
|---|---|
| **Данила** | `validator.py` → `analyzer.py` → `case_matcher.py` → `initiative_generator.py` → `roi_calculator.py` |
| **Маша** | Финализирует Артефакт 1, пишет диаграмму |
| **Алсу** | Проверяет реалистичность кейсов, валидирует выводы агента |
| **Злата** | Текстовая основа Артефакта 3, `pitch.md` черновик |
| **Даня** | `tests/` — пишет тесты под уже готовые модули |

### День 3 (UI + экономика + питч)

| Утро | День | Вечер |
|---|---|---|
| Данила: `reporter.py` + `pdf_renderer.py` + `app.py` (Streamlit) | Все: интеграционные тесты, прогон 5 сценариев | Злата: Артефакт 3 с реальными числами из `logs/` |
| Маша: финал спеки | Алсу: финальная проверка кейсов и числа | Все: репетиция питча с таймером |

---

## 8. Cost-per-task (предварительный расчёт)

| Шаг | Модель | Input токены | Output токены | Стоимость (USD) | Стоимость (₽) |
|---|---|---|---|---|---|
| Validator (только если CSV грязный) | Haiku 4.5 | 2 000 | 500 | $0.001 | 0.1 ₽ |
| CaseMatcher | Sonnet 4.6 | 8 000 | 2 000 | $0.054 | ~5 ₽ |
| InitiativeGenerator | Sonnet 4.6 | 6 000 | 3 000 | $0.063 | ~6 ₽ |
| Reporter | Sonnet 4.6 | 5 000 | 4 000 | $0.075 | ~7 ₽ |
| **Итого / 1 аудит** | | **~21k input** | **~9.5k output** | **~$0.20** | **~18 ₽** |

> Тарифы Sonnet 4.6 ориентировочные на момент составления — подтвердить актуальные на день питча.

**Для всего 3-дневного интенсива:**
- 30 прогонов на разработке + 20 прогонов на тестах + 10 прогонов на питче = ~60 прогонов
- 60 × $0.20 = **$12 на API за всё мероприятие**
- Streamlit Cloud — 0 ₽
- GitHub — 0 ₽
- WeasyPrint — 0 ₽
- Итого: **демо стоит < $15 ≈ 1300 ₽** для всей команды на 3 дня

---

## 9. Сценарии экономики (для Артефакта 3)

### Base — реалистичный

- 20 платных аудитов в месяц × 39 900 ₽ = 798 000 ₽/мес
- Cost: 20 × 18 ₽ = 360 ₽/мес на API
- Маржа: ~99.95%

### Optimistic — после первых партнёрств

- 80 разовых аудитов × 39 900 ₽ = 3 192 000 ₽/мес
- 10 подписок × 19 900 ₽ = 199 000 ₽/мес
- Cost API: ~1 800 ₽/мес
- Брокеридж от вендоров: ~5% от среднего внедрения 3 млн ₽ × 5 сделок = 750 000 ₽/мес
- Итого: ~4.1 млн ₽/мес выручки при марже 99.9%

### Trade-off (для Артефакта 3)

| Параметр | Sonnet 4.6 (текущий) | Haiku-only (альтернатива) |
|---|---|---|
| Качество анализа | 9/10 | 6/10 |
| Cost-per-task | $0.20 | $0.02 |
| Latency | ~25 сек | ~8 сек |
| Решение | **Берём Sonnet** — на этой задаче падение качества с 9 до 6 = клиент не доверяет рекомендациям → потеря выручки ≫ экономии $0.18 на запросе |

---

## 10. AI Policy (выжимка — полная версия в `artefacts/artefact-1-spec/ai_policy.md`)

| Параметр | Значение |
|---|---|
| Scope | Анализ списаний продуктового ритейла, рекомендации по AI-внедрениям |
| Источники данных | Загруженный CSV + `cases_db.json` (10 публичных кейсов). НИКАКИХ внешних API на инференсе. |
| Формат вывода | Каждый шаг — JSON по Pydantic-схеме. Финал — markdown отчёт. |
| При недостатке данных | Возвращать `{"status": "insufficient_data", "missing": [...]}`. Числа НЕ выдумывать. |
| При низкой уверенности (<0.5) | Помечать «требует верификации», давать диапазон вместо точки. |
| Запрещённые выводы | Конкретные люди / сотрудники, кражи, мошенничество, юридические рекомендации, точные обещания ROI без диапазона. |
| Эскалация | Если данные < 30 дней или confidence < 0.5 → «рекомендую очный аудит». |

---

## 11. Тестовые сценарии (минимум 5 для Артефакта 2)

| # | Имя | Вход | Ожидаемый результат |
|---|---|---|---|
| 1 | Базовый | `synthetic_writeoffs.csv` (10к строк, 6 мес) | Полный отчёт с 3 инициативами + ROI |
| 2 | Грязные данные | `dirty.csv` (пропуски, разные форматы дат) | Validator чистит → пайплайн проходит |
| 3 | Маленький датасет | `small.csv` (1 мес, 1 точка) | Эскалация: «недостаточный период для трендов» |
| 4 | Пустой | `empty.csv` (нулевые списания) | `{"status": "insufficient_data"}` без галлюцинаций |
| 5 | Негативный | `wrong_columns.csv` (не те колонки) | Понятная ошибка валидации, не падение |

---

## 12. План B (если что-то сломается на демо)

| Сценарий факапа | План B |
|---|---|
| Streamlit Cloud упал | Локальный запуск + ngrok-тоннель в проектор |
| Anthropic API rate limit | На демо используем кеш `cache.json` с предзаписанными ответами на демо-датасете |
| Интернета нет | Pre-recorded видео-демо (записать вечером Дня 2) |
| WeasyPrint глючит на macOS | Fallback: markdown → HTML, открываем в браузере и печатаем «Сохранить как PDF» |

---

## 13. Сквозная связь артефактов (для критерия 4)

```
Артефакт 1 (User Stories)
   │
   │ User Story «диагностика списаний» 
   │ → требует: парсинг CSV + analyzer + case_matcher + reporter
   │
   ▼
Артефакт 2 (Прототип)
   │
   │ Реализует те самые модули, прогоняет E2E
   │ → пишет JSONL-логи: токены, latency, costs
   │
   ▼
Артефакт 3 (Экономика)
   │
   │ Берёт реальные числа из логов Артефакта 2
   │ → cost-per-task ≈ $0.20 (НЕ выдуманное число — из логов)
   │ → base/optimistic сценарии исходят из этих чисел
   │
   ▼
Питч (PDLC-логика)
   │
   │ Проблема (из Артефакта 1) → требования → архитектура → реализация (Артефакт 2)
   │ → тесты (5 сценариев) → экономика (Артефакт 3)
   │ → следующие шаги (интеграция с 1С API, пилот с одной сетью)
```

**Это даёт +3 балла за «все три артефакта связаны» и +2 за «PDLC-логика питча» = 5 из 10 баллов критерия 4 уже заложены архитектурно.**

---

## 14. Готовность к Q&A (что жюри спросит)

| Вопрос | Кто отвечает | Короткий ответ |
|---|---|---|
| «Какие данные использовали для тестов?» | Алсу | Синтетика 10к строк + Kaggle + 5 edge cases |
| «Как работает RAG?» | Данила | 10 кейсов в JSON, keyword-матчинг → топ-3 → инжектим в Sonnet |
| «Откуда числа в экономике?» | Злата | Из JSONL-логов реальных прогонов прототипа |
| «Что если данные клиента не подходят под наши кейсы?» | Маша | AI Policy: эскалация на «требует очного аудита» |
| «Почему Streamlit, а не настоящий продукт?» | Злата | За 3 дня собрать сильный MVP важнее, чем красивый UI. В roadmap — Next.js + интеграция с 1С |
| «Сколько стоит внедрение для среднего ритейлера?» | Алсу | 39 900 ₽ за разовый аудит + опционально подписка 19 900 ₽/мес |
| «Какая ваша уникальность против Strategy Partners?» | Злата | Цена в 30–40× ниже + независимость от вендоров AI-решений |

---

## 15. Чек-лист готовности к питчу

- [ ] `make demo` запускает работающий пайплайн на любой машине после `git clone`
- [ ] E2E сценарий проходит на synthetic_writeoffs.csv без ручных правок
- [ ] PDF генерируется и открывается
- [ ] В `logs/` есть минимум 5 успешных прогонов
- [ ] Числа в `artefacts/artefact-3-economics/economics.md` совпадают с логами
- [ ] AI Policy — отдельный файл, не короче 1 страницы
- [ ] User Stories написаны в формате «Как [роль], я хочу [действие], чтобы [ценность]»
- [ ] У каждого участника команды есть свой блок, который он может объяснить
- [ ] Раздел «следующие шаги» вытекает из текущего состояния, не fantasy
- [ ] Хотя бы одна полная репетиция питча с таймером
