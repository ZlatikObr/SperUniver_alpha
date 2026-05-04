# Данные

Синтетические данные для тестирования и витрина услуг.

## Структура

```
data/
├── services_catalog.json   ← витрина: 15–20 консалтинговых услуг (Алсу)
└── test_cases/
    ├── retail_ok.json          ← профиль: ритейл, падение маржи
    ├── services_growth.json    ← профиль: B2B-услуги, хаотичный рост
    ├── manufacturing_cash.json ← профиль: производство, кассовый разрыв
    ├── minimal_profile.json    ← edge case: минимум данных
    ├── injection_attempt.json  ← негатив: prompt injection в полях
    └── docs/
        ├── sample_pl.pdf       ← синтетический P&L (2 стр.)
        ├── sample_marketing.xlsx ← маркетинговый отчёт (CAC, LTV)
        └── wrong_format.jpg    ← тест graceful ошибки
```

## Чек-лист (кто что готовит)

- [ ] `services_catalog.json` — Алсу (3 ч)
- [ ] 5 профилей `test_cases/*.json` — Данила / Claude (1 ч)
- [ ] 3 тестовых документа — Данила (1 ч)
