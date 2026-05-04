# База рекомендаций AI-внедрений для демо

Файл базы: `data/ai_recommendation_base.json`.

Цель базы - дать демо-версии "Ревизора" готовые рекомендации без LLM-запроса на каждом прохождении анкеты. Модель можно оставить только для будущего режима "расширенного анализа"; в демо достаточно посчитать риск-профиль и выбрать карточки из JSON.

## Как использовать без токенов

1. После опроса сформировать профиль клиента:
   - отрасль;
   - размер компании;
   - риск-оценки по зонам `finance`, `operations`, `marketing_sales`, `team`, `strategy`;
   - ключевые боли из ответов пользователя;
   - наличие загруженных финансовых/маркетинговых документов;
   - срочность.

2. Пройти по `recommendations` и посчитать score:
   - `+3`, если `primary_zone` имеет риск `>= 4`;
   - `+2`, если одна из `secondary_zones` имеет риск `>= 3`;
   - `+2` за каждое совпадение боли с `trigger_keywords`;
   - `+1`, если отрасль клиента подходит под `target_industries` или `industry_fit = universal`;
   - `+1`, если срочность высокая и `time_to_first_value_weeks <= 4`.

3. Вернуть top-3 рекомендации:
   - не больше одной рекомендации на одну основную зону, если есть альтернативы;
   - для каждой рекомендации подтянуть `evidence_case_ids`;
   - вставить данные в шаблоны из `report_block_templates`.

## Что лежит в карточке рекомендации

Каждая карточка содержит:

- `title` - название рекомендации для отчета;
- `trigger_keywords` - боли клиента, которые активируют карточку;
- `recommended_service` - что продавать как консалтинговую услугу;
- `implementation_pattern` - какой AI-паттерн предлагается;
- `demo_template` - готовый текст для отчета;
- `expected_effect` - ожидаемый измеримый эффект;
- `data_prerequisites` - какие данные нужны;
- `pilot_steps` - план пилота на 2-4 недели;
- `kpis` - метрики для проверки результата;
- `evidence_case_ids` - ссылки на кейсы крупных компаний.

## Кейсы-доказательства

В `evidence_cases` собраны крупные компании с публичными количественными эффектами:

| Компания | AI-кейс | Эффект | Для какой рекомендации |
|---|---|---|---|
| Klarna | AI-ассистент поддержки | 2/3 чатов, эквивалент 700 агентов, 11 мин -> <2 мин | клиентский сервис, квалификация лидов |
| Bank of America | Erica | 2+ млрд взаимодействий, 42+ млн клиентов | self-service ассистент |
| JPMorgan Chase | COIN | до 360 000 часов ручной юридической работы в год | документный AI |
| UPS | ORION | около 100 млн миль и 10 млн галлонов топлива в год | маршруты и операционная оптимизация |
| Walmart | GenAI catalog data | 850+ млн единиц каталожных данных, почти 100x ускорение | каталог и качество данных |
| Moderna | ChatGPT Enterprise | 750+ custom GPTs | внутренние AI-инструменты |
| Unilever | AI hiring | около 100 000 часов отбора/интервью | HR и онбординг |
| Danone | ML demand forecasting | около 20% снижения ошибки прогноза | спрос, запасы, финпланирование |

## Минимальный код выбора

```python
def recommendation_score(profile, rec):
    score = 0
    risks = profile["risk_scores"]
    pains = " ".join(profile.get("pain_keywords", [])).lower()

    if risks.get(rec["primary_zone"], 0) >= 4:
        score += 3

    for zone in rec.get("secondary_zones", []):
        if risks.get(zone, 0) >= 3:
            score += 2
            break

    for keyword in rec.get("trigger_keywords", []):
        if keyword.lower() in pains:
            score += 2

    industry = profile.get("industry", "").lower()
    if rec.get("industry_fit") == "universal" or industry in rec.get("target_industries", []):
        score += 1

    if profile.get("urgency") == "high" and rec.get("time_to_first_value_weeks", 99) <= 4:
        score += 1

    return score
```

## Рекомендация по демо-архитектуре

В демо лучше разделить два режима:

- `demo_static`: рекомендации только из `ai_recommendation_base.json`, без LLM;
- `expert_ai`: будущий режим, где LLM может уточнять формулировки, считать business case и адаптировать КП под конкретного клиента.

Так прототип будет быстрым, дешевым и воспроизводимым на защите: один и тот же профиль всегда дает один и тот же набор рекомендаций.
