"""Тесты модуля backend.catalog — загрузка и фильтрация витрины услуг."""
import pytest

from backend import catalog as cat


# ── Happy path ────────────────────────────────────────────────────────────────

def test_load_catalog_returns_valid_services():
    """Happy path: каталог загружается из data/services_catalog.json и валиден."""
    services = cat.load_catalog()

    assert isinstance(services, list)
    assert len(services) >= 5, "В каталоге должно быть минимум 5 услуг"

    required = {"id", "name", "zone", "description", "price_range", "duration", "roi_estimate"}
    seen_ids = set()
    for svc in services:
        assert required.issubset(svc.keys()), f"У услуги {svc.get('id')} нет полей: {required - svc.keys()}"
        assert svc["id"] not in seen_ids, f"Дублированный id услуги: {svc['id']}"
        seen_ids.add(svc["id"])


def test_filter_services_by_finance_zone():
    """Happy path: фильтр по зоне 'финансы' возвращает релевантные услуги."""
    catalog = cat.load_catalog()
    filtered = cat.filter_services(catalog, ["финансы"], industry="Розничная торговля")

    assert len(filtered) >= 3, "Должно вернуться минимум 3 услуги"
    # Хотя бы одна услуга должна быть из зоны 'финансы'
    assert any(s["zone"].lower() == "финансы" for s in filtered)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_filter_services_pads_when_zone_unknown():
    """Edge case: при незнакомой зоне фильтр всё равно возвращает >=3 услуги (padding)."""
    catalog = cat.load_catalog()
    filtered = cat.filter_services(catalog, ["несуществующая_зона"], industry="")

    # Согласно логике: padding гарантирует минимум 3 услуги
    assert len(filtered) >= 3


def test_filter_services_industry_specific_priority():
    """Edge case: услуги с industries=['all'] всегда подходят, даже при пустой отрасли."""
    catalog = [
        {"id": "a", "zone": "финансы", "industries": ["all"], "name": "A"},
        {"id": "b", "zone": "финансы", "industries": ["IT"], "name": "B"},
        {"id": "c", "zone": "маркетинг", "industries": ["all"], "name": "C"},
    ]
    filtered = cat.filter_services(catalog, ["финансы"], industry="")

    ids = {s["id"] for s in filtered}
    assert "a" in ids, "Услуга с industries=['all'] должна попасть в результат"


# ── Negative ──────────────────────────────────────────────────────────────────

def test_get_services_by_unknown_ids_returns_empty():
    """Негатив: запрос несуществующих id возвращает пустой список, а не падение."""
    catalog = cat.load_catalog()
    result = cat.get_services_by_ids(catalog, ["does_not_exist_1", "does_not_exist_2"])

    assert result == []


def test_get_services_by_ids_filters_correctly():
    """Happy path (вспомогательный): get_services_by_ids возвращает ровно запрошенные."""
    catalog = cat.load_catalog()
    if not catalog:
        pytest.skip("Каталог пуст")

    target_id = catalog[0]["id"]
    result = cat.get_services_by_ids(catalog, [target_id])

    assert len(result) == 1
    assert result[0]["id"] == target_id
