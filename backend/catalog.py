"""Service catalog loading and filtering."""
import json
from pathlib import Path

from .agent_logger import log_agent_step

_ROOT = Path(__file__).parent.parent  # repo root


def load_catalog() -> list[dict]:
    path = _ROOT / "data" / "services_catalog.json"
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    log_agent_step("catalog.load_catalog", "success", service_count=len(catalog))
    return catalog


def filter_services(
    catalog: list[dict],
    risk_zone_ids: list[str],
    industry: str = "",
) -> list[dict]:
    """
    Return services relevant to the identified risk zones.

    Priority order:
    1. Services matching both zone and industry
    2. Services matching zone with industries=["all"]
    3. If fewer than 3 results, pad with highest-impact services
    """
    zone_set = set(z.lower() for z in risk_zone_ids)
    industry_lower = industry.lower()

    matched = []
    seen_ids = set()

    for svc in catalog:
        svc_zone = svc.get("zone", "").lower()
        svc_industries = [i.lower() for i in svc.get("industries", ["all"])]

        if svc_zone not in zone_set:
            continue

        if "all" in svc_industries or any(ind in industry_lower for ind in svc_industries if ind != "all"):
            matched.append(svc)
            seen_ids.add(svc["id"])

    if len(matched) < 3:
        log_agent_step(
            "catalog.filter_services",
            "padding",
            requested_zones=sorted(zone_set),
            industry=industry,
            matched_before_padding=len(matched),
        )
        for svc in catalog:
            if svc["id"] not in seen_ids and svc.get("zone", "").lower() in zone_set:
                matched.append(svc)
                seen_ids.add(svc["id"])

    if len(matched) < 3:
        for svc in catalog:
            if svc["id"] not in seen_ids:
                matched.append(svc)
                seen_ids.add(svc["id"])
            if len(matched) >= 3:
                break

    log_agent_step(
        "catalog.filter_services",
        "success",
        requested_zones=sorted(zone_set),
        industry=industry,
        result_count=len(matched),
    )
    return matched


def get_services_by_ids(catalog: list[dict], ids: list[str]) -> list[dict]:
    id_set = set(ids)
    result = [s for s in catalog if s["id"] in id_set]
    log_agent_step("catalog.get_services_by_ids", "success", requested_count=len(ids), result_count=len(result))
    return result
