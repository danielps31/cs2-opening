"""Shared config loading."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


@lru_cache
def model_config() -> dict:
    with open(CONFIG_DIR / "model_config.yaml") as f:
        return yaml.safe_load(f)


@lru_cache
def maps_config() -> dict:
    with open(CONFIG_DIR / "maps.yaml") as f:
        return yaml.safe_load(f)


def map_zones(map_name: str) -> dict[str, list[str]]:
    cfg = maps_config()
    entry = cfg["maps"].get(map_name) or {}
    return entry.get("zones") or cfg["default_zones"]


def map_radar(map_name: str) -> dict | None:
    entry = maps_config()["maps"].get(map_name) or {}
    return entry.get("radar")
