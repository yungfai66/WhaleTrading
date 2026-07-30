"""Load and validate config/watchlist.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "watchlist.yaml"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_THRESHOLDS = {"momentum": 35, "rise": 50, "soar": 75}
DEFAULT_WEIGHTS = {"big_money_volume": 0.45, "dark_pool": 0.35, "inst_13f": 0.20}


@dataclass
class Config:
    watchlists: dict[str, list[str]]
    default_watchlist: str
    thresholds_default: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    thresholds_overrides: dict = field(default_factory=dict)
    price_lookback_years: int = 5
    finra_short_volume_days: int = 365
    baseline_window: int = 60
    volume_zscore_threshold: float = 1.25
    flow_window: int = 20
    whale_weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    sec_user_agent: str = "WhaleTrading research (set contact in config)"
    managers_13f: list[dict] = field(default_factory=list)
    issuer_aliases: dict = field(default_factory=dict)

    def thresholds_for(self, ticker: str) -> dict:
        merged = dict(self.thresholds_default)
        merged.update(self.thresholds_overrides.get(ticker.upper(), {}))
        return merged

    @property
    def all_tickers(self) -> list[str]:
        """Deduped union of every watchlist's tickers, in first-appearance
        order. Used by the CLI entry point (`python -m whaletrading.pipeline`
        with no args) for a full manual refresh — the app itself always
        refreshes just the currently active watchlist, not everything."""
        seen: set[str] = set()
        out: list[str] = []
        for tickers in self.watchlists.values():
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    @property
    def demo_mode(self) -> bool:
        return os.environ.get("WHALETRADING_DEMO", "").strip() in ("1", "true", "yes")


def load_config(path: Path | str | None = None) -> Config:
    raw = yaml.safe_load(Path(path or CONFIG_PATH).read_text()) or {}
    settings = raw.get("settings") or {}
    thresholds = raw.get("thresholds") or {}

    watchlists_raw = raw.get("watchlists") or {}
    watchlists = {
        str(name): [str(t).upper() for t in (tickers or []) if str(t).strip()]
        for name, tickers in watchlists_raw.items()
    }
    watchlists = {name: tickers for name, tickers in watchlists.items() if tickers}
    if not watchlists:
        raise ValueError("config has no non-empty watchlists — add at least one ticker to one watchlist")

    default_watchlist = str(raw.get("default_watchlist") or next(iter(watchlists)))
    if default_watchlist not in watchlists:
        default_watchlist = next(iter(watchlists))

    weights = dict(DEFAULT_WEIGHTS)
    weights.update(settings.get("whale_weights") or {})

    default_thr = dict(DEFAULT_THRESHOLDS)
    default_thr.update(thresholds.get("default") or {})

    return Config(
        watchlists=watchlists,
        default_watchlist=default_watchlist,
        thresholds_default=default_thr,
        thresholds_overrides={
            str(k).upper(): dict(v) for k, v in (thresholds.get("overrides") or {}).items()
        },
        price_lookback_years=int(settings.get("price_lookback_years", 5)),
        finra_short_volume_days=int(settings.get("finra_short_volume_days", 365)),
        baseline_window=int(settings.get("baseline_window", 60)),
        volume_zscore_threshold=float(settings.get("volume_zscore_threshold", 1.25)),
        flow_window=int(settings.get("flow_window", 20)),
        whale_weights=weights,
        sec_user_agent=str(settings.get("sec_user_agent", Config.sec_user_agent)),
        managers_13f=list(raw.get("managers_13f") or []),
        issuer_aliases={
            str(k).upper(): [str(a) for a in v] for k, v in (raw.get("issuer_aliases") or {}).items()
        },
    )
