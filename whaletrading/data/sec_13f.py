"""SEC EDGAR 13F holdings for a configurable set of large managers (free).

13F filings are quarterly with a 45-day lag — this is the slow "confirmation"
layer of the whale score, not a timing signal.

Flow per manager CIK:
  1. https://data.sec.gov/submissions/CIK##########.json  → recent 13F-HR filings
  2. filing index.json → locate the information-table XML
  3. parse holdings, match issuer names to watchlist tickers via aliases

SEC fair-access rules: descriptive User-Agent required, ≤10 requests/sec.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
REQUEST_GAP_SECONDS = 0.15


class EdgarClient:
    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get(self, url: str, **kwargs):
        time.sleep(REQUEST_GAP_SECONDS)
        resp = self.session.get(url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp


def fetch_13f_holdings(
    managers: list[dict],
    aliases: dict[str, list[str]],
    user_agent: str,
    filings_per_manager: int = 2,
) -> pd.DataFrame:
    """Return rows matching the inst_13f table for the last N filings per manager.

    `aliases` maps TICKER -> list of issuer-name substrings (case-insensitive).
    Managers or filings that fail are skipped with a warning.
    """
    matchers = {
        ticker: [a.upper() for a in names] for ticker, names in aliases.items() if names
    }
    if not matchers:
        log.warning("no issuer aliases configured — skipping 13F component")
        return pd.DataFrame()

    client = EdgarClient(user_agent)
    rows: list[dict] = []
    for mgr in managers:
        cik, name = int(mgr["cik"]), str(mgr.get("name", mgr["cik"]))
        try:
            filings = _recent_13f_filings(client, cik)[:filings_per_manager]
        except Exception as exc:
            log.warning("EDGAR submissions failed for %s (CIK %s): %s", name, cik, exc)
            continue
        for accession, report_period in filings:
            try:
                holdings = _fetch_info_table(client, cik, accession)
            except Exception as exc:
                log.warning("13F info table failed for %s %s: %s", name, accession, exc)
                continue
            for ticker, patterns in matchers.items():
                total_shares, total_value = 0, 0
                for h in holdings:
                    issuer = h["issuer"].upper()
                    if any(p in issuer for p in patterns):
                        total_shares += h["shares"]
                        total_value += h["value"]
                if total_shares:
                    rows.append(
                        {
                            "ticker": ticker,
                            "report_period": report_period,
                            "manager_cik": cik,
                            "manager_name": name,
                            "shares": total_shares,
                            "value_usd": total_value,
                        }
                    )
    return pd.DataFrame(rows)


def _recent_13f_filings(client: EdgarClient, cik: int) -> list[tuple[str, str]]:
    """Return [(accession_number, report_period), ...] newest first."""
    data = client.get(SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data.get("filings", {}).get("recent", {})
    out = []
    for form, accession, report in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("reportDate", []),
    ):
        if form == "13F-HR" and accession and report:
            out.append((accession, report))
    return out


def _fetch_info_table(client: EdgarClient, cik: int, accession: str) -> list[dict]:
    """Download and parse the information-table XML of one 13F-HR filing."""
    acc_nodash = accession.replace("-", "")
    index = client.get(
        ARCHIVES_URL.format(cik=cik, accession=acc_nodash) + "/index.json"
    ).json()
    xml_names = [
        item["name"]
        for item in index.get("directory", {}).get("item", [])
        if item.get("name", "").lower().endswith(".xml")
        and "primary_doc" not in item.get("name", "").lower()
    ]
    if not xml_names:
        raise ValueError("no information-table XML found in filing")
    # Prefer a file that looks like the info table; otherwise take the first XML.
    xml_names.sort(key=lambda n: ("infotable" not in n.lower(), n))
    xml_text = client.get(
        ARCHIVES_URL.format(cik=cik, accession=acc_nodash) + f"/{xml_names[0]}"
    ).text
    return _parse_info_table(xml_text)


def _parse_info_table(xml_text: str) -> list[dict]:
    # Strip namespaces so tag lookups are simple across filer variations.
    xml_text = re.sub(r'xmlns(:\w+)?="[^"]+"', "", xml_text)
    xml_text = re.sub(r"<(/?)\w+:", r"<\1", xml_text)
    root = ET.fromstring(xml_text)
    holdings = []
    for entry in root.iter("infoTable"):
        issuer = (entry.findtext("nameOfIssuer") or "").strip()
        value = _to_int(entry.findtext("value"))
        shares_el = entry.find("shrsOrPrnAmt")
        shares = _to_int(shares_el.findtext("sshPrnamt")) if shares_el is not None else 0
        stype = (
            shares_el.findtext("sshPrnamtType") if shares_el is not None else ""
        ) or ""
        if issuer and shares and stype.strip().upper() == "SH":
            holdings.append({"issuer": issuer, "shares": shares, "value": value})
    return holdings


def _to_int(text: str | None) -> int:
    try:
        return int(float(str(text).replace(",", "").strip()))
    except (TypeError, ValueError):
        return 0
