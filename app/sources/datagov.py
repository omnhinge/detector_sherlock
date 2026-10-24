import httpx
from typing import Any


WORLDBANK_API = "https://api.worldbank.org/v2"

# Mappings of common macroeconomic topics to official World Bank Indicator IDs
INDICATOR_MAP = {
    "gdp": ("NY.GDP.MKTP.CD", "Gross Domestic Product (GDP, current US$)"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "population": ("SP.POP.TOTL", "Total Population"),
    "unemployment": ("SL.UEM.TOTL.ZS", "Unemployment rate (% of total labor force)"),
    "life expectancy": ("SP.DYN.LE00.IN", "Life expectancy at birth (years)"),
    "co2": ("EN.ATM.CO2E.KT", "CO2 emissions (kt)"),
    "exports": ("NE.EXP.GNFS.CD", "Exports of goods and services (current US$)"),
    "imports": ("NE.IMP.GNFS.CD", "Imports of goods and services (current US$)"),
}

# Common country codes mapping
COUNTRY_MAP = {
    "india": "IND",
    "united states": "USA",
    "us": "USA",
    "usa": "USA",
    "china": "CHN",
    "japan": "JPN",
    "germany": "DEU",
    "united kingdom": "GBR",
    "uk": "GBR",
    "france": "FRA",
    "brazil": "BRA",
    "canada": "CAN",
    "world": "WLD",
    "global": "WLD",
}


def _detect_country_code(text: str) -> tuple[str, str]:
    """Detect country in query text. Defaults to ('IND', 'India') or ('WLD', 'World')."""
    text_lower = text.lower()
    for name, code in COUNTRY_MAP.items():
        if f" {name} " in f" {text_lower} ":
            return code, name.title()
    return "IND", "India"


def _detect_indicator(text: str) -> tuple[str, str]:
    """Detect indicator in query text. Defaults to GDP."""
    text_lower = text.lower()
    for keyword, (ind_id, ind_name) in INDICATOR_MAP.items():
        if keyword in text_lower:
            return ind_id, ind_name
    return "NY.GDP.MKTP.CD", "Gross Domestic Product (GDP, current US$)"


async def search_datagov(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search official macroeconomic and public statistics from the World Bank
    and Open Government datasets.
    Provides verifiable, authoritative figures for GDP, inflation, population,
    and economic indicators.
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(limit, 10))
    country_code, country_name = _detect_country_code(query)
    ind_code, ind_name = _detect_indicator(query)

    url = f"{WORLDBANK_API}/country/{country_code}/indicator/{ind_code}"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                params={"format": "json", "per_page": limit, "date": "2018:2025"},
            )
            if response.status_code != 200:
                return []

            data = response.json()
            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                return []

            records = data[1]
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    for item in records:
        val = item.get("value")
        if val is None:
            continue

        date_yr = item.get("date", "Recent")
        country_display = item.get("country", {}).get("value", country_name)
        indicator_display = item.get("indicator", {}).get("value", ind_name)

        # Format number nicely if it's large currency or percentage
        if isinstance(val, (int, float)):
            if val > 1_000_000_000_000:
                val_formatted = f"${val / 1_000_000_000_000:.2f} trillion (raw: {val:,.2f})"
            elif val > 1_000_000:
                val_formatted = f"${val / 1_000_000:.2f} million (raw: {val:,.2f})"
            elif "%" in indicator_display or "rate" in indicator_display.lower():
                val_formatted = f"{val:.2f}%"
            else:
                val_formatted = f"{val:,.2f}"
        else:
            val_formatted = str(val)

        title = f"{country_display} — {indicator_display} ({date_yr})"
        abstract = (
            f"Official macroeconomic record from World Bank / Open Government Data: "
            f"The {indicator_display} for {country_display} in {date_yr} is recorded as {val_formatted}. "
            f"Indicator code: {ind_code}."
        )
        record_url = f"https://data.worldbank.org/indicator/{ind_code}?locations={country_code}"

        results.append(
            {
                "title": title,
                "abstract": abstract,
                "url": record_url,
                "doi": None,
                "sources": ["World Bank / GovData"],
            }
        )

    return results
