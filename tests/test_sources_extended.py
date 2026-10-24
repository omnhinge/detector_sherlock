import pytest
from app.router import detect_domain, get_sources_for_domain
from app.sources.datagov import _detect_country_code, _detect_indicator


def test_detect_domain_financial():
    claim = "Tesla reported delivery figures for its electric vehicles in the first fiscal quarter."
    assert detect_domain(claim) == "financial"
    sources = get_sources_for_domain("financial")
    assert "SEC EDGAR" in sources
    assert "PubMed" not in sources


def test_detect_domain_government():
    claim = "India reported strong GDP growth and reduced inflation in 2024."
    assert detect_domain(claim) == "government"
    sources = get_sources_for_domain("government")
    assert "World Bank / GovData" in sources
    assert "PubMed" not in sources


def test_datagov_country_and_indicator_detection():
    country_code, country_name = _detect_country_code("The inflation rate in Japan increased slightly.")
    assert country_code == "JPN"
    assert country_name == "Japan"

    ind_code, ind_name = _detect_indicator("The inflation rate in Japan increased slightly.")
    assert ind_code == "FP.CPI.TOTL.ZG"
    assert "Inflation" in ind_name
