import pytest
import xml.etree.ElementTree as ET
import re
from app.router import detect_domain, get_sources_for_domain, NEWS_KEYWORDS


def test_detect_domain_news_keywords():
    claim1 = "Breaking: A major earthquake occurred near the border today."
    assert detect_domain(claim1) == "news"

    claim2 = "A ceasefire agreement was announced between both nations yesterday."
    assert detect_domain(claim2) == "news"

    claim3 = "Journalists reported that the election was held yesterday amidst heavy protests."
    assert detect_domain(claim3) == "news"


def test_news_sources_mapping():
    sources = get_sources_for_domain("news")
    assert "News" in sources
    assert "FactCheck" in sources
    assert "Wikipedia" in sources


def test_all_domains_include_wikipedia():
    domains = ["biomedical", "financial", "government", "news", "computing", "space", "general"]
    for d in domains:
        srcs = get_sources_for_domain(d)
        assert "Wikipedia" in srcs, f"Wikipedia missing from domain {d}"


def test_general_and_news_domains_include_all_three_new_sources():
    for d in ["general", "news"]:
        srcs = get_sources_for_domain(d)
        assert "Wikipedia" in srcs
        assert "FactCheck" in srcs
        assert "News" in srcs


def test_news_keywords_vocabulary():
    assert "breaking" in NEWS_KEYWORDS
    assert "ceasefire" in NEWS_KEYWORDS
    assert "reported" in NEWS_KEYWORDS
    assert "protest" in NEWS_KEYWORDS
    assert "yesterday" in NEWS_KEYWORDS
    assert "crisis" in NEWS_KEYWORDS


def test_news_rss_xml_parsing_logic():
    mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>&lt;b&gt;Global Trade Summit&lt;/b&gt; Concludes in Geneva - Reuters</title>
          <link>https://news.google.com/articles/12345</link>
          <pubDate>Mon, 20 Oct 2026 12:00:00 GMT</pubDate>
          <description>&lt;font color="#6f6f6f"&gt;Reuters&lt;/font&gt;&amp;nbsp;&amp;nbsp;World leaders finalized new export treaties today.</description>
          <source url="https://www.reuters.com">Reuters</source>
        </item>
      </channel>
    </rss>
    """
    root = ET.fromstring(mock_xml)
    items = root.findall(".//item")
    assert len(items) == 1

    item = items[0]
    raw_title = item.findtext("title", default="").strip()
    clean_title = re.sub(r"<[^>]+>", " ", raw_title).strip()
    assert clean_title == "Global Trade Summit  Concludes in Geneva - Reuters"

    description = item.findtext("description", default="").strip()
    clean_desc = re.sub(r"<[^>]+>", " ", description).strip()
    assert "World leaders finalized new export treaties today." in clean_desc


def test_wikipedia_pages_parsing_logic():
    mock_data = {
        "query": {
            "pages": {
                "456": {
                    "pageid": 456,
                    "title": "Artificial General Intelligence",
                    "extract": "Artificial general intelligence (AGI) is a hypothetical type of intelligent agent.",
                    "fullurl": "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
                }
            }
        }
    }
    pages = mock_data.get("query", {}).get("pages", {})
    results = []
    for _, page in pages.items():
        title = page.get("title", "").strip()
        extract = page.get("extract", "").strip()
        url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append({
            "title": title,
            "abstract": extract,
            "url": url,
            "sources": ["Wikipedia"],
        })

    assert len(results) == 1
    assert results[0]["title"] == "Artificial General Intelligence"
    assert "AGI" in results[0]["abstract"]
    assert results[0]["sources"] == ["Wikipedia"]
