from __future__ import annotations

import re
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
from bs4 import BeautifulSoup


USER_AGENTS = [
    # Windows Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    # macOS Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    # Windows Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 Edg/124.0",
    # Linux Chrome
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    # Firefox (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Firefox (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
    # iPhone Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    # Android Chrome
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36",
    # iPad Safari
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]


@dataclass
class ScrapedQuestion:
    question: str
    url: str
    title: Optional[str]
    site: Optional[str]
    discovered_at: datetime
    pdf_url: Optional[str] = None


class WebScraperService:
    def __init__(self) -> None:
        self.default_seeds: List[str] = [
            # Q&A hubs and interview resources (ensure ToS compliance externally)
            "https://stackoverflow.com/questions/tagged/interview",
            "https://stackoverflow.com/questions/tagged/system-design",
            "https://leetcode.com/discuss/interview-question",
            "https://leetcode.com/discuss/system-design",
            "https://www.geeksforgeeks.org/interview-corner/",
            "https://www.geeksforgeeks.org/category/interview-experiences/",
            "https://www.interviewbit.com/",
            "https://www.interviewcake.com/",
            "https://www.tutorialspoint.com/technical_interview_questions.htm",
            "https://www.javatpoint.com/interview-questions-and-answers",
            "https://www.freecodecamp.org/news/tag/interview/",
            "https://github.com/DopplerHQ/awesome-interview-questions",
            "https://www.scaler.com/topics/interview-questions/",
            "https://www.educative.io/blog/tag/interview",
            "https://www.codingninjas.com/codestudio/interview-experiences",
            "https://www.interviewkickstart.com/interview-questions",
            "https://www.simplilearn.com/resources/interview-questions",
            "https://www.edureka.co/blog/interview-questions/",
            "https://www.analyticsvidhya.com/blog/category/interview-questions/",
        ]
        self.timeout_sec = 10
        self.per_site_limit = 40
        self.crawl_depth = 1  # follow same-domain links to discover PDFs
        self.max_links_per_site = 15
        self._last_fetch_time_per_domain: dict[str, float] = {}
        self.min_interval_per_domain_sec = 0.5

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": random.choice(USER_AGENTS)}

    def _domain(self, url: str) -> str:
        return re.sub(r"^https?://(www\.)?", "", url).split("/")[0]

    def _normalize_url(self, base_url: str, href: str) -> Optional[str]:
        href = (href or "").strip()
        if not href:
            return None
        if href.startswith('//'):
            return 'https:' + href
        if href.startswith('/'):
            from urllib.parse import urljoin
            return urljoin(base_url, href)
        if href.startswith('http://') or href.startswith('https://'):
            return href
        return None

    def _collect_links(self, html: str, url: str) -> tuple[list[str], list[str]]:
        """Return (page_links, pdf_links) normalized and same-domain for page links."""
        soup = BeautifulSoup(html, "lxml")
        page_links: list[str] = []
        pdf_links: list[str] = []
        dom = self._domain(url)
        for a in soup.select('a[href]'):
            href = a.get('href') or ''
            norm = self._normalize_url(url, href)
            if not norm:
                continue
            if norm.lower().endswith('.pdf'):
                pdf_links.append(norm)
                continue
            if self._domain(norm) == dom:
                page_links.append(norm)
        # de-dup and cap
        def _dedup(seq: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for s in seq:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        return _dedup(page_links), _dedup(pdf_links)

    def _extract_questions(self, html: str, url: str, topic: str, extra_pdf_links: Optional[list[str]] = None) -> List[ScrapedQuestion]:
        soup = BeautifulSoup(html, "lxml")
        texts: List[str] = []

        # Titles, headings, list items, and question-like sentences
        for tag in soup.select("title, h1, h2, h3, h4, li, p"):
            text = (tag.get_text(separator=" ") or "").strip()
            if not text:
                continue
            texts.append(text)

        # Heuristics for question-like lines
        candidates: List[str] = []
        for t in texts:
            # Split long paragraphs into sentences
            parts = re.split(r"(?<=[?.!])\s+", t)
            for s in parts:
                s2 = s.strip()
                if len(s2) < 8:
                    continue
                # looks like a question or explicitly mentions interview/Q&A
                if s2.endswith("?") or re.search(r"(interview|question|how would you|explain|design)", s2, re.I):
                    candidates.append(s2)

        # Collect PDF links on page + extra discovered crawl links
        _, page_pdf_links = self._collect_links(html, url)
        pdf_links: List[str] = list(page_pdf_links)
        if extra_pdf_links:
            pdf_links.extend(extra_pdf_links)

        # Deduplicate and trim
        seen = set()
        results: List[ScrapedQuestion] = []
        title = (soup.title.string.strip() if soup.title and soup.title.string else None)
        site = self._domain(url)

        for c in candidates:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            # choose best-match pdf by topic token match
            pdf = None
            if pdf_links:
                topic_tokens = [t for t in re.split(r"\W+", topic.lower()) if t]
                scored = []
                for p in pdf_links:
                    fname = p.rsplit('/', 1)[-1].lower()
                    score = sum(1 for t in topic_tokens if t in fname)
                    scored.append((score, p))
                scored.sort(key=lambda x: (-x[0], x[1]))
                pdf = scored[0][1] if scored else pdf_links[0]
            results.append(ScrapedQuestion(
                question=c,
                url=url,
                title=title,
                site=site,
                discovered_at=datetime.utcnow(),
                pdf_url=pdf,
            ))

        return results[: self.per_site_limit]

    def _rate_limit(self, domain: str) -> None:
        now = time.time()
        last = self._last_fetch_time_per_domain.get(domain, 0.0)
        wait = self.min_interval_per_domain_sec - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch_time_per_domain[domain] = time.time()

    def _fetch(self, url: str) -> Optional[str]:
        try:
            self._rate_limit(self._domain(url))
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout_sec)
            if resp.status_code == 200 and resp.text:
                return resp.text
        except Exception:
            return None
        return None

    def scrape_topic(self, topic: str, seeds: Optional[List[str]] = None, max_sites: int = 15) -> List[ScrapedQuestion]:
        """Lightweight, best-effort scraping of seed URLs to extract question-like lines."""
        seeds = list(seeds or self.default_seeds)
        random.shuffle(seeds)
        collected: List[ScrapedQuestion] = []
        
        for url in seeds[:max_sites]:
            html = self._fetch(url)
            if not html:
                continue
            # Crawl same-domain shallowly to discover PDFs
            page_links, pdf_links = self._collect_links(html, url)
            discovered_pdfs: list[str] = list(pdf_links)
            if self.crawl_depth > 0 and page_links:
                for link in page_links[: self.max_links_per_site]:
                    sub_html = self._fetch(link)
                    if not sub_html:
                        continue
                    _, sub_pdfs = self._collect_links(sub_html, link)
                    discovered_pdfs.extend(sub_pdfs)

            items = self._extract_questions(html, url, topic, extra_pdf_links=discovered_pdfs)
            # Simple topical filter before heavier steps
            topic_l = topic.lower()
            topical = [i for i in items if topic_l in i.question.lower() or topic_l in (i.title or '').lower()]
            collected.extend(topical or items)
            # Respectful small delay between sites
            time.sleep(0.5)
        return collected


web_scraper_service = WebScraperService()


