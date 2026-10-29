from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, quote_plus, urljoin, urlparse

import yaml

from .basic import (
    ALGORITHMS_PATH,
    TASK_TYPES_PATH,
    clear_knowledge_base_cache,
    format_algorithm_record,
    get_algorithm_detail,
)


from aag.utils.path_utils import DEFAULT_CONFIG_PATH

YIGRAPH_CONFIG_PATH = DEFAULT_CONFIG_PATH
BASE_DIR = Path(__file__).resolve().parent
ALGORITHMS_NEW_PATH = ALGORITHMS_PATH.with_name("algorithms_new.yaml")

NETWORKX_DOCS_BASE = "https://networkx.org/documentation/stable/"
NETWORKX_SEARCHINDEX_URL = urljoin(NETWORKX_DOCS_BASE, "searchindex.js")
NETWORKX_GENERATED_MARKER = "reference/algorithms/generated/"
SCHOLAR_BASE_URL = "https://scholar.google.com/scholar"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_BASE = "https://en.wikipedia.org/wiki/"
DEFAULT_TIMEOUT = 30
PDF_TIMEOUT = 45
OPENALEX_WORK_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
PDF_HINTS = (".pdf", "/pdf/", "download", "/retrieve/", "/bitstream/", "viewcontent.cgi", "/document/")


@dataclass(slots=True)
class YiGraphLLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 3500


@dataclass(slots=True)
class CanonicalAlgorithmName:
    raw_name: str
    normalized_name: str
    slug_name: str
    query_candidates: list[str]


@dataclass(slots=True)
class NetworkXCandidate:
    title: str
    url: str
    filename: str
    docname: str
    score: float
    matched_variants: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperCandidate:
    title: str
    landing_url: str | None
    pdf_url: str | None
    doi: str | None
    authors: str | None
    year: str | None
    snippet: str | None
    query: str


@dataclass(slots=True)
class PdfCandidate:
    url: str
    source: str
    note: str = ""
    doi: str | None = None


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str


@dataclass(slots=True)
class FieldMapping:
    field: str
    content: str


@dataclass(slots=True)
class EvidenceRecord:
    source_id: str
    source_type: str
    source_url: str
    excerpt: str
    field_mappings: list[FieldMapping] = field(default_factory=list)


class YiGraphLLMClient:
    def __init__(self, config: YiGraphLLMConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required to call the configured LLM API") from exc

        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
        return response.choices[0].message.content or ""


class SearchDrivenAlgorithmGenerator:
    def __init__(self, llm_client: YiGraphLLMClient, progress: Callable[[str], None] | None = None) -> None:
        self.llm_client = llm_client
        self.progress = progress or (lambda _message: None)
        self.session = self._build_session()

    def generate(self, algorithm_name: str) -> dict[str, Any]:
        self.progress(f"Normalize algorithm name: {algorithm_name}")
        canonical = normalize_algorithm_name(algorithm_name)

        self.progress("Search NetworkX documentation")
        networkx_lookup = find_networkx_candidates(self.session, canonical.query_candidates)
        self.progress(f"NetworkX candidates: {len(networkx_lookup['candidates'])}")
        selected_networkx, selection_notes = select_best_networkx_candidate(
            networkx_lookup["candidates"],
            canonical.slug_name,
        )
        networkx_lookup["selection_notes"] = selection_notes
        networkx_lookup["selected_candidate_url"] = selected_networkx.url if selected_networkx else None

        networkx_source = {
            "networkx_source_found": False,
            "networkx_url": None,
            "networkx_http_status": None,
            "networkx_full_text": "",
            "networkx_chunks_for_llm": [],
            "networkx_complexity_candidates": [],
            "networkx_parse_notes": [],
        }
        if selected_networkx is not None:
            self.progress(f"Fetch NetworkX page: {selected_networkx.url}")
            networkx_source = fetch_networkx_page_text(self.session, selected_networkx.url)
            networkx_source["networkx_chunks_for_llm"] = chunk_text(str(networkx_source.get("networkx_full_text", "")))
        else:
            self.progress("No NetworkX page selected")

        self.progress("Search Google Scholar")
        scholar_results = self._collect_scholar_results(canonical.query_candidates)
        selected_papers = select_top_papers(scholar_results, canonical.query_candidates, limit=3)
        self.progress(f"Scholar papers selected: {len(selected_papers)}")
        paper_sources = self._collect_paper_sources(selected_papers)

        self.progress("Search Wikipedia")
        wikipedia_search, wikipedia_source = fetch_wikipedia_source(self.session, canonical.raw_name)
        self.progress(f"Wikipedia source found: {bool(wikipedia_source.get('wikipedia_source_found'))}")

        networkx_materials = prepare_networkx_materials(networkx_source)
        paper_materials = prepare_paper_materials(paper_sources)
        wikipedia_materials = prepare_wikipedia_materials(wikipedia_source)
        self.progress(
            "Prepared materials: "
            f"networkx={len(networkx_materials)}, paper={len(paper_materials)}, wikipedia={len(wikipedia_materials)}"
        )
        if not networkx_materials and not paper_materials and not wikipedia_materials:
            errors = []
            if networkx_lookup.get("error"):
                errors.append(f"networkx={networkx_lookup['error']}")
            for result in scholar_results:
                if result.get("error"):
                    errors.append(f"scholar:{result.get('query')}={result['error']}")
            if wikipedia_search.get("error"):
                errors.append(f"wikipedia={wikipedia_search['error']}")
            details = "; ".join(errors) if errors else "all sources returned empty results"
            raise RuntimeError(f"No usable materials were retrieved: {details}")

        evidence_prompt = build_evidence_prompt(
            canonical.raw_name,
            networkx_materials,
            paper_materials,
            wikipedia_materials,
            abstract_only=bool(paper_materials) and not networkx_materials,
        )
        self.progress("Call LLM: evidence mapping")
        evidence_raw = self.llm_client.generate(evidence_prompt, temperature=0.2, max_tokens=3500)
        evidence_records = parse_evidence_response(evidence_raw)
        self.progress(f"Evidence records parsed: {len(evidence_records)}")

        task_types = load_task_types()
        entry_prompt = build_entry_json_prompt(canonical.slug_name, evidence_records, task_types)
        self.progress("Call LLM: entry JSON generation")
        entry_raw = self.llm_client.generate(entry_prompt, temperature=0.1, max_tokens=2500)
        generated_entry = parse_generated_entry_json(entry_raw)
        yaml_output = yaml.safe_dump([generated_entry], allow_unicode=True, sort_keys=False, width=120)
        validation = validate_generated_yaml(yaml_output, evidence_records)
        self.progress(f"Generated YAML valid: {validation['yaml_valid']}")

        return {
            "final_yaml": yaml_output,
            "validation": validation,
        }

    def _collect_scholar_results(self, queries: list[str]) -> list[dict[str, Any]]:
        results = []
        seen = set()
        for query in queries:
            normalized = query.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            self.progress(f"Search Scholar query: {normalized}")
            result = search_scholar(self.session, normalized, max_results=5)
            self.progress(f"Scholar query result papers: {len(result.get('papers', []))}")
            results.append(result)
        return results

    def _collect_paper_sources(self, papers: list[PaperCandidate]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, paper in enumerate(papers, start=1):
            source_id = f"paper_{index}"
            self.progress(f"Resolve PDF candidates for paper {index}: {paper.title}")
            pdf_candidates, resolution_logs = discover_pdf_candidates(self.session, paper)
            self.progress(f"PDF candidates discovered for paper {index}: {len(pdf_candidates)}")

            pdf_source_built = False
            for candidate_index, candidate in enumerate(pdf_candidates, start=1):
                self.progress(f"Download PDF candidate {candidate_index}/{len(pdf_candidates)} for paper {index}: {candidate.source}")
                pdf_bytes, download_meta = download_pdf(self.session, candidate.url)
                if not pdf_bytes:
                    self.progress(f"PDF candidate failed at {download_meta.get('failure_stage')}: {download_meta.get('error')}")
                    continue

                self.progress(f"Parse PDF pages for paper {index}")
                pages, parse_meta = extract_pages_from_pdf(pdf_bytes)
                if parse_meta.get("error"):
                    self.progress(f"PDF parse failed: {parse_meta['error']}")
                    continue

                self.progress(f"Chunk PDF for paper {index}: pages={parse_meta.get('page_count')}")
                full_text, chunks, complexity_candidates, chunk_meta = build_paper_chunks(pages)
                self.progress(
                    "PDF chunk summary: "
                    f"text={chunk_meta.get('raw_text_length')}, complexity={chunk_meta.get('complexity_candidate_count')}"
                )
                if full_text.strip() and chunks_has_substance(chunks):
                    resolved_pdf_url = str(download_meta.get("final_url") or candidate.url)
                    sources.append(
                        {
                            "source_id": source_id,
                            "title": paper.title,
                            "source_url": resolved_pdf_url,
                            "source_type": "paper_pdf",
                            "landing_url": paper.landing_url,
                            "pdf_url": resolved_pdf_url,
                            "raw_text": full_text,
                            "chunks": chunks,
                            "complexity_candidates": complexity_candidates,
                            "notes": [f"resolved_from={candidate.source}", candidate.note],
                            "error": None,
                        }
                    )
                    pdf_source_built = True
                    break

            if pdf_source_built:
                continue

            self.progress(f"PDF unavailable for paper {index}; fallback to abstract page")
            if not paper.landing_url:
                continue
            source, meta = fetch_paper_abstract_source(
                self.session,
                source_id=source_id,
                source_url=paper.landing_url,
                title_hint=paper.title,
                pdf_url=paper.pdf_url,
            )
            if source is not None:
                sources.append(source)
        self.progress(f"Paper sources fetched: {len(sources)}")
        return sources

    @staticmethod
    def _build_session():
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests package is required for algorithm creation") from exc

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        return session


def create_algorithm(
    name: str,
    lang: str = "zh-CN",
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    algorithm_name = (name or "").strip()
    if not algorithm_name:
        raise ValueError("Algorithm name is required")

    emit = progress or (lambda _message: None)
    emit(f"Start creating algorithm: {algorithm_name}")
    normalized_id = normalize_algorithm_name(algorithm_name).slug_name
    existing = get_algorithm_detail(normalized_id, lang=lang)
    if existing is not None and not overwrite:
        emit(f"Algorithm already exists in algorithms.yaml: {normalized_id}")
        return {
            "success": True,
            "data": existing["data"],
        }

    emit("Load YiGraph LLM config")
    config = load_yigraph_llm_config()
    generator = SearchDrivenAlgorithmGenerator(YiGraphLLMClient(config), progress=emit)
    generation_result = generator.generate(algorithm_name)
    emit("Parse generated YAML")
    generated_entry = parse_generated_yaml_entry(generation_result["final_yaml"])
    normalize_generated_entry(generated_entry, normalized_id)
    emit(f"Write generated algorithm to {ALGORITHMS_NEW_PATH}")
    append_algorithm_entry(generated_entry, overwrite=overwrite)
    emit(f"Update task type mapping in {TASK_TYPES_PATH}")
    update_task_type_mapping(generated_entry, overwrite=overwrite)
    clear_knowledge_base_cache()
    emit("Done")

    return {
        "success": True,
        "data": format_algorithm_record(generated_entry, lang=lang),
    }


def load_yigraph_llm_config(config_path: Path = YIGRAPH_CONFIG_PATH) -> YiGraphLLMConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"YiGraph engine config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    llm_config = ((config.get("reasoner") or {}).get("llm") or {})
    provider = str(llm_config.get("provider") or "openai").strip().lower()
    if provider != "openai":
        raise ValueError(f"Algorithm creation requires an OpenAI-compatible LLM provider, got: {provider}")

    openai_config = llm_config.get("openai") or {}
    api_key = str(openai_config.get("api_key") or "").strip()
    base_url = str(openai_config.get("base_url") or "").strip()
    model = str(openai_config.get("model") or "").strip()

    if not api_key or api_key == "your-api-key-here":
        raise ValueError(f"OpenAI API key is not configured in {config_path}")
    if not base_url or "your-api-endpoint" in base_url:
        raise ValueError(f"OpenAI base_url is not configured in {config_path}")
    if not model:
        raise ValueError(f"OpenAI model is not configured in {config_path}")

    return YiGraphLLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(openai_config.get("temperature", 0.1)),
        max_tokens=int(openai_config.get("max_tokens", 3500)),
    )


def normalize_algorithm_name(algorithm_name: str) -> CanonicalAlgorithmName:
    raw = (algorithm_name or "").strip()
    normalized = re.sub(r"\s+", " ", raw.replace("-", " ").replace("_", " ")).strip().lower()
    slug = normalized.replace(" ", "_")
    queries = []
    for candidate in (raw, normalized, slug):
        value = candidate.strip()
        if value and value not in queries:
            queries.append(value)
    return CanonicalAlgorithmName(raw_name=raw, normalized_name=normalized, slug_name=slug, query_candidates=queries)


def find_networkx_candidates(session: Any, variants: list[str]) -> dict[str, Any]:
    result = {"index_url": NETWORKX_SEARCHINDEX_URL, "http_status": None, "candidates": [], "error": None, "notes": []}
    try:
        response = session.get(NETWORKX_SEARCHINDEX_URL, timeout=DEFAULT_TIMEOUT)
        result["http_status"] = response.status_code
        response.raise_for_status()
        index_data = parse_searchindex_payload(response.text)
        result["candidates"] = score_networkx_candidates(index_data, variants)
        result["notes"].append(f"parsed_searchindex_candidates={len(result['candidates'])}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["notes"].append("searchindex_fetch_or_parse_failed")
    return result


def fetch_networkx_page_text(session: Any, url: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "networkx_source_found": False,
        "networkx_url": url,
        "networkx_http_status": None,
        "networkx_full_text": "",
        "networkx_complexity_candidates": [],
        "networkx_parse_notes": [],
    }
    if NETWORKX_GENERATED_MARKER not in url:
        payload["networkx_parse_notes"] = ["not_generated_page"]
        return payload

    try:
        response = session.get(url, timeout=DEFAULT_TIMEOUT)
        payload["networkx_http_status"] = response.status_code
        response.raise_for_status()
        text, notes = extract_main_text(response.text)
        payload["networkx_source_found"] = bool(text)
        payload["networkx_full_text"] = text
        payload["networkx_complexity_candidates"] = extract_complexity_candidates(text)
        payload["networkx_parse_notes"] = notes
    except Exception as exc:
        payload["networkx_parse_notes"] = [f"{type(exc).__name__}: {exc}"]
    return payload


def parse_searchindex_payload(raw_text: str) -> dict[str, Any]:
    match = re.search(r"Search\.setIndex\((.*)\)\s*;?\s*$", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("searchindex.js did not match expected Sphinx format")
    return json.loads(match.group(1))


def score_networkx_candidates(index_data: dict[str, Any], variants: list[str]) -> list[NetworkXCandidate]:
    filenames = index_data.get("filenames") or []
    docnames = index_data.get("docnames") or []
    titles = index_data.get("titles") or []
    prepared_variants = [prepare_variant(variant) for variant in variants if variant.strip()]
    candidates: dict[str, NetworkXCandidate] = {}

    for idx, filename in enumerate(filenames):
        if not isinstance(filename, str) or NETWORKX_GENERATED_MARKER not in filename:
            continue
        docname = str(docnames[idx] if idx < len(docnames) else filename)
        title = str(titles[idx] if idx < len(titles) else docname)
        score, matched = candidate_score(filename, docname, title, prepared_variants)
        if score <= 0:
            continue
        url = urljoin(NETWORKX_DOCS_BASE, filename.replace(".rst", ".html"))
        candidate = NetworkXCandidate(title=title, url=url, filename=filename, docname=docname, score=score, matched_variants=matched)
        if url not in candidates or candidates[url].score < score:
            candidates[url] = candidate
    return sorted(candidates.values(), key=lambda item: (-item.score, item.url))


def select_best_networkx_candidate(candidates: list[NetworkXCandidate], canonical_slug: str) -> tuple[NetworkXCandidate | None, list[str]]:
    if not candidates:
        return None, ["no_candidates"]
    exact = [candidate for candidate in candidates if basename(candidate.filename) == canonical_slug]
    if exact:
        return sorted(exact, key=lambda item: (path_depth(item.docname), item.url))[0], ["selected_shortest_exact_basename_path"]
    if len(candidates) == 1:
        return candidates[0], ["single_candidate_available"]
    ranked = sorted(candidates, key=lambda item: (-item.score, item.url))
    if len(ranked) >= 2 and ranked[0].score >= ranked[1].score + 5:
        return ranked[0], ["selected_by_score_gap"]
    return None, ["candidate_not_resolved"]


def fetch_wikipedia_source(session: Any, algorithm_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    search_payload = search_wikipedia(session, algorithm_name)
    source_payload: dict[str, Any] = {
        "wikipedia_source_found": False,
        "page_title": None,
        "page_url": None,
        "http_status": None,
        "lead_text": "",
        "infobox_text": "",
        "sections": [],
        "parse_notes": [],
        "error": None,
    }
    selected_title = search_payload.get("selected_title")
    if not selected_title:
        source_payload["parse_notes"] = ["no_selected_wikipedia_page"]
        return search_payload, source_payload
    try:
        response = session.get(
            WIKIPEDIA_API_URL,
            params={"action": "parse", "page": selected_title, "prop": "text|displaytitle", "redirects": 1, "format": "json", "formatversion": 2},
            timeout=DEFAULT_TIMEOUT,
        )
        source_payload["http_status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        parse_payload = payload.get("parse") or {}
        page_title = str(parse_payload.get("title") or selected_title).strip()
        lead_text, infobox_text, sections, parse_notes = extract_wikipedia_content(str(parse_payload.get("text") or ""))
        source_payload.update(
            {
                "wikipedia_source_found": bool(lead_text or infobox_text or sections),
                "page_title": page_title,
                "page_url": build_wikipedia_page_url(page_title),
                "lead_text": lead_text,
                "infobox_text": infobox_text,
                "sections": sections,
                "parse_notes": parse_notes,
            }
        )
    except Exception as exc:
        source_payload["error"] = f"{type(exc).__name__}: {exc}"
        source_payload["parse_notes"] = ["wikipedia_page_fetch_failed"]
    return search_payload, source_payload


def search_wikipedia(session: Any, algorithm_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": algorithm_name,
        "api_url": WIKIPEDIA_API_URL,
        "http_status": None,
        "candidates": [],
        "selected_title": None,
        "selected_url": None,
        "error": None,
        "notes": [],
    }
    try:
        response = session.get(
            WIKIPEDIA_API_URL,
            params={"action": "query", "list": "search", "srsearch": algorithm_name, "srlimit": 5, "format": "json", "formatversion": 2},
            timeout=DEFAULT_TIMEOUT,
        )
        payload["http_status"] = response.status_code
        response.raise_for_status()
        search_results = ((response.json() or {}).get("query") or {}).get("search") or []
        candidates = []
        for item in search_results:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            snippet = clean_text(str(item.get("snippet") or ""))
            score = text_candidate_score(algorithm_name, title, snippet)
            candidates.append({"title": title, "snippet": snippet, "pageid": item.get("pageid"), "score": score, "url": build_wikipedia_page_url(title)})
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["title"]).lower()))
        payload["candidates"] = candidates
        if candidates:
            payload["selected_title"] = candidates[0]["title"]
            payload["selected_url"] = candidates[0]["url"]
            payload["notes"].append(f"selected_from_search_candidates={len(candidates)}")
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["notes"].append("wikipedia_search_failed")
    return payload


def search_scholar(session: Any, query: str, max_results: int = 5) -> dict[str, Any]:
    search_url = f"{SCHOLAR_BASE_URL}?hl=zh-CN&as_sdt=0%2C5&q={quote_plus(query)}&btnG="
    result: dict[str, Any] = {"query": query, "search_url": search_url, "http_status": None, "papers": [], "error": None, "notes": []}
    try:
        time.sleep(random.uniform(1, 2))
        response = session.get(search_url, timeout=DEFAULT_TIMEOUT)
        result["http_status"] = response.status_code
        response.raise_for_status()
        if "recaptcha" in response.text.lower() or "unusual traffic" in response.text.lower():
            result["error"] = "Scholar returned CAPTCHA"
            result["notes"].append("captcha_detected")
            return result
        result["papers"] = parse_scholar_results(response.text, query, max_results)
        result["notes"].append(f"parsed={len(result['papers'])}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["notes"].append("fetch_failed")
    return result


def parse_scholar_results(html: str, query: str, limit: int) -> list[PaperCandidate]:
    soup = make_soup(html)
    papers = []
    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
    for item in soup.select("div.gs_r.gs_or.gs_scl")[:limit]:
        heading = item.select_one("h3.gs_rt")
        if not heading:
            continue
        link = heading.find("a", href=True)
        title = re.sub(r"^\[[^\]]+\]\s*", "", heading.get_text(" ", strip=True))
        pdf_link = item.select_one("div.gs_or_ggsm a[href]")
        metadata = item.select_one("div.gs_a")
        metadata_text = metadata.get_text(" ", strip=True) if metadata else ""
        year_match = re.search(r"\b(19|20)\d{2}\b", metadata_text)
        snippet_node = item.select_one("div.gs_rs, div.gs_a + div")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else None
        doi_match = doi_pattern.search(" ".join([title, link["href"] if link else "", pdf_link["href"] if pdf_link else "", item.get_text(" ")]))
        papers.append(
            PaperCandidate(
                title=title,
                landing_url=link["href"].strip() if link else None,
                pdf_url=pdf_link["href"].strip() if pdf_link else None,
                doi=doi_match.group(0) if doi_match else None,
                authors=metadata_text.split(" - ")[0] if metadata_text else None,
                year=year_match.group(0) if year_match else None,
                snippet=snippet,
                query=query,
            )
        )
    return papers


def fetch_paper_abstract_source(session: Any, source_id: str, source_url: str, title_hint: str, pdf_url: str | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {"source_id": source_id, "source_url": source_url, "http_status": None, "error": None, "notes": []}
    try:
        response = session.get(source_url, timeout=DEFAULT_TIMEOUT)
        meta["http_status"] = response.status_code
        response.raise_for_status()
        soup = make_soup(response.text)
        title = extract_page_title(soup, title_hint)
        abstract = extract_abstract(soup)
        visible_text = extract_visible_text(soup)
        complexity_candidates = extract_complexity_candidates(visible_text)
        source = {
            "source_id": source_id,
            "title": title or title_hint,
            "source_url": source_url,
            "source_type": "paper_abstract",
            "landing_url": source_url,
            "pdf_url": pdf_url,
            "raw_text": visible_text or abstract or title,
            "chunks": {
                "title": title or title_hint,
                "abstract": abstract,
                "intro_chunks": [abstract] if abstract else [],
                "method_chunks": [],
                "complexity_chunks": [candidate["excerpt"] for candidate in complexity_candidates],
                "conclusion_chunks": keyword_sentences(visible_text, ("application", "applications", "discussion", "limitation", "limitations")),
            },
            "complexity_candidates": complexity_candidates,
            "notes": meta["notes"],
            "error": None,
        }
        return source, meta
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta


def discover_pdf_candidates(session: Any, paper: PaperCandidate, max_candidates: int = 12) -> tuple[list[PdfCandidate], list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    candidates: list[PdfCandidate] = []
    seen_candidates: set[str] = set()
    inspect_queue: list[tuple[str, str]] = []
    seen_inspect_urls: set[str] = set()
    queried_dois: set[str] = set()

    doi = paper.doi or extract_doi_from_text(" ".join(filter(None, [paper.landing_url, paper.pdf_url, paper.title, paper.snippet])))
    if paper.pdf_url:
        append_pdf_candidate(candidates, seen_candidates, PdfCandidate(paper.pdf_url, "scholar_rhs_pdf", "Scholar right-hand PDF candidate", doi))
        queue_inspect_url(inspect_queue, seen_inspect_urls, paper.pdf_url, "scholar_rhs_url")
    if paper.landing_url:
        if looks_like_pdf_url(paper.landing_url):
            append_pdf_candidate(candidates, seen_candidates, PdfCandidate(paper.landing_url, "landing_pdf", "Landing URL resembles a PDF endpoint", doi))
        queue_inspect_url(inspect_queue, seen_inspect_urls, paper.landing_url, "landing_url")
    if doi:
        extend_from_openalex(session, doi, candidates, seen_candidates, inspect_queue, seen_inspect_urls, logs, queried_dois)

    inspect_index = 0
    while inspect_index < len(inspect_queue) and len(candidates) < max_candidates:
        inspect_url, source_label = inspect_queue[inspect_index]
        inspect_index += 1
        inspection = inspect_html_or_pdf_endpoint(session, inspect_url, source_label)
        logs.append(inspection["log"])
        if inspection["verified_pdf_url"]:
            append_pdf_candidate(
                candidates,
                seen_candidates,
                PdfCandidate(str(inspection["verified_pdf_url"]), "page_direct_pdf", f"Verified directly while inspecting {source_label}", doi),
            )
            continue
        discovered_doi = inspection.get("doi")
        if isinstance(discovered_doi, str) and discovered_doi and discovered_doi not in queried_dois:
            doi = discovered_doi
            extend_from_openalex(session, doi, candidates, seen_candidates, inspect_queue, seen_inspect_urls, logs, queried_dois)
        meta_pdf_url = inspection.get("meta_pdf_url")
        if isinstance(meta_pdf_url, str) and meta_pdf_url:
            append_pdf_candidate(candidates, seen_candidates, PdfCandidate(meta_pdf_url, "page_meta_pdf", f"Found citation_pdf_url while inspecting {source_label}", doi))
        for candidate_url in inspection.get("anchor_pdf_urls", []):
            append_pdf_candidate(candidates, seen_candidates, PdfCandidate(candidate_url, "page_anchor_pdf", f"Found PDF-like link while inspecting {source_label}", doi))

    priority = {
        "scholar_rhs_pdf": 0,
        "openalex_best_oa_pdf": 1,
        "openalex_location_pdf": 2,
        "page_meta_pdf": 3,
        "page_direct_pdf": 4,
        "page_anchor_pdf": 5,
        "landing_pdf": 6,
    }
    ranked = sorted(candidates, key=lambda candidate: (priority.get(candidate.source, 100), len(candidate.url), candidate.url))
    return ranked[:max_candidates], logs


def extend_from_openalex(
    session: Any,
    doi: str,
    candidates: list[PdfCandidate],
    seen_candidates: set[str],
    inspect_queue: list[tuple[str, str]],
    seen_inspect_urls: set[str],
    logs: list[dict[str, Any]],
    queried_dois: set[str],
) -> None:
    if not doi or doi in queried_dois:
        return
    queried_dois.add(doi)
    log = {
        "stage": "openalex_lookup",
        "doi": doi,
        "requested_url": OPENALEX_WORK_URL.format(doi=quote(doi, safe="/")),
        "http_status": None,
        "error": None,
        "best_oa_pdf_url": None,
        "location_pdf_urls": [],
        "location_landing_urls": [],
    }
    try:
        response = session.get(log["requested_url"], timeout=DEFAULT_TIMEOUT)
        log["http_status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        locations = []
        best_location = payload.get("best_oa_location") or {}
        if isinstance(best_location, dict):
            locations.append(("openalex_best_oa", best_location))
        for location in payload.get("locations") or []:
            if isinstance(location, dict):
                locations.append(("openalex_location", location))
        for source, location in locations:
            pdf_url = str(location.get("pdf_url") or "").strip()
            landing_url = str(location.get("landing_page_url") or "").strip()
            if pdf_url:
                log["location_pdf_urls"].append(pdf_url)
                append_pdf_candidate(candidates, seen_candidates, PdfCandidate(pdf_url, f"{source}_pdf", f"OA PDF from {source}", doi))
            if landing_url:
                log["location_landing_urls"].append(landing_url)
                queue_inspect_url(inspect_queue, seen_inspect_urls, landing_url, f"{source}_landing")
    except Exception as exc:
        log["error"] = f"{type(exc).__name__}: {exc}"
    logs.append(log)


def inspect_html_or_pdf_endpoint(session: Any, url: str, source_label: str) -> dict[str, Any]:
    log = {
        "source_label": source_label,
        "requested_url": url,
        "final_url": None,
        "http_status": None,
        "content_type": None,
        "error": None,
        "verified_pdf_url": None,
        "meta_pdf_url": None,
        "doi": None,
        "anchor_pdf_urls": [],
    }
    try:
        response = session.get(url, timeout=DEFAULT_TIMEOUT)
        log["final_url"] = response.url
        log["http_status"] = response.status_code
        log["content_type"] = response.headers.get("content-type")
        response.raise_for_status()
        if is_verified_pdf_response(response):
            log["verified_pdf_url"] = response.url
            return {"log": log, "verified_pdf_url": response.url, "meta_pdf_url": None, "doi": extract_doi_from_text(" ".join(filter(None, [url, response.url]))), "anchor_pdf_urls": []}

        soup = make_soup(response.text)
        meta_pdf_url = extract_meta_pdf_url(soup, response.url)
        doi = extract_doi_from_html(soup, response.text, response.url)
        anchor_pdf_urls = extract_pdf_links_from_html(soup, response.url)
        log["meta_pdf_url"] = meta_pdf_url
        log["doi"] = doi
        log["anchor_pdf_urls"] = anchor_pdf_urls
        return {"log": log, "verified_pdf_url": None, "meta_pdf_url": meta_pdf_url, "doi": doi, "anchor_pdf_urls": anchor_pdf_urls}
    except Exception as exc:
        log["error"] = f"{type(exc).__name__}: {exc}"
        return {"log": log, "verified_pdf_url": None, "meta_pdf_url": None, "doi": None, "anchor_pdf_urls": []}


def download_pdf(session: Any, pdf_url: str) -> tuple[bytes | None, dict[str, Any]]:
    meta = {
        "requested_url": pdf_url,
        "final_url": None,
        "http_status": None,
        "content_type": None,
        "content_length": None,
        "downloaded": False,
        "is_pdf_verified": False,
        "failure_stage": None,
        "error": None,
    }
    try:
        response = session.get(pdf_url, timeout=PDF_TIMEOUT)
        meta["final_url"] = response.url
        meta["http_status"] = response.status_code
        meta["content_type"] = response.headers.get("content-type")
        meta["content_length"] = response.headers.get("content-length")
        response.raise_for_status()
        if not is_verified_pdf_response(response):
            meta["failure_stage"] = "html_instead_of_pdf"
            meta["error"] = "Response did not verify as a PDF file"
            return None, meta
        meta["downloaded"] = True
        meta["is_pdf_verified"] = True
        return response.content, meta
    except Exception as exc:
        meta["failure_stage"] = meta["failure_stage"] or "request_failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta


def is_verified_pdf_response(response: Any) -> bool:
    content_type = str(response.headers.get("content-type") or "").lower()
    if "application/pdf" in content_type:
        return True
    content = response.content or b""
    return content[:4] == b"%PDF"


def extract_pages_from_pdf(pdf_bytes: bytes) -> tuple[list[PageText], dict[str, Any]]:
    meta = {"parser": "pymupdf", "page_count": 0, "error": None}
    try:
        import fitz
    except ImportError:
        meta["error"] = "PyMuPDF (fitz) is not installed"
        return [], meta
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta["page_count"] = document.page_count
        pages = []
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            pages.append(PageText(page_number=page_number + 1, text=extract_page_text(page)))
        return pages, meta
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return [], meta


def build_paper_chunks(pages: list[PageText]) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    full_text = "\n\n".join(page.text for page in pages if page.text.strip()).strip()
    meta = {"raw_text_length": len(full_text), "used_keyword_fallback": False, "used_section_scan": False}
    chunks = {"title": "", "abstract": "", "intro_chunks": [], "method_chunks": [], "complexity_chunks": [], "conclusion_chunks": []}
    if not full_text:
        meta["error"] = "empty_pdf_text"
        return "", chunks, [], meta
    first_page_lines = [line.strip() for line in pages[0].text.splitlines() if line.strip()] if pages else []
    chunks["title"] = guess_title(first_page_lines)
    chunks["abstract"] = extract_section(full_text, ("abstract",), stop_terms=("introduction", "keywords"))
    intro_text = extract_section(full_text, ("introduction",), stop_terms=("method", "algorithm", "approach", "background", "preliminaries", "related work"))
    if intro_text:
        chunks["intro_chunks"] = paragraph_chunks(intro_text, max_chunks=3)
    method_text = extract_section(full_text, ("method", "algorithm", "approach", "formulation"), stop_terms=("complexity", "analysis", "experiments", "results", "discussion", "conclusion"))
    if method_text:
        chunks["method_chunks"] = paragraph_chunks(method_text, max_chunks=4)
    complexity_candidates = extract_pdf_complexity_candidates(pages, full_text)
    if complexity_candidates:
        chunks["complexity_chunks"] = [candidate["excerpt"] for candidate in complexity_candidates]
        meta["used_section_scan"] = True
    else:
        meta["used_keyword_fallback"] = True
    conclusion_text = extract_section(full_text, ("conclusion", "discussion", "applications", "limitations"), stop_terms=())
    if conclusion_text:
        chunks["conclusion_chunks"] = paragraph_chunks(conclusion_text, max_chunks=3)
    if not chunks["abstract"]:
        abstracts = keyword_paragraphs(full_text, keywords=("abstract",), limit=1)
        if abstracts:
            chunks["abstract"] = abstracts[0]
    meta["complexity_candidate_count"] = len(complexity_candidates)
    return full_text, chunks, complexity_candidates, meta


def chunks_has_substance(chunks: dict[str, Any]) -> bool:
    return any(
        [
            chunks.get("title"),
            chunks.get("abstract"),
            chunks.get("intro_chunks"),
            chunks.get("method_chunks"),
            chunks.get("complexity_chunks"),
            chunks.get("conclusion_chunks"),
        ]
    )


def prepare_networkx_materials(source: dict[str, Any]) -> list[dict[str, Any]]:
    materials = []
    url = str(source.get("networkx_url") or "")
    full_text = str(source.get("networkx_full_text") or "")
    if full_text:
        materials.append({"source_id": "networkx_full_text", "source_type": "networkx_full_text", "source_url": url, "text": full_text})
    for idx, chunk in enumerate(source.get("networkx_chunks_for_llm") or [], start=1):
        materials.append({"source_id": f"networkx_chunk_{idx}", "source_type": "networkx_chunk", "source_url": url, "text": chunk})
    for idx, candidate in enumerate(source.get("networkx_complexity_candidates") or [], start=1):
        excerpt = str(candidate.get("excerpt") or "")
        if excerpt:
            materials.append({"source_id": f"networkx_complexity_{idx}", "source_type": "networkx_chunk", "source_url": url, "text": excerpt, "evidence_type": candidate.get("evidence_type")})
    return materials


def prepare_paper_materials(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    materials = []
    for source in sources:
        chunks = source.get("chunks") or {}
        source_type = "paper_pdf_chunk" if source.get("source_type") == "paper_pdf" else "paper_abstract_chunk"
        for label in ("title", "abstract"):
            text = str(chunks.get(label) or "")
            if text:
                materials.append({"source_id": f"{source['source_id']}_{label}", "source_type": source_type, "source_url": source["source_url"], "text": text})
        for group in ("intro_chunks", "method_chunks", "complexity_chunks", "conclusion_chunks"):
            for idx, text in enumerate(chunks.get(group) or [], start=1):
                if text:
                    materials.append({"source_id": f"{source['source_id']}_{group}_{idx}", "source_type": source_type, "source_url": source["source_url"], "text": text})
    return materials


def prepare_wikipedia_materials(source: dict[str, Any]) -> list[dict[str, Any]]:
    materials = []
    page_url = str(source.get("page_url") or "")
    for source_id, source_type, key in (
        ("wikipedia_lead", "wikipedia_lead", "lead_text"),
        ("wikipedia_infobox", "wikipedia_infobox", "infobox_text"),
    ):
        text = str(source.get(key) or "").strip()
        if text:
            materials.append({"source_id": source_id, "source_type": source_type, "source_url": page_url, "text": text})
    for idx, section in enumerate(source.get("sections") or [], start=1):
        text = str(section.get("text") or "").strip()
        if text:
            materials.append({"source_id": f"wikipedia_section_{idx}", "source_type": "wikipedia_section", "source_url": page_url, "text": text, "section_title": section.get("section_title")})
    return materials


def build_evidence_prompt(algorithm_name: str, networkx_materials: list[dict[str, Any]], paper_materials: list[dict[str, Any]], wikipedia_materials: list[dict[str, Any]], abstract_only: bool) -> str:
    rules = "Only abstract-page materials are available for papers. Be conservative." if abstract_only else "Use only the provided text. Treat NetworkX, paper, and Wikipedia excerpts as parallel evidence sources."
    return f"""
You are an evidence organizer for a graph algorithm knowledge base.

Return strict JSON only. Do not output markdown. Do not output code fences.

Required JSON schema:
{{
  "evidence_records": [
    {{
      "source_id": "string",
      "source_type": "networkx_full_text|networkx_chunk|paper_pdf_chunk|paper_abstract_chunk|wikipedia_lead|wikipedia_infobox|wikipedia_section",
      "source_url": "string",
      "excerpt": "supporting excerpt",
      "field_mappings": [
        {{
          "field": "Deployment_method.support_engine|Deployment_method.input_schema|Deployment_method.output_schema|Deployment_method.graph_type|Application_scenario|Principles.description|Principles.time_complexity|solvable_questions_context",
          "content": "grounded content for this field only"
        }}
      ]
    }}
  ]
}}

Rules:
- {rules}
- Do not invent facts.
- Each excerpt must be 280 characters or fewer.
- Each field_mappings.content value must be 220 characters or fewer.
- For Principles.time_complexity, use explicit Big-O or explicit iterative-runtime evidence only.
- For Deployment_method fields, emit content only when implementation/input/output/graph facts are explicit.
- Keep field content concise and directly usable downstream.

Algorithm name:
{algorithm_name}

NetworkX materials:
{json.dumps(networkx_materials, ensure_ascii=False, indent=2)}

Paper materials:
{json.dumps(paper_materials, ensure_ascii=False, indent=2)}

Wikipedia materials:
{json.dumps(wikipedia_materials, ensure_ascii=False, indent=2)}
""".strip()


def build_entry_json_prompt(normalized_id: str, evidence_records: list[EvidenceRecord], task_types: list[dict[str, Any]]) -> str:
    task_type_options = [
        {
            "id": item.get("id"),
            "task_type": item.get("task_type"),
            "description": item.get("description"),
        }
        for item in task_types
        if isinstance(item, dict)
    ]
    return f"""
You are generating one algorithm entry for a graph algorithm knowledge base.

Use only the structured evidence below.

Return strict JSON only. Do not output markdown. Do not output code fences.

Required JSON schema:
{{
  "id": "{normalized_id}",
  "task_type_id": "existing category id or new category id",
  "new_task_type": null,
  "Deployment_method": {{
    "support_engine": "value or unknown",
    "input_schema": {{
      "parameters": {{
        "parameter_name": {{
          "type": "type",
          "description": "description",
          "required": true,
          "default": "default value when supported"
        }}
      }}
    }},
    "output_schema": {{
      "type": "output type",
      "description": "output description"
    }},
    "graph_type": {{
      "directed": "true|false|both|unknown",
      "multigraph": "true|false|unknown",
      "weighted": "true|false|unknown",
      "heterogeneous": "true|false|unknown"
    }}
  }},
  "Application_scenario": "string or unknown",
  "Principles": {{
    "description": "string or unknown",
    "time_complexity": "string or unknown"
  }},
  "solvable_questions": ["question"]
}}

Rules:
- Use English for YAML values.
- Do not invent unsupported facts.
- If a field has no evidence, use unknown.
- Do not put all inputs into one string. Use Deployment_method.input_schema.parameters as an object.
- Do not put output_schema into one string. Use Deployment_method.output_schema.type and description.
- For graph_type, use an object when graph constraints are supported.
- Generate 4-8 solvable questions when there is enough support.
- Return one JSON object, not a list.
- Choose task_type_id from the existing task type list when any category fits.
- Only create a new category when none of the existing task types fit.
- If creating a new category, set new_task_type to {{"id": "Category Name", "task_type": "Category Name", "description": "short description"}}.
- If using an existing category, set new_task_type to null.

Existing task types:
{json.dumps(task_type_options, ensure_ascii=False, indent=2)}

Structured evidence:
{json.dumps([evidence_to_dict(record) for record in evidence_records], ensure_ascii=False, indent=2)}
""".strip()


def parse_evidence_response(raw_text: str) -> list[EvidenceRecord]:
    payload = extract_json_object(raw_text)
    records = []
    for item in payload.get("evidence_records", []):
        mappings = [
            FieldMapping(field=str(mapping.get("field", "")).strip(), content=str(mapping.get("content", "")).strip())
            for mapping in item.get("field_mappings", [])
            if str(mapping.get("field", "")).strip() and str(mapping.get("content", "")).strip()
        ]
        if mappings:
            records.append(
                EvidenceRecord(
                    source_id=str(item.get("source_id", "")).strip(),
                    source_type=str(item.get("source_type", "")).strip(),
                    source_url=str(item.get("source_url", "")).strip(),
                    excerpt=str(item.get("excerpt", "")).strip(),
                    field_mappings=mappings,
                )
            )
    return records


def parse_generated_entry_json(raw_text: str) -> dict[str, Any]:
    payload = extract_json_object(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Generated entry response must be a JSON object")
    return payload


def validate_generated_yaml(yaml_text: str, evidence_records: list[EvidenceRecord]) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(yaml_text)
        yaml_valid = isinstance(parsed, list) and bool(parsed) and isinstance(parsed[0], dict)
        parse_error = None
    except Exception as exc:
        parsed = None
        yaml_valid = False
        parse_error = f"{type(exc).__name__}: {exc}"

    evidence_fields = {mapping.field for record in evidence_records for mapping in record.field_mappings}
    uncovered_fields = []
    if yaml_valid:
        entry = parsed[0]
        for field_name in ("Deployment_method", "Application_scenario", "Principles", "solvable_questions"):
            if has_real_value(entry.get(field_name)) and not any(field.startswith(field_name) for field in evidence_fields):
                uncovered_fields.append(field_name)
    return {"yaml_valid": yaml_valid, "yaml_parse_error": parse_error, "uncovered_fields": uncovered_fields}


def parse_generated_yaml_entry(yaml_text: str) -> dict[str, Any]:
    payload = yaml.safe_load(yaml_text)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError("Generated YAML must contain a non-empty list with one algorithm entry")
    return payload[0]


def normalize_generated_entry(entry: dict[str, Any], normalized_id: str) -> None:
    entry["id"] = normalized_id
    entry.setdefault("task_type_id", "Unknown")
    if entry.get("new_task_type") in ("", "unknown", "Unknown"):
        entry["new_task_type"] = None
    if not isinstance(entry.get("Deployment_method"), dict):
        entry["Deployment_method"] = {"support_engine": entry.get("Deployment_method") or "unknown", "input_schema": {}, "output_schema": {}, "graph_type": {}}
    if not isinstance(entry.get("Principles"), dict):
        entry["Principles"] = {"description": entry.get("Principles") or "unknown", "time_complexity": "unknown"}
    if not isinstance(entry.get("solvable_questions"), list):
        entry["solvable_questions"] = []


def append_algorithm_entry(entry: dict[str, Any], overwrite: bool = False) -> None:
    entry = dict(entry)
    entry.pop("new_task_type", None)
    if ALGORITHMS_NEW_PATH.exists():
        with ALGORITHMS_NEW_PATH.open("r", encoding="utf-8") as file:
            algorithms = yaml.safe_load(file) or []
    else:
        algorithms = []
    if not isinstance(algorithms, list):
        raise ValueError(f"Expected a list in {ALGORITHMS_NEW_PATH}")

    normalized_id = str(entry.get("id") or "").strip()
    if overwrite:
        algorithms = [item for item in algorithms if not (isinstance(item, dict) and item.get("id") == normalized_id)]
    elif any(isinstance(item, dict) and item.get("id") == normalized_id for item in algorithms):
        raise ValueError(f"Algorithm already exists: {normalized_id}")

    algorithms.append(entry)
    with ALGORITHMS_NEW_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(algorithms, file, allow_unicode=True, sort_keys=False, width=120)


def load_task_types() -> list[dict[str, Any]]:
    with TASK_TYPES_PATH.open("r", encoding="utf-8") as file:
        task_types = yaml.safe_load(file) or []
    if not isinstance(task_types, list):
        raise ValueError(f"Expected a list in {TASK_TYPES_PATH}")
    return [item for item in task_types if isinstance(item, dict)]


def update_task_type_mapping(entry: dict[str, Any], overwrite: bool = False) -> None:
    task_types = load_task_types()
    algorithm_id = str(entry.get("id") or "").strip()
    task_type_id = str(entry.get("task_type_id") or "").strip()
    new_task_type = entry.get("new_task_type")
    if not algorithm_id or not task_type_id:
        return

    target = None
    for item in task_types:
        if str(item.get("id") or "").strip() == task_type_id:
            target = item
            break

    if target is None:
        if isinstance(new_task_type, dict):
            new_item = {
                "id": str(new_task_type.get("id") or task_type_id).strip(),
                "task_type": str(new_task_type.get("task_type") or task_type_id).strip(),
                "description": str(new_task_type.get("description") or f"Algorithms related to {task_type_id}.").strip(),
                "algorithm": [],
            }
        else:
            new_item = {
                "id": task_type_id,
                "task_type": task_type_id,
                "description": f"Algorithms related to {task_type_id}.",
                "algorithm": [],
            }
        append_new_task_type_text(new_item, algorithm_id)
        return

    algorithms = target.get("algorithm")
    if not isinstance(algorithms, list):
        algorithms = []
    if overwrite and algorithm_id in algorithms:
        return
    if algorithm_id not in algorithms:
        append_algorithm_to_task_type_text(task_type_id, algorithm_id)


def append_algorithm_to_task_type_text(task_type_id: str, algorithm_id: str) -> None:
    lines = TASK_TYPES_PATH.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == f"- id: {task_type_id}":
            start = index
            break
    if start is None:
        raise ValueError(f"Task type not found: {task_type_id}")

    algorithm_line = None
    next_section = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("- id: "):
            next_section = index
            break
        if lines[index].strip() == "algorithm:":
            algorithm_line = index
    if algorithm_line is None:
        insert_at = next_section
        lines.insert(insert_at, "  algorithm:")
        lines.insert(insert_at + 1, f"    - {algorithm_id}")
    else:
        insert_at = next_section
        while insert_at > algorithm_line + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, f"    - {algorithm_id}")

    TASK_TYPES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_new_task_type_text(task_type: dict[str, Any], algorithm_id: str) -> None:
    content = TASK_TYPES_PATH.read_text(encoding="utf-8").rstrip()
    block = (
        f"- id: {task_type['id']}\n"
        f"  task_type: \"{task_type['task_type']}\"\n"
        f"  description: {task_type['description']}\n"
        "  algorithm:\n"
        f"    - {algorithm_id}\n"
    )
    TASK_TYPES_PATH.write_text(f"{content}\n\n{block}", encoding="utf-8")


def make_soup(html: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 package is required for algorithm creation") from exc
    return BeautifulSoup(html or "", "html.parser")


def extract_main_text(html: str) -> tuple[str, list[str]]:
    soup = make_soup(html)
    for unwanted in soup.select("script, style, nav, footer, header, aside"):
        unwanted.decompose()
    main = soup.select_one("main") or soup.select_one("div.body") or soup.body or soup
    text = clean_text(main.get_text("\n", strip=True))
    return text[:50000], [f"main_text_chars={min(len(text), 50000)}"]


def extract_wikipedia_content(html: str) -> tuple[str, str, list[dict[str, str]], list[str]]:
    soup = make_soup(html)
    for unwanted in soup.select("style, script, sup.reference, span.mw-editsection, div.reflist, ol.references, table.navbox, div.navbox"):
        unwanted.decompose()
    infobox = soup.select_one("table.infobox")
    infobox_text = clean_text(infobox.get_text("\n", strip=True))[:3000] if infobox else ""
    if infobox:
        infobox.decompose()
    root = soup.select_one("div.mw-parser-output") or soup
    lead_parts = []
    sections = []
    current_title = None
    current_parts = []
    in_lead = True
    stop_titles = {"references", "external links", "further reading", "notes", "bibliography"}
    for node in root.find_all({"h2", "h3", "p", "ul", "ol"}, recursive=True):
        if node.name in {"h2", "h3"}:
            heading = clean_heading(node.get_text(" ", strip=True))
            if heading.lower() in stop_titles:
                break
            if current_title and current_parts:
                sections.append({"section_title": current_title, "text": clean_text("\n".join(current_parts))[:3500]})
            current_title = heading
            current_parts = []
            in_lead = False
            continue
        text = clean_text(node.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        if in_lead:
            lead_parts.append(text)
        else:
            current_parts.append(text)
    if current_title and current_parts:
        sections.append({"section_title": current_title, "text": clean_text("\n".join(current_parts))[:3500]})
    return clean_text("\n\n".join(lead_parts))[:5000], infobox_text, sections[:6], [f"section_count={len(sections)}"]


def extract_page_title(soup: Any, title_hint: str) -> str:
    meta_title = soup.find("meta", attrs={"name": "citation_title"})
    if meta_title and meta_title.get("content"):
        return str(meta_title["content"]).strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return title_hint.strip()


def extract_abstract(soup: Any) -> str:
    for attrs in ({"name": "citation_abstract"}, {"name": "description"}, {"property": "og:description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content") and len(str(tag["content"])) > 20:
            return str(tag["content"]).strip()
    for node in soup.select("[class*='abstract'], [id*='abstract'], section.abstract, div.abstract, p.abstract"):
        text = clean_text(node.get_text(" ", strip=True))
        if len(text) > 20:
            return text
    return ""


def extract_visible_text(soup: Any) -> str:
    body = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    for unwanted in body.select("nav, header, footer, script, style, aside, form"):
        unwanted.decompose()
    return clean_text(body.get_text("\n", strip=True))[:20000]


def extract_page_text(page: Any) -> str:
    blocks = page.get_text("blocks")
    rows = []
    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, _, _, text = block[:5]
        cleaned = (text or "").strip()
        if cleaned:
            rows.append((float(y0), float(x0), cleaned))
    if rows:
        rows.sort(key=lambda item: (item[0], item[1]))
        text = "\n".join(row[2] for row in rows)
    else:
        text = page.get_text("text")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def guess_title(lines: list[str]) -> str:
    for line in lines[:12]:
        if len(line) >= 20 and not re.match(r"^(abstract|keywords?)\b", line.lower()):
            return line
    return lines[0] if lines else ""


def extract_section(text: str, start_terms: tuple[str, ...], stop_terms: tuple[str, ...]) -> str:
    lower = text.lower()
    start_index = -1
    for term in start_terms:
        match = re.search(rf"\b{re.escape(term)}\b", lower, flags=re.IGNORECASE)
        if match:
            start_index = match.start()
            break
    if start_index < 0:
        return ""
    end_index = len(text)
    for stop_term in stop_terms:
        match = re.search(rf"\b{re.escape(stop_term)}\b", lower[start_index + 1:], flags=re.IGNORECASE)
        if match:
            end_index = min(end_index, start_index + 1 + match.start())
    return text[start_index:end_index].strip()


def paragraph_chunks(text: str, max_chunks: int) -> list[str]:
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if len(para.strip()) > 40]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
    return paragraphs[:max_chunks]


def keyword_paragraphs(text: str, keywords: tuple[str, ...], limit: int) -> list[str]:
    results = []
    for para in [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]:
        if any(keyword in para.lower() for keyword in keywords) and len(para) > 30:
            results.append(para)
        if len(results) >= limit:
            break
    return results


def extract_pdf_complexity_candidates(pages: list[PageText], full_text: str) -> list[dict[str, Any]]:
    section_terms = ("computational complexity", "time complexity", "complexity", "analysis", "runtime", "convergence", "implementation")
    stop_terms = ("experiments", "results", "evaluation", "discussion", "conclusion", "references", "appendix")
    candidates = []
    seen = set()
    for section_term in section_terms:
        section_text = extract_section(full_text, (section_term,), stop_terms=stop_terms)
        if not section_text:
            continue
        for paragraph in paragraph_chunks(section_text, max_chunks=4):
            if not contains_complexity_signal(paragraph):
                continue
            append_complexity_candidate(
                candidates,
                seen,
                {
                    "excerpt": paragraph,
                    "evidence_type": classify_complexity_evidence(paragraph),
                    "page_number": guess_page_number(pages, paragraph),
                    "section_name": section_term,
                },
            )
    for page in pages:
        sentences = split_sentences(page.text)
        for index, sentence in enumerate(sentences):
            if not contains_complexity_signal(sentence):
                continue
            context = " ".join(sentences[max(0, index - 1): min(len(sentences), index + 2)]).strip()
            if len(context) < 30:
                continue
            append_complexity_candidate(
                candidates,
                seen,
                {
                    "excerpt": context,
                    "evidence_type": classify_complexity_evidence(context),
                    "page_number": page.page_number,
                    "section_name": guess_section_name(page.text, sentence),
                },
            )
    return candidates[:6]


def append_complexity_candidate(candidates: list[dict[str, Any]], seen: set[str], candidate: dict[str, Any]) -> None:
    key = re.sub(r"\s+", " ", str(candidate.get("excerpt", "")).lower()).strip()
    if key and key not in seen:
        seen.add(key)
        candidates.append(candidate)


def contains_complexity_signal(text: str) -> bool:
    terms = ("o(", "time complexity", "running time", "computational cost", "cost per iteration", "each iteration", "per iteration", "number of iterations", "iterations", "converges", "convergence", "until tolerance", "tolerance", "requires")
    return any(term in (text or "").lower() for term in terms)


def guess_page_number(pages: list[PageText], snippet: str) -> int | None:
    normalized = normalize_for_match(snippet)
    for page in pages:
        if normalized[:160] and normalized[:160] in normalize_for_match(page.text):
            return page.page_number
    return None


def guess_section_name(page_text: str, sentence: str) -> str | None:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    normalized_sentence = normalize_for_match(sentence)
    for index, line in enumerate(lines):
        if normalized_sentence[:120] and normalized_sentence[:120] in normalize_for_match(line):
            for candidate in reversed(lines[max(0, index - 8): index]):
                if looks_like_heading(candidate):
                    return candidate
            break
    return None


def looks_like_heading(line: str) -> bool:
    text = (line or "").strip()
    if len(text) < 3 or len(text) > 120:
        return False
    lowered = text.lower()
    return bool(re.match(r"^(\d+(\.\d+)*)?\s*(complexity|analysis|runtime|convergence|implementation)\b", lowered) or (text == text.upper() and len(text.split()) <= 8))


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def extract_complexity_candidates(text: str) -> list[dict[str, str]]:
    terms = ("o(", "complexity", "time complexity", "running time", "computational cost", "per iteration", "iterations", "convergence", "tolerance")
    sentences = split_sentences(text)
    candidates = []
    seen = set()
    for index, sentence in enumerate(sentences):
        if not any(term in sentence.lower() for term in terms):
            continue
        context = " ".join(sentences[max(0, index - 1): min(len(sentences), index + 2)]).strip()
        key = re.sub(r"\s+", " ", context.lower())
        if len(context) < 30 or key in seen:
            continue
        seen.add(key)
        candidates.append({"excerpt": context, "evidence_type": classify_complexity_evidence(context)})
        if len(candidates) >= 6:
            break
    return candidates


def keyword_sentences(text: str, keywords: tuple[str, ...]) -> list[str]:
    matches = []
    for sentence in split_sentences(text):
        if any(keyword in sentence.lower() for keyword in keywords) and len(sentence.strip()) > 20:
            matches.append(sentence.strip())
        if len(matches) >= 3:
            break
    return matches


def extract_doi_from_text(text: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text or "", flags=re.IGNORECASE)
    return match.group(0).rstrip(").,;]") if match else None


def extract_meta_pdf_url(soup: Any, base_url: str) -> str | None:
    for attrs in ({"name": "citation_pdf_url"}, {"property": "citation_pdf_url"}, {"name": "dc.identifier.pdf"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return urljoin(base_url, str(tag["content"]).strip())
    return None


def extract_doi_from_html(soup: Any, html_text: str, base_url: str) -> str | None:
    for attrs in ({"name": "citation_doi"}, {"name": "dc.identifier"}, {"property": "og:url"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            doi = extract_doi_from_text(str(tag["content"]))
            if doi:
                return doi
    return extract_doi_from_text(" ".join(filter(None, [html_text[:4000], base_url])))


def extract_pdf_links_from_html(soup: Any, base_url: str) -> list[str]:
    discovered = []
    seen = set()
    for selector, attribute in (("a[href]", "href"), ("iframe[src]", "src"), ("embed[src]", "src"), ("object[data]", "data")):
        for node in soup.select(selector):
            raw_url = str(node.get(attribute, "")).strip()
            if not raw_url or raw_url.startswith(("javascript:", "mailto:", "#")):
                continue
            resolved = urljoin(base_url, raw_url)
            if not looks_like_pdf_candidate_link(resolved, node.get_text(" ", strip=True)):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered[:10]


def append_pdf_candidate(candidates: list[PdfCandidate], seen: set[str], candidate: PdfCandidate) -> None:
    key = candidate.url.strip()
    if key and key not in seen:
        seen.add(key)
        candidates.append(candidate)


def queue_inspect_url(queue: list[tuple[str, str]], seen_urls: set[str], url: str, source_label: str) -> None:
    normalized = (url or "").strip()
    if normalized and normalized not in seen_urls:
        seen_urls.add(normalized)
        queue.append((normalized, source_label))


def looks_like_pdf_candidate_link(url: str, text: str = "") -> bool:
    return looks_like_pdf_url(url) or "pdf" in (text or "").lower()


def looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    lower = (url or "").lower()
    return parsed.path.lower().endswith(".pdf") or any(hint in lower for hint in PDF_HINTS)


def select_top_papers(results: list[dict[str, Any]], reference_terms: list[str], limit: int = 3) -> list[PaperCandidate]:
    ref_tokens = normalize_tokens(" ".join(reference_terms))
    scored = []
    for result in results:
        for paper in result.get("papers", []):
            scored.append((len(normalize_tokens(paper.title) & ref_tokens), paper))
    scored.sort(key=lambda item: -item[0])
    return [paper for _, paper in scored[:limit]]


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    content = (text or "").strip()
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]
    chunks = []
    start = 0
    while start < len(content):
        end = min(len(content), start + max_chars)
        split = content.rfind("\n", start, end)
        if split <= start + int(max_chars * 0.5):
            split = content.rfind(". ", start, end)
        if split <= start:
            split = end
        chunks.append(content[start:split].strip())
        start = split
    return [chunk for chunk in chunks if chunk]


def prepare_variant(value: str) -> dict[str, Any]:
    normalized = normalize_text_token(value)
    slug = normalized.replace(" ", "_")
    return {"raw": value, "normalized": normalized, "slug": slug, "tokens": normalized.split()}


def candidate_score(filename: str, docname: str, title: str, variants: list[dict[str, Any]]) -> tuple[float, list[str]]:
    haystacks = {"filename": normalize_text_token(filename), "docname": normalize_text_token(docname), "title": normalize_text_token(title)}
    score = 0.0
    matched = []
    for variant in variants:
        normalized = str(variant["normalized"])
        slug = str(variant["slug"])
        tokens = set(variant["tokens"])
        variant_score = 0.0
        if slug and slug in haystacks["filename"]:
            variant_score += 10
        if normalized and normalized in haystacks["title"]:
            variant_score += 8
        if normalized and normalized in haystacks["docname"]:
            variant_score += 8
        if tokens:
            for haystack in haystacks.values():
                overlap = len(tokens & set(haystack.replace("/", " ").split()))
                variant_score += overlap * 1.5
        if variant_score:
            matched.append(str(variant["raw"]))
            score += variant_score
    return score, matched


def basename(filename: str) -> str:
    return Path(filename).stem


def path_depth(path: str) -> int:
    return len([part for part in path.split("/") if part])


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def clean_heading(text: str) -> str:
    return re.sub(r"\[edit\]$", "", clean_text(text)).strip()


def normalize_text_token(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text or "").lower()).strip()


def normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def text_candidate_score(query: str, title: str, snippet: str) -> float:
    query_tokens = normalize_tokens(query)
    title_tokens = normalize_tokens(title)
    snippet_tokens = normalize_tokens(snippet)
    return len(query_tokens & title_tokens) * 3 + len(query_tokens & snippet_tokens)


def build_wikipedia_page_url(title: str) -> str:
    return WIKIPEDIA_PAGE_BASE + title.replace(" ", "_")


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text or "") if sentence.strip()]


def classify_complexity_evidence(text: str) -> str:
    lower = (text or "").lower()
    if "o(" in lower:
        return "explicit_big_o"
    if any(keyword in lower for keyword in ("per iteration", "iterations", "convergence", "tolerance")):
        return "explicit_iterative_runtime"
    return "explicit_runtime"


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def evidence_to_dict(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "source_id": record.source_id,
        "source_type": record.source_type,
        "source_url": record.source_url,
        "excerpt": record.excerpt,
        "field_mappings": [asdict(mapping) for mapping in record.field_mappings],
    }


def has_real_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() != "unknown"
    if isinstance(value, list):
        return any(has_real_value(item) for item in value)
    if isinstance(value, dict):
        return any(has_real_value(item) for item in value.values())
    return True
