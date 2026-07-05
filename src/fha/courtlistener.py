"""
CourtListener REST v4 client: search(), get_opinion(), get_cluster(),
harvest(). Token optional (config/.cl_token or COURTLISTENER_TOKEN); backs
off on HTTP 429.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger("fha.courtlistener")

import requests

from . import config


@dataclass
class CaseRecord:
    """One federal decision, flattened for the downstream pipeline."""
    cluster_id: int
    case_name: str = ""
    court_id: str = ""
    circuit: str | None = None
    court_level: str | None = None
    court_jurisdiction: str = ""
    date_filed: str | None = None
    year: int | None = None
    docket_number: str = ""
    citations: list[str] = field(default_factory=list)
    cite_count: int = 0
    judges: str = ""
    panel_names: list[str] = field(default_factory=list)
    precedential_status: str = ""
    nature_of_suit: str = ""        # federal civil cover-sheet code (443 = housing)
    posture: str = ""              # procedural posture
    procedural_history: str = ""
    opinion_ids: list[int] = field(default_factory=list)
    text: str = ""
    text_len: int = 0
    text_source: str = "none"      # "full" (token) | "snippet" | "none"
    source: str = "courtlistener"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class CourtListenerClient:
    def __init__(self, token: str | None = None, base_url: str | None = None,
                 polite_delay_s: float | None = None):
        self.token = config.courtlistener_token(token)
        self.base = (base_url or config.SETTINGS.cl_base_url).rstrip("/")
        # authenticated rate limits are far higher, so a shorter courtesy delay
        # is fine; stay slow when anonymous.
        if polite_delay_s is not None:
            self.delay = polite_delay_s
        elif self.token:
            self.delay = 0.25
        else:
            self.delay = config.SETTINGS.polite_delay_s
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "fha-courts-research/1.0"})
        if self.token:
            self.s.headers["Authorization"] = f"Token {self.token}"
        self.authenticated = bool(self.token)
        self.throttle_hits = 0          # count of 429 back-offs (observability)
        # size the connection pool for parallel harvesting (default maxsize=10)
        adapter = requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)

    # -- low level -----------------------------------------------------------
    max_retries = 2          # keep low: each 429 retry also costs quota

    def _get(self, url: str, params: dict | None = None, _tries: int = 0) -> dict:
        """GET with polite delay and bounded 429 back-off.

        On a *sustained* throttle we raise HTTPError rather than retrying
        forever -- retries consume the same rolling quota, so grinding makes the
        exhaustion worse. The caller saves progress and resumes when quota frees.
        """
        time.sleep(self.delay)
        r = self.s.get(url, params=params, timeout=45)
        if r.status_code == 429:
            self.throttle_hits += 1
            if _tries < self.max_retries:
                wait = min(float(r.headers.get("Retry-After", 2 ** _tries)), 20.0)
                time.sleep(wait)
                return self._get(url, params, _tries + 1)
        r.raise_for_status()
        return r.json()

    # -- endpoints -----------------------------------------------------------
    def search(self, query: str, *, court: str | None = None,
               filed_after: str | None = None, filed_before: str | None = None,
               max_pages: int = 5, order_by: str = "dateFiled asc"
               ) -> Iterator[dict]:
        """Yield opinion-search result rows (cursor pagination)."""
        params = {"type": "o", "q": query, "order_by": order_by}
        if court:
            params["court"] = court
        if filed_after:
            params["filed_after"] = filed_after
        if filed_before:
            params["filed_before"] = filed_before
        url = f"{self.base}/search/"
        page = 0
        while url and page < max_pages:
            data = self._get(url, params if page == 0 else None)
            for row in data.get("results", []):
                yield row
            url = data.get("next")
            page += 1

    # Limit the opinion payload to the text we actually use. The endpoint
    # otherwise returns plain_text AND html AND html_with_citations AND
    # xml_harvard -- up to ~4x the opinion size per request. plain_text is
    # populated for the vast majority; html_with_citations is the one fallback.
    OPINION_FIELDS = "id,type,plain_text,html_with_citations,author_str"

    def get_opinion(self, opinion_id: int, fields: str | None = OPINION_FIELDS) -> dict:
        params = {"fields": fields} if fields else None
        return self._get(f"{self.base}/opinions/{opinion_id}/", params=params)

    def get_cluster(self, cluster_id: int) -> dict:
        return self._get(f"{self.base}/clusters/{cluster_id}/")

    @staticmethod
    def _opinion_text(op: dict) -> str:
        """Best available text field for one opinion."""
        for key in ("plain_text", "html_with_citations", "html", "xml_harvard"):
            val = op.get(key)
            if val and val.strip():
                # crude tag strip for html fallbacks
                if key != "plain_text":
                    import re
                    val = re.sub(r"<[^>]+>", " ", val)
                return val.strip()
        return ""

    # -- high level ----------------------------------------------------------
    def _row_to_record(self, row: dict, fetch_text: bool = True) -> CaseRecord:
        """Build a CaseRecord from one search row, optionally fetching full text.
        Thread-safe: the unit of work for parallel harvesting."""
        from .reference import court_to_circuit, court_level
        cid = int(row.get("cluster_id") or row.get("cluster"))
        court_id = (row.get("court_id") or "").lower()
        date_filed = row.get("dateFiled")
        year = None
        if date_filed and len(date_filed) >= 4 and date_filed[:4].isdigit():
            year = int(date_filed[:4])
        opinion_ids = [o["id"] for o in row.get("opinions", []) if o.get("id")]
        rec = CaseRecord(
            cluster_id=cid,
            case_name=row.get("caseName", "") or "",
            court_id=court_id,
            circuit=court_to_circuit(court_id),
            court_level=court_level(court_id),
            court_jurisdiction=row.get("court_jurisdiction", "") or "",
            date_filed=date_filed,
            year=year,
            docket_number=row.get("docketNumber", "") or "",
            citations=row.get("citation", []) or [],
            cite_count=row.get("citeCount", 0) or 0,
            judges=row.get("judge", "") or "",
            panel_names=row.get("panel_names", []) or [],
            precedential_status=row.get("status", "") or "",
            nature_of_suit=row.get("suitNature", "") or "",
            posture=row.get("posture", "") or "",
            procedural_history=row.get("procedural_history", "") or "",
            opinion_ids=opinion_ids,
        )
        if fetch_text:
            if not opinion_ids:
                try:
                    cluster = self.get_cluster(rec.cluster_id)
                    if not rec.nature_of_suit:
                        rec.nature_of_suit = cluster.get("nature_of_suit", "") or ""
                    for u in cluster.get("sub_opinions", []) or []:
                        try:
                            opinion_ids.append(int(u.rstrip("/").split("/")[-1]))
                        except (ValueError, AttributeError):
                            pass
                    rec.opinion_ids = opinion_ids
                except requests.HTTPError:
                    pass
            texts = []
            for oid in rec.opinion_ids:
                try:
                    texts.append(self._opinion_text(self.get_opinion(oid)))
                except requests.HTTPError:
                    continue
            full = "\n\n".join(t for t in texts if t)
            if full:
                rec.text, rec.text_source = full, "full"
            else:
                snips = [o.get("snippet", "") for o in row.get("opinions", [])]
                snip = "\n".join(s for s in snips if s)
                if snip:
                    rec.text, rec.text_source = snip, "snippet"
            rec.text_len = len(rec.text)
        return rec

    def harvest_parallel(self, queries, *, filed_after: str | None = None,
                         filed_before: str | None = None, max_pages: int = 5,
                         federal_only: bool = True, workers: int = 1,
                         fetch_text: bool = True, skip_ids: set | None = None,
                         max_cases: int | None = None, on_progress=None
                         ) -> Iterator[CaseRecord]:
        """Harvest across queries. fetch_text=False is a fast metadata+snippet
        sweep (search endpoint only -- lightly throttled). fetch_text=True adds
        full opinion text per case (opinions endpoint -- heavily throttled, so
        keep workers=1 and let the capped back-off self-pace)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .reference import court_level
        if isinstance(queries, str):
            queries = [queries]
        seen: set[int] = set(skip_ids or ())    # cross-run resume
        rows: list[dict] = []
        for q in queries:                       # Phase 1: metadata sweep (fast)
            for row in self.search(q, filed_after=filed_after,
                                   filed_before=filed_before, max_pages=max_pages):
                cid = row.get("cluster_id") or row.get("cluster")
                if cid is None or cid in seen:
                    continue
                court_id = (row.get("court_id") or "").lower()
                if federal_only and court_level(court_id) is None:
                    continue
                seen.add(cid)
                rows.append(row)
            if max_cases and len(rows) >= max_cases:
                break
        if max_cases:
            rows = rows[:max_cases]
        if on_progress:
            on_progress(0, len(rows))           # phase-1 total known
        def _safe(make, cid):
            try:
                return make()
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 429:
                    raise                       # let the caller stop gracefully
                log.warning("skipped cluster %s: HTTP %s", cid,
                            getattr(e.response, "status_code", "?"))
            except Exception as e:              # noqa: BLE001 - logged, not hidden
                log.warning("skipped cluster %s: %s", cid, e)
            return None

        if not fetch_text or workers <= 1:      # sequential (no/avoid throttle)
            for i, row in enumerate(rows, 1):
                rec = _safe(lambda: self._row_to_record(row, fetch_text),
                            row.get("cluster_id"))
                if on_progress:
                    on_progress(i, len(rows))
                if rec is not None:
                    yield rec
            return
        done = 0                                # Phase 2: parallel text fetch
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._row_to_record, row, True) for row in rows]
            for fut in as_completed(futs):
                done += 1
                rec = _safe(fut.result, None)
                if on_progress:
                    on_progress(done, len(rows))
                if rec is not None:
                    yield rec

    def harvest(self, query: str, *, court: str | None = None,
                filed_after: str | None = None, filed_before: str | None = None,
                max_pages: int = 5, fetch_text: bool = True,
                federal_only: bool = True,
                max_cases: int | None = None) -> Iterator[CaseRecord]:
        """Search + enrich into CaseRecords.

        federal_only drops any court not in the federal circuit map (the search
        index also contains state courts, which the FHA-federal scope excludes).
        Full opinion text requires a token; without one we keep the search
        snippet as text and tag text_source accordingly.
        """
        from .reference import court_to_circuit, court_level
        seen: set[int] = set()
        n = 0
        for row in self.search(query, court=court, filed_after=filed_after,
                               filed_before=filed_before, max_pages=max_pages):
            cid = row.get("cluster_id") or row.get("cluster")
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            court_id = (row.get("court_id") or "").lower()
            if federal_only and court_level(court_id) is None:
                continue   # state court or unmapped -> outside FHA-federal scope
            date_filed = row.get("dateFiled")
            year = None
            if date_filed and len(date_filed) >= 4 and date_filed[:4].isdigit():
                year = int(date_filed[:4])
            opinion_ids = [o["id"] for o in row.get("opinions", []) if o.get("id")]

            rec = CaseRecord(
                cluster_id=int(cid),
                case_name=row.get("caseName", "") or "",
                court_id=court_id,
                circuit=court_to_circuit(court_id),
                court_level=court_level(court_id),
                court_jurisdiction=row.get("court_jurisdiction", "") or "",
                date_filed=date_filed,
                year=year,
                docket_number=row.get("docketNumber", "") or "",
                citations=row.get("citation", []) or [],
                cite_count=row.get("citeCount", 0) or 0,
                judges=row.get("judge", "") or "",
                panel_names=row.get("panel_names", []) or [],
                precedential_status=row.get("status", "") or "",
                # `suitNature` ships on the search row -- no cluster fetch needed
                nature_of_suit=row.get("suitNature", "") or "",
                posture=row.get("posture", "") or "",
                procedural_history=row.get("procedural_history", "") or "",
                opinion_ids=opinion_ids,
            )

            if fetch_text:
                if not opinion_ids:
                    # fall back to the cluster's sub_opinions for ids
                    try:
                        cluster = self.get_cluster(rec.cluster_id)
                        if not rec.nature_of_suit:
                            rec.nature_of_suit = cluster.get("nature_of_suit", "") or ""
                        for u in cluster.get("sub_opinions", []) or []:
                            try:
                                opinion_ids.append(int(u.rstrip("/").split("/")[-1]))
                            except (ValueError, AttributeError):
                                pass
                        rec.opinion_ids = opinion_ids
                    except requests.HTTPError:
                        pass
                texts = []
                for oid in rec.opinion_ids:
                    try:
                        texts.append(self._opinion_text(self.get_opinion(oid)))
                    except requests.HTTPError:
                        continue   # 401 w/o token, or 404; snippet fallback below
                full = "\n\n".join(t for t in texts if t)
                if full:
                    rec.text, rec.text_source = full, "full"
                else:
                    # Fall back to the (real) highlighted snippets from search.
                    snips = [o.get("snippet", "") for o in row.get("opinions", [])]
                    snip = "\n".join(s for s in snips if s)
                    if snip:
                        rec.text, rec.text_source = snip, "snippet"
                rec.text_len = len(rec.text)

            yield rec
            n += 1
            if max_cases and n >= max_cases:
                return


def harvest_to_jsonl(out_path: Path, query: str | None = None, **kwargs) -> int:
    """Run a harvest and stream CaseRecords to a JSONL file. Returns count."""
    client = CourtListenerClient()
    query = query or config.SETTINGS.search_query
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in client.harvest(query, **kwargs):
            fh.write(rec.to_json() + "\n")
            n += 1
    return n
