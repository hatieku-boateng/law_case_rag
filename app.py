import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


st.set_page_config(
    page_title="The Law Student Assistant",
    page_icon="⚖️",
    layout="wide",
)

# Chat layout: user messages on the right, assistant on the left.
st.markdown(
        """
<style>
    div[data-testid="stChatMessage"]:has(.user-msg-marker) {
        flex-direction: row-reverse;
    }
    div[data-testid="stChatMessage"]:has(.user-msg-marker) div[data-testid="stChatMessageContent"] {
        text-align: right;
    }
</style>
""",
        unsafe_allow_html=True,
)

def _mask_key(key: str) -> str:
    if not key:
        return "(missing)"
    return key[:7] + "…" + key[-4:] if len(key) > 12 else "(set)"


load_dotenv()

MODEL = "gpt-4o"
VECTOR_STORE_NAME = "supreme court cases"


BASE_SYSTEM_PROMPT = """
You are The Law Student Assistant: a legal research assistant specialised in analysing Supreme Court case law
ONLY from the documents available in the vector store and/or retrieved excerpts.

NON-NEGOTIABLE GROUNDING RULE:
- Use ONLY the text you retrieve from the uploaded rulings/metadata.
- Do NOT use general legal knowledge, memory, or assumptions.
- If the retrieved text does not contain an answer, say: "Not found in retrieved text." and ask a targeted follow-up.

RETRIEVAL WORKFLOW (FOLLOW THIS ORDER):
1) Retrieve: search for the most relevant excerpts first (especially headings like JUDGMENT/JUDGEMENT/RULING/OPINION and the judge surname + "JSC").
    If the user asks for submissions/opinions/votes per judge, you MUST repeat retrieval per judge (one judge at a time) until you either find a quote for that judge or you conclude it is not present in retrieved text.
2) Verify: before you claim anything (submission/vote/outcome), confirm the exact wording exists in the retrieved text.
3) Quote: when you present a per-judge submission/opinion/conclusion/vote, include at least one direct quote copied verbatim.
4) If you cannot quote it verbatim, you must write: "Not found in retrieved text." (no paraphrase, no guessing).

CASE NAME + CORAM RULE (FIRST PAGE):
- The case name and Coram (judges) usually appear on page 1.
- Do NOT treat Coram, case caption, or front-matter as a judge's submission/reasoning.

JUDGE ATTRIBUTION SAFETY:
- Only attribute text to a judge if it appears inside that judge’s own opinion/judgment section.
- Do NOT misattribute citations like "opinion of Justice X in another case" as Justice X’s submission in the current case.
- Ignore role markers like "(PRESIDING)" for submissions; those are not reasoning.

VOTES / OUTCOME SAFETY:
- Do NOT infer "dismissed/allowed/unanimous" unless the retrieved text explicitly states it and you can quote it.
- Do NOT infer how each judge voted unless the retrieved text explicitly indicates concurrence/dissent/allow/dismiss (quote required).

MANDATORY PER-JUDGE COVERAGE:
- If you have a Coram list (judges who sat), you MUST produce a per-judge entry for EVERY judge in the Coram whenever the user asks about submissions/opinions/reasoning/conclusions/votes.
- For each judge, output either (a) a direct quote showing that judge's submission/opinion/reasoning, or (b) "Not found in retrieved text.".
- Do not skip judges.

RESPONSE FORMAT (USE THIS STRUCTURE WHEN APPLICABLE):

Case: [Exact case name as it appears on page 1 (or AVAILABLE CASES list)]

Coram (from page 1 if available):
- [Judge 1]
- ...

Per-Judge Analysis (evidence-only):
- Judge [Name] (repeat for every Coram judge):
    - Submission/Reasoning (quote): "..." OR Not found in retrieved text.
    - Conclusion/Vote (quote): "..." OR Not found in retrieved text.

Decision Outcome (evidence-only):
- "..." OR Not found in retrieved text.

Participation Notes (if stated):
- "..." OR Not found in retrieved text.

If the user asks for legal advice:
- Provide general information only from retrieved case law and state it is not legal advice.
""".strip()


def _load_vector_store_record() -> Optional[dict[str, Any]]:
    paths = [
        os.path.join("embeddings", "vector_store.supreme-court-cases.json"),
        os.path.join("vector_store", "doc_1.vector_store_record.json"),
        os.path.join("vector_store", "doc_2.vector_store_record.json"),
    ]
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _load_all_doc_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = Path("vector_store")
    if not base.exists() or not base.is_dir():
        return records
    for path in sorted(base.glob("*.vector_store_record.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def _find_vector_store_id_by_name(client: OpenAI, name: str) -> Optional[str]:
    try:
        page = client.vector_stores.list(limit=100)
    except Exception:
        return None
    for vs in getattr(page, "data", []) or []:
        if (getattr(vs, "name", "") or "") == name:
            return getattr(vs, "id", None)
    return None


def _extract_file_citations(resp: Any) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    output = getattr(resp, "output", None) or []
    for out in output:
        content = getattr(out, "content", None) or []
        for item in content:
            if getattr(item, "type", None) != "output_text":
                continue
            annotations = getattr(item, "annotations", None) or []
            for ann in annotations:
                ann_type = getattr(ann, "type", None)
                if ann_type != "file_citation":
                    continue
                file_id = getattr(ann, "file_id", "") or ""
                quote = getattr(ann, "quote", "") or ""
                citations.append({"file_id": file_id, "quote": quote})

    # De-dup
    seen = set()
    unique: list[dict[str, str]] = []
    for c in citations:
        key = (c.get("file_id", ""), c.get("quote", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def _summarize_doc_record(rec: dict[str, Any]) -> str:
    meta = rec.get("metadata") or {}
    title = (meta.get("title") or "").strip()
    cite = (meta.get("media_neutral_citation") or meta.get("citation_full") or "").strip()
    date = (meta.get("judgment_date_display") or meta.get("judgment_date") or "").strip()
    bits = [b for b in [title, cite, date] if b]
    return " — ".join(bits) if bits else "(metadata not available)"


@st.cache_data(show_spinner=False)
def _extract_first_page_text(pdf_path: str, doc_sha256: str = "") -> str:
    """Return extracted text for the first page of a PDF.

    `doc_sha256` is included to ensure Streamlit cache invalidates when the file changes.
    """
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            return ""
        return (reader.pages[0].extract_text() or "").strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _extract_full_pdf_text(pdf_path: str, doc_sha256: str = "") -> str:
    """Return extracted text for all pages of a PDF.

    Used for evidence-based extraction (e.g., opinions/votes) when users ask
    questions that require scanning beyond the first page.
    """
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(pdf_path)
        parts: list[str] = []
        for page in reader.pages or []:
            parts.append((page.extract_text() or "").strip())
        return "\n\n".join([p for p in parts if p]).strip()
    except Exception:
        return ""


def _discover_local_pdfs() -> list[Path]:
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists() or not docs_dir.is_dir():
        return []
    return sorted(docs_dir.glob("*.pdf"))


def _local_pdf_manifest() -> str:
    """A stable string that changes when docs PDFs change.

    Used to bust Streamlit caches on deploys/updates.
    """
    parts: list[str] = []
    for p in _discover_local_pdfs():
        try:
            st_ = p.stat()
            parts.append(f"{p.name}:{st_.st_size}:{st_.st_mtime_ns}")
        except Exception:
            parts.append(f"{p.name}:?")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _case_names_from_local_pdfs(_cache_buster: str = "") -> list[str]:
    names: list[str] = []
    for pdf_path in _discover_local_pdfs():
        # Include file stats to avoid stale cache when PDFs are updated.
        try:
            st_ = pdf_path.stat()
            sig = f"{st_.st_size}:{st_.st_mtime_ns}"
        except Exception:
            sig = ""
        text = _extract_first_page_text(str(pdf_path), doc_sha256=sig)
        name = _case_name_from_first_page_text(text)
        if name:
            names.append(name)
    return sorted(set(names))


def _case_name_from_first_page_text(text: str) -> str:
    if not text:
        return ""

    raw_lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in raw_lines if ln]
    if not lines:
        return ""

    # Filter out common header noise.
    noise = re.compile(
        r"^(republic of|in the|supreme court|court of|holden at|coram|before|judg(e)?ment|date\b|between\b)",
        re.IGNORECASE,
    )
    candidates = [ln for ln in lines[:60] if not noise.search(ln)]
    if not candidates:
        candidates = lines[:60]

    sep_re = re.compile(r"\b(v|vs|vrs|versus)\b\.?", re.IGNORECASE)

    # 1) Look for a single line containing both parties.
    for ln in candidates:
        if sep_re.search(ln) and len(ln) <= 220:
            # Normalize connector to 'v'.
            normalized = sep_re.sub("v", ln)
            normalized = re.sub(r"\s+", " ", normalized).strip(" -–—:;,.\t")
            # Must contain at least one connector and reasonable length.
            if re.search(r"\bv\b", normalized) and 5 < len(normalized) <= 220:
                return normalized

    # 2) Handle captions split across lines, e.g. party A / 'Vrs' / party B.
    for i, ln in enumerate(candidates):
        if re.match(r"^(v|vs|vrs|versus)\b\.?$", ln, flags=re.IGNORECASE):
            prev_ln = candidates[i - 1] if i > 0 else ""
            next_ln = candidates[i + 1] if i + 1 < len(candidates) else ""
            combined = f"{prev_ln} v {next_ln}".strip()
            combined = re.sub(r"\s+", " ", combined).strip(" -–—:;,.\t")
            if re.search(r"\bv\b", combined) and 5 < len(combined) <= 220:
                return combined

        if re.match(r"^(v|vs|vrs|versus)\b", ln, flags=re.IGNORECASE):
            prev_ln = candidates[i - 1] if i > 0 else ""
            rest = sep_re.sub("v", ln)
            combined = f"{prev_ln} {rest}".strip()
            combined = re.sub(r"\s+", " ", combined).strip(" -–—:;,.\t")
            if re.search(r"\bv\b", combined) and 5 < len(combined) <= 220:
                return combined

    # 3) Fallback: search across joined candidate text.
    joined = " ".join(candidates)
    joined = re.sub(r"\s+", " ", joined)
    m = re.search(r"(.{0,60}?\b(?:v|vs|vrs|versus)\b\.?\s+.{0,80})", joined, flags=re.IGNORECASE)
    if m:
        s = sep_re.sub("v", m.group(1))
        s = re.sub(r"\s+", " ", s).strip(" -–—:;,.\t")
        if re.search(r"\bv\b", s) and 5 < len(s) <= 220:
            return s

    return ""


def _judges_from_first_page_text(text: str) -> list[str]:
    """Extract the panel/coram judges from first-page text.

    Handles both:
    - A single CORAM line containing multiple judges
    - Multiple lines each containing a judge ending with JSC/CJ/etc.
    """
    if not text:
        return []

    # Normalize spacing to make regex matching stable.
    normalized = re.sub(r"\s+", " ", text)

    # Match common judge suffixes seen in Ghana SC documents.
    # Examples: "ATUGUBA, JSC (PRESIDING)", "BAFFOE-BONNIE AG. CJ (PRESIDING)", "AMADU JSC"
    judge_re = re.compile(
        r"(?P<judge>"
        r"[A-Z][A-Z\- '\.()]*?"
        r"\s*(?:,\s*)?"
        r"(?:AG\.?\s*CJ|CJ|JSC|J\.?SC|JA|J\.?A\.?)(?:\s*\(PRESIDING\))?"
        r")",
        flags=re.IGNORECASE,
    )

    found = []
    for m in judge_re.finditer(normalized):
        j = m.group("judge")
        j = re.sub(r"\s+", " ", j)
        j = j.replace(" ,", ",").strip(" -–—:;,.\t")
        if j:
            found.append(j)

    # De-dup, preserve order.
    seen = set()
    judges: list[str] = []
    for j in found:
        key = re.sub(r"\s+", " ", j).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        judges.append(j)
    return judges


def _judges_from_first_page(rec: dict[str, Any]) -> list[str]:
    doc_path = (rec.get("document_path") or rec.get("document_filename") or "").strip()
    if not doc_path:
        return []
    pdf_path = Path(doc_path)
    if not pdf_path.is_absolute():
        pdf_path = Path(str(pdf_path).replace("\\", os.sep))
    if not pdf_path.exists():
        pdf_path = Path(__file__).parent / Path(str(doc_path).replace("\\", os.sep))
    text = _extract_first_page_text(str(pdf_path), (rec.get("document_sha256") or "").strip())
    return _judges_from_first_page_text(text)


def _main_case_name_from_first_page(rec: dict[str, Any]) -> str:
    doc_path = (rec.get("document_path") or rec.get("document_filename") or "").strip()
    if not doc_path:
        return ""
    pdf_path = Path(doc_path)
    if not pdf_path.is_absolute():
        pdf_path = Path(str(pdf_path).replace("\\", os.sep))
    if not pdf_path.exists():
        # Try resolving relative to workspace root.
        pdf_path = Path(__file__).parent / Path(str(doc_path).replace("\\", os.sep))
    text = _extract_first_page_text(str(pdf_path), (rec.get("document_sha256") or "").strip())

    extracted = _case_name_from_first_page_text(text)

    # If metadata title is present on page 1, prefer it (cleaner formatting) while
    # still honoring the "first page" rule.
    meta_title = ((rec.get("metadata") or {}).get("title") or "").strip()
    if meta_title and text:
        hay = re.sub(r"\s+", " ", text).lower()
        needle = re.sub(r"\s+", " ", meta_title).lower()
        if needle and needle in hay:
            return meta_title

    return extracted


def _summarize_doc_record_with_case_name(rec: dict[str, Any], case_name: str) -> str:
    meta = rec.get("metadata") or {}
    cite = (meta.get("media_neutral_citation") or meta.get("citation_full") or "").strip()
    date = (meta.get("judgment_date_display") or meta.get("judgment_date") or "").strip()
    title = case_name.strip() or (meta.get("title") or "").strip()
    bits = [b for b in [title, cite, date] if b]
    return " — ".join(bits) if bits else "(metadata not available)"


def _is_case_list_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    # Examples:
    # - what cases do you have
    # - what are the main cases
    # - list the available cases
    # - which cases are in the vector store
    pattern = (
        r"\b(what|which|list|show)\b.*\b(main|available)?\b.*\bcases\b"
        r"|\bcases\b.*\b(do you have|are available|in the vector store|in the uploaded documents)\b"
    )
    return re.search(pattern, t) is not None


def _is_vote_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    pattern = (
        r"\b(vote|voted|voting|majority|minority|dissent|dissenting|concurring|concur|\bwho\s+won\b|\bwho\s+lost\b)\b"
        r"|\bhow\s+did\s+each\s+judge\s+vote\b"
        r"|\bhow\s+did\s+the\s+judges\s+vote\b"
        r"|\bwas\s+it\s+unanimous\b"
    )
    return re.search(pattern, t) is not None


def _is_judicial_breakdown_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    if _is_vote_query(t):
        return True

    # Queries asking for per-judge submissions/opinions/reasoning/conclusions.
    pattern = (
        r"\b(each|every|all)\s+judge\b"
        r"|\bjudges\s+who\s+sat\b"
        r"|\bper\s+judge\b"
        r"|\bcoram\b"
        r"|\b(submission|submissions|opinion|opinions|reasoning|conclusion|conclusions)\b"
    )
    return re.search(pattern, t) is not None


def _best_local_pdf_for_case(selected_case: str) -> Optional[Path]:
    """Best-effort mapping from a selected case name to a local PDF."""
    want = (selected_case or "").strip().lower()
    if not want:
        return None

    manifest = _local_pdf_manifest()
    for pdf_path in _discover_local_pdfs():
        try:
            st_ = pdf_path.stat()
            sig = f"{st_.st_size}:{st_.st_mtime_ns}:{manifest}"
        except Exception:
            sig = manifest
        first = _extract_first_page_text(str(pdf_path), doc_sha256=sig)
        name = _case_name_from_first_page_text(first).strip().lower()
        if name and name == want:
            return pdf_path

    # Fallback: token overlap heuristic.
    want_tokens = {w for w in re.split(r"\W+", want) if len(w) >= 4}
    best: tuple[int, Optional[Path]] = (0, None)
    for pdf_path in _discover_local_pdfs():
        try:
            st_ = pdf_path.stat()
            sig = f"{st_.st_size}:{st_.st_mtime_ns}:{manifest}"
        except Exception:
            sig = manifest
        first = _extract_first_page_text(str(pdf_path), doc_sha256=sig)
        name = _case_name_from_first_page_text(first).strip().lower()
        hay = name or first.lower()
        score = sum(1 for tok in want_tokens if tok in hay)
        if score > best[0]:
            best = (score, pdf_path)
    return best[1]


def _extract_votes_from_full_text(full_text: str, judges: list[str]) -> dict[str, dict[str, str]]:
    """Extract evidence-backed vote + submission snippets per judge.

    Returns mapping judge ->
      {
        "vote": str,
        "vote_evidence": str,
        "submission_extract": str,
      }

    Only fills fields when explicit extractable evidence is found.
    """
    if not full_text or not judges:
        return {}

    text = re.sub(r"\r\n?", "\n", full_text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Find likely start positions for each judge's opinion/ruling section.
    # Avoid grabbing the coram/front-matter occurrence by preferring contexts like:
    # "RULING/JUDGMENT/OPINION ... BY/DELIVERED BY ... [JUDGE]".
    header_cue_re = re.compile(
        r"\b(ruling|judg(e)?ment|opinion|delivered\s+by|by\s+his\s+lordship|by\s+her\s+ladyship)\b",
        flags=re.IGNORECASE,
    )

    judge_starts: list[tuple[int, int, str]] = []
    for judge in judges:
        j_norm = re.sub(r"\s+", " ", judge).strip()
        surname = re.split(r"[,\s]", j_norm, maxsplit=1)[0]
        surname = re.sub(r"[^A-Za-z\-']", "", surname)

        pats: list[str] = []
        if j_norm:
            pats.append(re.escape(j_norm))
        if surname and len(surname) >= 3:
            pats.append(re.escape(surname) + r".{0,40}?(?:JSC|CJ|AG\.?\s*CJ|JA)\b")

        best: tuple[int, Optional[tuple[int, int]]] = (-10_000, None)
        for pat in pats:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                s, e = m.start(), m.end()
                # Score this occurrence.
                score = 0
                before = text[max(0, s - 120) : s]
                after = text[e : min(len(text), e + 400)]

                if header_cue_re.search(before) or header_cue_re.search(after):
                    score += 6
                if re.search(r"\b(ruling|judg(e)?ment)\b\s*(?:of|by)\b", before, flags=re.IGNORECASE):
                    score += 6
                if re.search(r"\b(delivered\s+by)\b", before, flags=re.IGNORECASE):
                    score += 6

                # Penalize very-early matches (often coram/front matter).
                if s < 2500:
                    score -= 4

                # Prefer matches that appear as a standalone heading line.
                line_start = text.rfind("\n", 0, s) + 1
                line_end = text.find("\n", e)
                if line_end == -1:
                    line_end = min(len(text), e + 120)
                line = text[line_start:line_end]
                if len(line.strip()) <= 90 and re.search(r"\b(JSC|CJ|JA)\b", line, flags=re.IGNORECASE):
                    score += 2

                if score > best[0]:
                    best = (score, (s, e))

        if best[1] is not None:
            s, e = best[1]
            judge_starts.append((s, e, judge))

    # Sort and de-dup starts.
    judge_starts = sorted(judge_starts, key=lambda x: x[0])
    seen: set[str] = set()
    headings: list[tuple[int, int, str]] = []
    for s, e, j in judge_starts:
        key = j.lower()
        if key in seen:
            continue
        seen.add(key)
        headings.append((s, e, j))

    def _classify(snippet: str) -> tuple[str, str]:
        seg = re.sub(r"\s+", " ", snippet).strip()
        if not seg:
            return ("", "")
        if re.search(r"\bdissent(ing)?\b", seg, flags=re.IGNORECASE):
            return ("Dissenting", seg)
        if re.search(r"\bconcurring\b|\bconcur\b", seg, flags=re.IGNORECASE):
            return ("Concurring", seg)
        if re.search(r"\bI\s+(?:fully\s+)?agree\b.{0,120}\b(dismiss|allowed?)\b", seg, flags=re.IGNORECASE):
            return ("Agrees (explicit)", seg)
        if re.search(
            r"\bI\s+would\s+dismiss\b|\bI\s+dismiss\b|\bthe\s+appeal\s+is\s+dismissed\b",
            seg,
            flags=re.IGNORECASE,
        ):
            return ("Supports dismissal", seg)
        if re.search(
            r"\bI\s+would\s+allow\b|\bI\s+allow\b|\bthe\s+appeal\s+is\s+allowed\b",
            seg,
            flags=re.IGNORECASE,
        ):
            return ("Supports allowing", seg)
        return ("", "")

    def _extract_submission(window: str) -> str:
        # Keep line breaks for heading detection, but normalize excessive whitespace.
        w = re.sub(r"\t+", " ", window)
        w = re.sub(r"[ ]{2,}", " ", w)
        w = re.sub(r"\n{3,}", "\n\n", w)
        w = w.strip()
        if not w:
            return ""

        # Prefer content under a RULING/JUDGMENT heading if present.
        m_head = re.search(r"\b(ruling|judg(e)?ment|judgement)\b", w, flags=re.IGNORECASE)
        if m_head:
            start = m_head.start()
            w2 = w[start : start + 5000]
        else:
            w2 = w[:5000]

        # Extract a meaningful excerpt (not just the heading line).
        w2_one = re.sub(r"\s+", " ", w2).strip()
        m = re.search(
            r"(.{0,120}?(?:I\s+(?:have\s+)?(?:read|consider)|in\s+my\s+view|I\s+am\s+of\s+the\s+opinion|I\s+hold|I\s+think|I\s+agree|I\s+dissent).{0,520})",
            w2_one,
            flags=re.IGNORECASE,
        )
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()

        # Fallback: take the first substantial excerpt beyond any short heading.
        return w2_one[:700].strip()

    results = {}
    text_len = len(text)
    
    headings = []
    # General strategy to find sections belonging to a judge 
    # Matches patterns like "JUSTICE SMITH JSC:", "SMITH JSC", "SMITH, JSC", "OPINION OF SMITH, JSC"
    for j in judges:
        j_n = re.sub(r"\s+", " ", j).strip()
        surname = re.split(r"[,\s]", j_n, maxsplit=1)[0]
        surname = re.sub(r"[^A-Za-z\-']", "", surname)
        
        # Broad lookup looking for surname + Judicial Title
        pat = r"(?im)^.{0,30}" + re.escape(surname) + r".{0,40}?(?:JSC|J\.S\.C|CJ|AG\.?\s*CJ|JA)\b.{0,30}$"
        hits = []
        for m in re.finditer(pat, text):
            if m.start() > 2500:  # Skip coram/front matter, usually on page 1-2
                hits.append(m.start())
        if hits:
            # We take the first hit that occurs after title pages
            headings.append((hits[0], j))
            
    headings.sort(key=lambda x: x[0])

    for idx, (s, judge) in enumerate(headings):
        nxt = headings[idx+1][0] if idx + 1 < len(headings) else text_len
        section = text[s:nxt]
        
        vote = ""
        vote_ev = ""
        
        # Look for explicit concluding language
        m_dismiss = re.search(r"\b(I\s+(?:would|therefore)?\s*dismiss|petition\s+is\s+(?:hereby\s+)?dismissed|appeal\s+is\s+(?:hereby\s+)?dismissed)\b", section, re.I)
        m_allow = re.search(r"\b(I\s+(?:would|therefore)?\s*allow|petition\s+is\s+(?:hereby\s+)?allowed|appeal\s+is\s+(?:hereby\s+)?allowed)\b", section, re.I)
        
        if m_dismiss and m_allow:
            vote = "Mixed opinion (mentions both dismiss and allow)"
            start_ev = max(0, m_dismiss.start()-50)
            end_ev = min(len(section), m_dismiss.end()+50)
            vote_ev = section[start_ev:end_ev].replace('\n',' ')
        elif m_dismiss:
            vote = "Supports dismissal"
            start_ev = max(0, m_dismiss.start()-50)
            end_ev = min(len(section), m_dismiss.end()+50)
            vote_ev = section[start_ev:end_ev].replace('\n',' ')
        elif m_allow:
            vote = "Supports allowing"
            start_ev = max(0, m_allow.start()-50)
            end_ev = min(len(section), m_allow.end()+50)
            vote_ev = section[start_ev:end_ev].replace('\n',' ')
        else:
            m_concur = re.search(r"\b(I\s+(?:fully\s+)?(?:agree|concur)\s+with)\b", section, re.I)
            if m_concur:
                vote = "Concurring"
                start_ev = max(0, m_concur.start()-30)
                end_ev = min(len(section), m_concur.end()+80)
                vote_ev = section[start_ev:end_ev].replace('\n',' ')

        # Grab qualitative substring. Skip possible empty line directly under heading
        start_idx = section.find("\n")
        if start_idx == -1: start_idx = 0
        
        # We look around 1200 chars to form a good abstract of the opinion
        sub_text = section[start_idx:start_idx+1200].replace('\n', ' ')
        sub_text = re.sub(r"\s+", " ", sub_text).strip()
        
        results[judge] = {
            "vote": vote,
            "vote_evidence": vote_ev.strip(),
            "submission_extract": sub_text
        }

    return results


def _case_name_from_record(rec: dict[str, Any]) -> str:
    meta = rec.get("metadata") or {}
    title = (meta.get("title") or "").strip()
    if title:
        return title
    filename = (rec.get("document_filename") or rec.get("filename") or "").strip()
    return filename


def _get_remote_filename(client: OpenAI, file_id: str) -> str:
    try:
        fobj = client.files.retrieve(file_id)
    except Exception:
        return ""
    return (getattr(fobj, "filename", "") or "").strip()


def _is_election_petition_case_name(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    return ("akufo" in n and "mahama" in n) or ("election" in n and "petition" in n)

if "messages" not in st.session_state:
    st.session_state.messages = []


doc_records = _load_all_doc_records()
doc_record_by_file_id = {
    (r.get("file_id") or "").strip(): r for r in doc_records if (r.get("file_id") or "").strip()
}

sidebar_case_names = sorted(
    set(
        [
            n
            for n in (
                _case_names_from_local_pdfs(_local_pdf_manifest())
                + [(_main_case_name_from_first_page(r) or _case_name_from_record(r)) for r in doc_records]
            )
            if n
        ]
    )
)

dropdown_case_names = [n for n in sidebar_case_names if _is_election_petition_case_name(n)]

if "selected_case" not in st.session_state:
    st.session_state.selected_case = dropdown_case_names[0] if dropdown_case_names else ""
if "selected_case_prev" not in st.session_state:
    st.session_state.selected_case_prev = st.session_state.selected_case


with st.sidebar:
    practice_area = st.selectbox(
        "Practice area",
        [
            
            "Supreme Court",
        ],
        index=0,
    )
    jurisdiction = st.selectbox("Jurisdiction", ["Ghana"], index=0)

    st.divider()

    st.markdown("**Cases in the database**")
    if dropdown_case_names:
        if st.session_state.selected_case not in dropdown_case_names:
            st.session_state.selected_case = dropdown_case_names[0]
        st.selectbox(
            "Select a case",
            dropdown_case_names,
            key="selected_case",
        )
        current = st.session_state.selected_case
        if current != st.session_state.selected_case_prev:
            st.session_state.selected_case_prev = current
            if current:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Selected case: **{current}**. What would you like to know about it?",
                    }
                )
            st.rerun()
    else:
        st.caption(
            "Election petition case not found. Ensure the Akufo-Addo v Mahama ruling is available in `docs/` and/or `vector_store/*.vector_store_record.json`."
        )

    if st.button("New chat"):
        st.session_state.messages = []


st.markdown("# ⚖️ The Law Student Assistant")
st.caption(
    "Research-and-answer chat for legal teams. Outputs may be incomplete; verify against primary sources."
)


client: Optional[OpenAI]
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("Set OPENAI_API_KEY in .env to enable chat.")


vector_store_record = _load_vector_store_record()
vector_store_id = (vector_store_record or {}).get("vector_store_id") if vector_store_record else None
if not vector_store_id and client is not None:
    vector_store_id = _find_vector_store_id_by_name(client, VECTOR_STORE_NAME)
retrieval_enabled = True

file_search_tools = None
if retrieval_enabled:
    if vector_store_id:
        file_search_tools = [
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ]
    else:
        st.info(
            "Vector store not found. Run the notebook ingestion cells first (it should create embeddings/vector_store.supreme-court-cases.json)."
        )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "user":
            st.markdown('<span class="user-msg-marker"></span>', unsafe_allow_html=True)


prompt = st.chat_input("Ask a question about your matter…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render the just-submitted user message immediately.
    # The chat history above was rendered before we appended this prompt.
    with st.chat_message("user"):
        st.markdown(prompt)
        st.markdown('<span class="user-msg-marker"></span>', unsafe_allow_html=True)

    intake_bits = []
    if practice_area:
        intake_bits.append(f"Practice area: {practice_area}")
    if jurisdiction.strip():
        intake_bits.append(f"Jurisdiction: {jurisdiction.strip()}")
    selected_case = (st.session_state.get("selected_case") or "").strip()
    if selected_case and selected_case != "(All cases)":
        intake_bits.append(f"Selected case (user focus): {selected_case}")

    system = BASE_SYSTEM_PROMPT
    if intake_bits:
        system += "\n\n" + "\n".join(intake_bits)

    if selected_case and selected_case != "(All cases)":
        system += (
            "\n\nFOCUS CASE:\n"
            f"- The user selected this case to focus on: {selected_case}\n"
            "- If the user’s question doesn’t name a case, assume they mean the selected case.\n"
            "- If the user names a different case, follow the user."
        )

    # Prompt refinement: case names must come from the first page of each PDF.
    system += (
        "\n\nCASE NAME RULE (FIRST PAGE):\n"
        "- The main case name is always on the first page of the document.\n"
        "- When you mention or list a case, use the exact case name as it appears on page 1.\n"
        "- If the user asks for the main/available cases, respond ONLY with the case names (no summaries, no filenames)."
    )

    system += (
        "\n\nJUDGES RULE (FIRST PAGE):\n"
        "- The Coram (judges who sat) usually appears on the first page of the document.\n"
        "- If the user asks who sat on a case (or who presided), answer ONLY from the first-page Coram/judges list.\n"
        "- If judges are not available for a case, say so and do not guess."
    )

    system += (
        "\n\nJUDICIAL OPINIONS / VOTES RULE (SCAN DOCUMENT):\n"
        "- A judge’s ruling/opinion, reasoning, conclusion, vote (majority/concurring/dissent), and participation notes are usually NOT on page 1.\n"
        "- Extract these only from retrieved text found elsewhere in the ruling (scan all retrieved sections).\n"
        "- If the retrieved material does not contain this information, say so and ask a clarifying question."
    )

    first_page_case_names = []
    for r in doc_records:
        name = _main_case_name_from_first_page(r)
        if not name:
            name = _case_name_from_record(r)
        if name:
            first_page_case_names.append(name)

    # Also derive names directly from local PDFs (helps on Streamlit Cloud if vector_store/*.json
    # or vector store tools aren't available).
    first_page_case_names.extend(_case_names_from_local_pdfs(_local_pdf_manifest()))

    first_page_case_names = sorted(set(first_page_case_names))
    if first_page_case_names:
        system += "\n\nAVAILABLE CASES (from first page):\n" + "\n".join(
            f"- {name}" for name in first_page_case_names
        )

    judges_lines: list[str] = []
    for r in doc_records:
        case_name = _main_case_name_from_first_page(r) or _case_name_from_record(r)
        judges = _judges_from_first_page(r)
        if case_name and judges:
            judges_lines.append(f"- {case_name}: " + "; ".join(judges))

    if judges_lines:
        system += "\n\nJUDGES (from first page):\n" + "\n".join(judges_lines)

    # Deterministic handling for judicial breakdown queries (submissions/votes by judge).
    # Retrieval can miss opinion sections; for these queries we scan the selected local PDF
    # and return quote-backed per-judge entries for ALL judges in Coram.
    if _is_judicial_breakdown_query(prompt):
        target_case = selected_case if (selected_case and selected_case != "(All cases)") else ""
        pdf_path = _best_local_pdf_for_case(target_case) if target_case else None

        if pdf_path is None:
            local_pdfs = _discover_local_pdfs()
            pdf_path = local_pdfs[0] if local_pdfs else None

        if pdf_path is not None:
            try:
                st_ = pdf_path.stat()
                sig = f"{st_.st_size}:{st_.st_mtime_ns}"
            except Exception:
                sig = ""

            first_page = _extract_first_page_text(str(pdf_path), doc_sha256=sig)
            case_name = _case_name_from_first_page_text(first_page) or pdf_path.name
            coram = _judges_from_first_page_text(first_page)
            full_text = _extract_full_pdf_text(str(pdf_path), doc_sha256=sig)
            per_judge = _extract_votes_from_full_text(full_text, coram)

            lines: list[str] = []
            lines.append(f"Case: {case_name}")
            lines.append("")
            lines.append("Coram (from first page):")
            if coram:
                lines.extend([f"- {j}" for j in coram])
            else:
                lines.append("- Not found in retrieved text.")

            lines.append("")
            lines.append("Per-Judge Analysis (evidence-only):")

            if coram:
                for j in coram:
                    info = per_judge.get(j, {})
                    submission = (info.get("submission_extract") or "").strip()
                    vote = (info.get("vote") or "").strip()
                    vote_ev = (info.get("vote_evidence") or "").strip()

                    lines.append(f"- **Judge {j}**:")
                    if submission:
                        # Clean up formatting for display
                        clean_sub = re.sub(r'\s+', ' ', submission)
                        lines.append(f"  - **Submission/Reasoning (quote)**: \"{clean_sub}...\"")
                    else:
                        lines.append("  - **Submission/Reasoning (quote)**: Not clearly extracted. Please verify in the full document.")

                    if vote and vote_ev:
                        lines.append(f"  - **Conclusion/Vote**: {vote}")
                        clean_ev = re.sub(r'\s+', ' ', vote_ev)
                        lines.append(f"  - **Vote Evidence (quote)**: \"{clean_ev}\"")
                    elif vote:
                        lines.append(f"  - **Conclusion/Vote**: {vote}")
                    else:
                        lines.append("  - **Conclusion/Vote**: Not explicitly matched by heuristic keywords (check submission text).")
            else:
                lines.append("- Unable to build per-judge breakdown because Coram was not found on page 1.")

            lines.append("")
            lines.append("Decision Outcome (evidence-only):")
            lines.append("- Not found in retrieved text.")

            answer = "\n".join(lines)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

    # Deterministic handling: deployed models can occasionally drift on instruction-following.
    # For case-list queries, return the first-page-derived case names directly.
    if _is_case_list_query(prompt):
        if first_page_case_names:
            answer = "\n".join(f"- {name}" for name in first_page_case_names)
        else:
            answer = (
                "No cases are available on this deployment. "
                "Ensure your PDFs are committed under the `docs/` folder (e.g., `docs/doc_1.pdf`, `docs/doc_2.pdf`) "
                "and/or that `vector_store/*.vector_store_record.json` exists."
            )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    available_case_names = sorted({n for n in (_case_name_from_record(r) for r in doc_records) if n})
    if available_case_names:
        system += "\n\nAVAILABLE CASES (from metadata):\n" + "\n".join(
            f"- {name}" for name in available_case_names
        )
    augmented_user = prompt

    with st.chat_message("assistant"):
        if client is None:
            st.markdown("API key missing. Add `OPENAI_API_KEY` to your `.env`.")
            answer = "API key missing."
        else:
            with st.spinner("Thinking…"):
                resp = None
                try:
                    create_kwargs: dict[str, Any] = {}
                    if file_search_tools:
                        create_kwargs["tools"] = file_search_tools

                    resp = client.responses.create(
                        model=MODEL,
                        instructions=system,
                        input=st.session_state.messages,
                        temperature=0,
                        **create_kwargs,
                    )
                    answer = (resp.output_text or "").strip() or "(No response text.)"
                except Exception as e:
                    answer = f"Request failed: {e}"

            st.markdown(answer)

            if file_search_tools and resp is not None:
                citations = _extract_file_citations(resp)
                if citations:
                    with st.expander("Sources"):
                        for i, c in enumerate(citations, start=1):
                            file_id = (c.get("file_id", "") or "").strip()
                            local_rec = doc_record_by_file_id.get(file_id)
                            remote_filename = _get_remote_filename(client, file_id) if (client and file_id) else ""

                            st.markdown(f"**Source {i}**")
                            if local_rec is not None:
                                case_name = _main_case_name_from_first_page(local_rec)
                                st.caption(_summarize_doc_record_with_case_name(local_rec, case_name))
                                st.caption(f"filename: {local_rec.get('document_filename', '')}")
                            elif remote_filename:
                                st.caption(f"filename: {remote_filename}")
                            if file_id:
                                st.caption(f"file_id: {file_id}")
                            if c.get("quote"):
                                st.write(c["quote"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
