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

    # Build positions of judge headings if present (common pattern: "NAME, JSC" etc).
    headings: list[tuple[int, int, str]] = []
    for judge in judges:
        j_norm = re.sub(r"\s+", " ", judge).strip()
        surname = re.split(r"[,\s]", j_norm, maxsplit=1)[0]
        surname = re.sub(r"[^A-Za-z\-']", "", surname)

        pats: list[str] = []
        if j_norm:
            pats.append(re.escape(j_norm))
        if surname and len(surname) >= 3:
            pats.append(re.escape(surname) + r".{0,40}?(?:JSC|CJ|AG\.?\s*CJ|JA)\b")

        for pat in pats:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                headings.append((m.start(), m.end(), judge))
                break

    headings = sorted(headings, key=lambda x: x[0])
    # De-dup by judge (keep earliest heading).
    seen_j: set[str] = set()
    unique: list[tuple[int, int, str]] = []
    for s, e, j in headings:
        key = j.lower()
        if key in seen_j:
            continue
        seen_j.add(key)
        unique.append((s, e, j))
    headings = unique

    def _classify(snippet: str) -> tuple[str, str]:
        seg = re.sub(r"\s+", " ", snippet).strip()
        if not seg:
            return ("", "")
        if re.search(r"\bdissent(ing)?\b", seg, flags=re.IGNORECASE):
            return ("Dissenting", seg)
        if re.search(r"\bconcurring\b|\bconcur\b", seg, flags=re.IGNORECASE):
            return ("Concurring", seg)
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
        w = re.sub(r"\s+", " ", window).strip()
        if not w:
            return ""
        m = re.search(
            r"(.{0,80}?(?:I\s+(?:have\s+)?(?:read|consider)|in\s+my\s+view|I\s+am\s+of\s+the\s+opinion|I\s+hold|I\s+think|I\s+agree|I\s+dissent).{0,420})",
            w,
            flags=re.IGNORECASE,
        )
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return w[:520].strip()

    results: dict[str, dict[str, str]] = {}

    if headings:
        for idx, (s, e, judge) in enumerate(headings):
            end = headings[idx + 1][0] if idx + 1 < len(headings) else min(len(text), s + 20000)
            segment = text[s:end]
            window = segment[:12000]
            m = re.search(
                r"(.{0,120}?(?:dissent(ing)?|concurring|concur|I\s+would\s+dismiss|I\s+would\s+allow|appeal\s+is\s+dismissed|appeal\s+is\s+allowed).{0,160})",
                window,
                flags=re.IGNORECASE | re.DOTALL,
            )
            snippet = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
            vote, vote_ev = _classify(snippet)

            after_heading = segment[max(0, e - s) : max(0, e - s) + 6000]
            submission = _extract_submission(after_heading)

            out: dict[str, str] = {}
            if vote and vote_ev:
                out["vote"] = vote
                out["vote_evidence"] = vote_ev
            if submission:
                out["submission_extract"] = submission
            if out:
                results[judge] = out

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

    system = """
You are a legal research assistant specialised in analysing Supreme Court case law
from the documents available in the vector store.

Legal cases must be interpreted technically. Therefore, your responses must reflect
the judicial structure of the decision-making process where available in the
retrieved documents.

Your task is to answer questions ONLY using retrieved information from the uploaded
Supreme Court rulings and their metadata.

RESPONSE RULES:

1. Ground all answers strictly in retrieved case law content.
    Do NOT rely on general legal knowledge or assumptions.

2. Where applicable, your analysis MUST extract and present (ONLY if available in the retrieved documents):
    a. The judges who sat on the case (Coram)
    b. Each judge’s opinion or submission
    c. The conclusion reached by each judge
    d. How each judge voted (Majority / Concurring / Dissenting)
    e. Whether any judge abstained, did not participate, or recused themselves
    If any of the above is not available in the retrieved material, say so explicitly and do not guess.

    IMPORTANT:
    - The case name and the Coram (judges) usually appear on the first page.
    - The judges’ opinions/reasoning, votes, and the decision outcome usually appear later in the document.
    - Therefore, you MUST scan the retrieved text from across the document to extract opinions/votes/outcome.

3. When referencing a case, always structure your response as (where applicable):

    Case: [Case Name]

    Coram:
    [List of judges retrieved from the ruling]

    Judicial Opinions:
    - Judge [Name]:
         Position:
         Reasoning:
         Conclusion/Vote:

    Decision Outcome:
    [Majority holding based ONLY on retrieved content]

    Participation Notes (if applicable):
    [Abstentions / Non-participation / Recusal]

    Source:
    Retrieved Supreme Court Ruling

4. If no relevant information is retrieved from the vector store:
    Clearly state: "No relevant Supreme Court ruling was found in the uploaded documents."
    Then ask a clarifying question to improve retrieval.

5. If the user asks for legal advice:
    Provide general legal information from retrieved case law only.
    Clearly state that this is not legal advice.
    Suggest consulting a qualified legal practitioner.

6. If the user asks:
    "What cases do you have?"
    Respond ONLY with the names of the available cases based on file metadata.
    Do NOT summarise or describe them.
    Treat the following as the same request and respond the same way:
    - "What are the main cases?"
    - "List the available cases"
    - "Which cases are in the vector store / uploaded documents?"

7. Do NOT fabricate:
    - case names
    - judges
    - judicial opinions
    - votes
    - legal principles
    - abstentions or participation status

8. Your training knowledge must not be used unless it appears in the retrieved documents.
""".strip()
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

    # Deterministic handling: vote questions are prone to hallucination if retrieval doesn't
    # include explicit vote wording. For these queries, extract evidence directly from the PDF
    # text and only report what is explicitly found.
    if _is_vote_query(prompt):
        target_case = selected_case if (selected_case and selected_case != "(All cases)") else ""
        pdf_path = _best_local_pdf_for_case(target_case) if target_case else None

        if pdf_path is None:
            # If no explicit case selected, fall back to first local PDF (prototype-friendly).
            local_pdfs = _discover_local_pdfs()
            pdf_path = local_pdfs[0] if local_pdfs else None

        if pdf_path is None:
            answer = (
                "No local PDF is available to scan for judge votes on this deployment. "
                "Ensure the ruling PDF is present under the `docs/` folder."
            )
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

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
        lines.append("Coram:")
        if coram:
            lines.extend([f"- {j}" for j in coram])
        else:
            lines.append("(Coram not found on the first page text.)")
        lines.append("")
        lines.append("Judicial Opinions / Votes (evidence-based extracts):")
        if per_judge:
            for j in coram:
                info = per_judge.get(j)
                if not info:
                    lines.append(f"- Judge {j}: Vote not explicitly found in extractable text.")
                    continue
                lines.append(f"- Judge {j}:")
                if info.get("vote"):
                    lines.append(f"  Conclusion/Vote: {info.get('vote','')}")
                    lines.append(f"  Vote Evidence: “{info.get('vote_evidence','')}”")
                else:
                    lines.append("  Conclusion/Vote: (Not explicitly found in extractable text.)")

                if info.get("submission_extract"):
                    lines.append(f"  Submission (extract): “{info.get('submission_extract','')}”")
                else:
                    lines.append("  Submission (extract): (Not located in the extractable text window for this judge.)")
        else:
            lines.append(
                "No explicit vote wording could be located in the extractable text returned from this PDF. "
                "If you want, ask a narrower question (e.g., 'quote where [Judge] states whether they dissent/agree') "
                "or ensure the vector-store retrieval is returning the opinion sections."
            )
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
