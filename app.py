import json
import html
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

st.markdown(
    """
<style>
  .case-list { overflow-x: auto; }
  .case-list ul { padding-left: 1.25rem; margin: 0; }
  .case-list li { white-space: nowrap; }
</style>
""",
    unsafe_allow_html=True,
)


def _render_case_list_html(names: list[str]) -> str:
    items = "\n".join(f"<li>{html.escape(n)}</li>" for n in names)
    return f"<div class=\"case-list\"><ul>{items}</ul></div>"

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
                + [_main_case_name_from_first_page(r) for r in doc_records]
            )
            if n
        ]
    )
)


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
    if sidebar_case_names:
        st.markdown(_render_case_list_html(sidebar_case_names), unsafe_allow_html=True)
    else:
        st.caption("No cases found. Add PDFs under the docs folder and/or add vector_store record JSON files.")

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


prompt = st.chat_input("Ask a question about your matter…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    intake_bits = []
    if practice_area:
        intake_bits.append(f"Practice area: {practice_area}")
    if jurisdiction.strip():
        intake_bits.append(f"Jurisdiction: {jurisdiction.strip()}")

    system = """
You are a legal research assistant specialised in analysing Supreme Court case law from the documents available in the vector store.

Your task is to answer questions ONLY using retrieved information from the uploaded Supreme Court rulings and their metadata.

RESPONSE RULES:

1. Ground all answers strictly in retrieved case law content.
    Do NOT rely on general legal knowledge or assumptions.

2. When referencing a case, always:
    - State the case name
    - Briefly explain the legal principle, holding, or ratio decidendi
    - Base the explanation ONLY on retrieved content

3. If no relevant information is retrieved from the vector store:
    - Clearly state: "No relevant Supreme Court ruling was found in the uploaded documents."
    - Ask the user a clarifying question to improve retrieval.

4. If the user asks for legal advice:
    - Provide general legal information from retrieved case law only
    - Clearly state that this is not legal advice
    - Suggest consulting a qualified legal practitioner.

5. If the user asks: "What cases do you have?"
    Respond ONLY with the names of the available cases based on file metadata.
    Do NOT summarise or describe them.
    Treat the following as the same request and respond the same way:
    - "What are the main cases?"
    - "List the available cases"
    - "Which cases are in the vector store / uploaded documents?"

6. Do NOT fabricate:
    - case names
    - legal principles
    - holdings
    - citations

7. Where applicable, structure your response as:

    Case: [Case Name]
    Principle:
    Application (if relevant to user's question):
    Source: Retrieved Supreme Court Ruling
""".strip()
    if intake_bits:
        system += "\n\n" + "\n".join(intake_bits)

    # Prompt refinement: case names must come from the first page of each PDF.
    system += (
        "\n\nCASE NAME RULE (FIRST PAGE):\n"
        "- The main case name is always on the first page of the document.\n"
        "- When you mention or list a case, use the exact case name as it appears on page 1.\n"
        "- If the user asks for the main/available cases, respond ONLY with the case names (no summaries, no filenames)."
    )

    system += (
        "\n\nJUDGES RULE (FIRST PAGE):\n"
        "- The judges/coram are listed on the first page of the document.\n"
        "- If the user asks who sat on a case (or who presided), answer ONLY from the first-page judges list.\n"
        "- If judges are not available for a case, say so and do not guess."
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
