import json
import os
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

doc_records = _load_all_doc_records()
doc_record_by_file_id = {
    (r.get("file_id") or "").strip(): r for r in doc_records if (r.get("file_id") or "").strip()
}

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
                        input=[
                            {"role": "system", "content": system},
                            *st.session_state.messages[:-1],
                            {"role": "user", "content": augmented_user},
                        ],
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
                                st.caption(_summarize_doc_record(local_rec))
                                st.caption(f"filename: {local_rec.get('document_filename', '')}")
                            elif remote_filename:
                                st.caption(f"filename: {remote_filename}")
                            if file_id:
                                st.caption(f"file_id: {file_id}")
                            if c.get("quote"):
                                st.write(c["quote"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
