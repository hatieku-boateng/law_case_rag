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
- Use ONLY the text you retrieve from the uploaded rulings.
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

Case: [Exact case name as it appears on page 1]

Coram (from page 1 if available):
- [Judge 1]
- ...

Per-Judge Analysis (evidence-only):
- Judge [Name] (repeat for every Coram judge):
    - Submission/Reasoning Summary: [Provide a very brief summary highlighting only the very key points of what they said, in 1-3 short sentences.]
    - Key Quote: "..." OR Not found in retrieved text.
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
            for n in [_case_name_from_record(r) for r in doc_records]
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
            "Election petition case not found. Ensure the case rulings are available in the vector store."
        )

    if st.button("New chat"):
        st.session_state.messages = []


st.markdown("# ⚖️ The Law Student Assistant")
st.caption(
    "Research-and-answer chat for legal teams. Outputs may be incomplete; verify against primary sources."
)

if st.session_state.get("selected_case"):
    st.info(f"**Focused Case:** {st.session_state.selected_case}")


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
            "Vector store not found. Please ensure your documents are ingested into the OpenAI Vector Store."
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
                                case_name = _case_name_from_record(local_rec)
                                st.caption(_summarize_doc_record(local_rec))
                                st.caption(f"filename: {local_rec.get('document_filename', '')}")
                            elif remote_filename:
                                st.caption(f"filename: {remote_filename}")
                            if file_id:
                                st.caption(f"file_id: {file_id}")
                            if c.get("quote"):
                                st.write(c["quote"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
