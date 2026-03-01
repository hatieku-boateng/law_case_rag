import json
import os
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
VECTOR_STORE_NAME = os.getenv("OPENAI_VECTOR_STORE_NAME", "supreme court cases")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip() or None
VECTOR_STORE_SUMMARY_PATH = os.path.join("embeddings", "vector_store.supreme-court-cases.json")


BASE_SYSTEM_PROMPT = """
You are The Law Student Assistant: a legal research assistant specialised in analysing Supreme Court case law
ONLY from the documents available in the vector store and/or retrieved excerpts.

NON-NEGOTIABLE GROUNDING RULE:
- Use ONLY the text you retrieve from the uploaded rulings.
- Do NOT use general legal knowledge, memory, or assumptions.
- If the retrieved text does not contain an answer, say: "Not found in retrieved text." and ask a targeted follow-up.

RETRIEVAL WORKFLOW (FOLLOW THIS ORDER):
1) Retrieve: search for the most relevant excerpts first (especially headings like JUDGMENT/JUDGEMENT/RULING/OPINION and the judge surname + "JSC").
2) Verify: before you claim anything (submission/vote/outcome), confirm the exact wording exists in the retrieved text.
3) Quote: when you present a per-judge submission/opinion/conclusion/vote, include at least one direct quote copied verbatim.
4) If you cannot quote it verbatim, you must write: "Not found in retrieved text." (no paraphrase, no guessing).

JUDGE ATTRIBUTION SAFETY:
- Only attribute text to a judge if it appears inside that judge’s own opinion/judgment section.
- Do NOT misattribute citations like "opinion of Justice X in another case" as Justice X’s submission in the current case.

VOTES / OUTCOME SAFETY:
- Do NOT infer "dismissed/allowed/unanimous" unless the retrieved text explicitly states it and you can quote it.
- Do NOT infer how each judge voted unless the retrieved text explicitly indicates concurrence/dissent/allow/dismiss (quote required).

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

If the user asks for legal advice:
- Provide general information only from retrieved case law and state it is not legal advice.
""".strip()


def _load_vector_store_id_from_summary_file() -> Optional[str]:
    try:
        with open(VECTOR_STORE_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        vs_id = (data.get("vector_store_id") or "").strip()
        return vs_id or None
    except Exception:
        return None


def _find_vector_store_id_by_name(client: OpenAI, name: str) -> Optional[str]:
    try:
        page = client.vector_stores.list(limit=100)
    except Exception:
        return None

    for vs in getattr(page, "data", []) or []:
        if (getattr(vs, "name", "") or "") == name:
            return getattr(vs, "id", None)
    return None


# Hardcoded display labels for known documents.
# Key = exact filename as uploaded to the vector store.
# Value = short human-readable case label shown in the UI.
CASE_LABELS: dict[str, str] = {
    "doc_2.pdf": "2012 Presidential Election Petition (Akufo-Addo & Others v Mahama & Others)",
}


@st.cache_data(ttl=3600)
def _fetch_case_title_from_vs(api_key: str, vector_store_id: str, filename: str) -> Optional[str]:
    """Ask the vector store for the exact case title of the given document."""
    if not api_key or not vector_store_id or not filename:
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model="gpt-4o-mini",
            instructions=(
                "You are a legal document assistant. "
                "Retrieve the exact case name/title as it appears on the first page of the document. "
                "Return ONLY the case name, nothing else. "
                "If you cannot find it, return the filename."
            ),
            input=f"What is the exact case name or title in the document '{filename}'? Return only the case name.",
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id], "max_num_results": 5}],
            temperature=0,
        )
        title = (resp.output_text or "").strip()
        return title if title else None
    except Exception:
        return None


def _display_case_label(filename: str, api_key: str = "", vector_store_id: str = "") -> str:
    if filename == "(All cases)":
        return filename
    # 1. Check hardcoded map first
    if filename in CASE_LABELS:
        return CASE_LABELS[filename]
    # 2. Fall back to VS query
    if api_key and vector_store_id:
        title = _fetch_case_title_from_vs(api_key, vector_store_id, filename)
        if title:
            return title
    return filename


@st.cache_data(ttl=300)
def _list_vector_store_filenames(api_key_present: bool, vector_store_id: str) -> list[str]:
    if not api_key_present or not vector_store_id:
        return []

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())

    filenames: list[str] = []
    after = None
    # Keep this conservative; it’s just for a dropdown.
    for _ in range(5):
        kwargs: dict[str, Any] = {"vector_store_id": vector_store_id, "limit": 100}
        if after:
            kwargs["after"] = after
        page = client.vector_stores.files.list(**kwargs)
        items = getattr(page, "data", []) or []
        if not items:
            break

        for item in items:
            file_id = getattr(item, "file_id", None) or getattr(item, "id", None)
            if not file_id:
                continue
            try:
                fobj = client.files.retrieve(file_id)
                fn = (getattr(fobj, "filename", "") or "").strip()
                if fn:
                    filenames.append(fn)
            except Exception:
                continue

        if not bool(getattr(page, "has_more", False)):
            break
        after = getattr(items[-1], "id", None)
        if not after:
            break

    # De-dup + stable sort
    return sorted({f for f in filenames if f}, key=lambda s: s.lower())


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
                if getattr(ann, "type", None) != "file_citation":
                    continue
                citations.append(
                    {
                        "file_id": getattr(ann, "file_id", "") or "",
                        "quote": getattr(ann, "quote", "") or "",
                    }
                )

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for c in citations:
        key = (c.get("file_id", ""), c.get("quote", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def _mask_key(key: str) -> str:
    if not key:
        return "(missing)"
    return key[:7] + "…" + key[-4:] if len(key) > 12 else "(set)"


def main() -> None:
    st.set_page_config(page_title="The Law Student Assistant", page_icon="⚖️", layout="wide")

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None

    st.markdown("# ⚖️ The Law Student Assistant")
    st.caption("Research-and-answer chat for legal teams. Outputs may be incomplete; verify against primary sources.")

    if not api_key:
        st.warning("Set OPENAI_API_KEY in .env to enable chat.")

    vector_store_id = VECTOR_STORE_ID or _load_vector_store_id_from_summary_file()
    if not vector_store_id and client is not None:
        vector_store_id = _find_vector_store_id_by_name(client, VECTOR_STORE_NAME)

    retrieval_enabled = bool(api_key and vector_store_id)

    with st.sidebar:
        practice_area = st.selectbox("Practice area", ["Supreme Court"], index=0)
        jurisdiction = st.selectbox("Jurisdiction", ["Ghana"], index=0)

        st.divider()

        case_names: list[str] = []
        if retrieval_enabled:
            case_names = _list_vector_store_filenames(True, vector_store_id or "")

        dropdown = ["(All cases)"] + case_names if case_names else ["(All cases)"]

        if "selected_case" not in st.session_state:
            # Auto-select the first real case if only one exists, otherwise default to All cases
            st.session_state.selected_case = case_names[0] if len(case_names) == 1 else dropdown[0]

        st.markdown("**Cases**")
        st.selectbox(
            "Focused case",
            dropdown,
            key="selected_case",
            format_func=lambda fn: _display_case_label(fn, api_key, vector_store_id or ""),
        )

        if st.button("New chat"):
            st.session_state.messages = []
            st.rerun()

    selected_case = (st.session_state.get("selected_case") or "").strip()
    if selected_case and selected_case != "(All cases)":
        st.info(f"**Focused Case:** {_display_case_label(selected_case, api_key, vector_store_id or '')}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your matter…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if client is None:
        with st.chat_message("assistant"):
            st.markdown("API key missing. Add OPENAI_API_KEY to your .env.")
        return

    system = BASE_SYSTEM_PROMPT
    system += f"\n\nPractice area: {practice_area}" if practice_area else ""
    system += f"\nJurisdiction: {jurisdiction}" if jurisdiction else ""

    case_label = _display_case_label(selected_case, api_key, vector_store_id or "") if selected_case and selected_case != "(All cases)" else ""
    if case_label:
        system += (
            "\n\nFOCUS CASE:\n"
            f"- The user is asking about: {case_label}\n"
            f"- Internal document filename: {selected_case}\n"
            "- Search the vector store specifically for passages from this case.\n"
            "- If the user's question doesn't name a specific case, assume they mean this focused case.\n"
            "- If the user explicitly names a different case, follow the user."
        )

    file_search_tools = None
    if retrieval_enabled and vector_store_id:
        file_search_tools = [
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 20,
            }
        ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = None
            try:
                create_kwargs: dict[str, Any] = {}
                if file_search_tools:
                    create_kwargs["tools"] = file_search_tools

                # Build the messages list, augmenting the last user message with
                # the focused case label so the vector embedding targets the
                # actual case name rather than a raw filename.
                messages_for_api = list(st.session_state.messages)
                if case_label and messages_for_api and messages_for_api[-1]["role"] == "user":
                    last_user_content = messages_for_api[-1]["content"]
                    messages_for_api[-1] = {
                        "role": "user",
                        "content": f"[Case: {case_label}] {last_user_content}",
                    }

                resp = client.responses.create(
                    model=MODEL,
                    instructions=system,
                    input=messages_for_api,
                    temperature=0,
                    **create_kwargs,
                )
                answer = (resp.output_text or "").strip() or "(No response text.)"
            except Exception as e:
                answer = f"Request failed: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        if resp is not None and file_search_tools:
            citations = _extract_file_citations(resp)
            # Resolve file_id → human-readable case label
            file_id_to_name: dict[str, str] = {}
            for c in citations:
                fid = (c.get("file_id") or "").strip()
                if fid and fid not in file_id_to_name:
                    try:
                        fobj = client.files.retrieve(fid)
                        fn = (getattr(fobj, "filename", "") or "").strip()
                        file_id_to_name[fid] = _display_case_label(fn, api_key, vector_store_id or "") if fn else fid
                    except Exception:
                        file_id_to_name[fid] = fid

            if citations:
                with st.expander("Sources"):
                    for i, c in enumerate(citations, start=1):
                        fid = (c.get("file_id") or "").strip()
                        quote = (c.get("quote") or "").strip()
                        label = file_id_to_name.get(fid, fid)
                        st.markdown(f"**[{i}] {label}**")
                        if quote:
                            st.caption(f'"{quote}"')


if __name__ == "__main__":
    main()
