"""
Web UI for the financial RAG system.

Run from the project root:
    streamlit run ui.py
Then open the address it prints (usually http://localhost:8501).
"""
import os
import re

import streamlit as st

from app import answer_question, COMPANY_INDEX
from src.config import SETTINGS, PROMPT_VERSION
from src.highlight import highlighted_pages

PDF_DIR = "data/production_baseline"

EXAMPLES = [
    "What are the primary risk factors related to supply chain disruptions for Cisco and FedEx?",
    "What was Cisco's total revenue for fiscal 2026?",
    "How does seasonality affect the results of Estée Lauder and Intuit?",
    "What was Tesla's total revenue in fiscal 2026?",
]

st.set_page_config(page_title="10-K Research Assistant", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📊 10-K Research Assistant")
    st.caption("Answers questions from SEC 10-K filings. Every claim is cited to a "
               "filing and page; if the filings don't support an answer, it refuses.")

    st.subheader("Companies")
    for entry in sorted(COMPANY_INDEX.values(), key=lambda e: e["name"].lower()):
        st.markdown(f"- {entry['name']}")

    st.subheader("Settings")
    llm = SETTINGS["llm"]
    model = llm.get("cohere_model") if llm.get("provider") == "cohere" else llm.get("model")
    st.markdown(
        f"- LLM: `{llm.get('provider')}` / `{model}`\n"
        f"- Prompt version: `{PROMPT_VERSION}`\n"
        f"- Chunks per question: `{SETTINGS['retrieval']['final_k']}`\n"
        f"- Min relevance: `{SETTINGS['enforcement']['min_relevance']}`"
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ------------------------------------------------------------------ one answer
def split_label(label):
    """'csco.pdf | Cisco Systems, Inc. | pp. 22-23' -> (file, company, pages)"""
    parts = [p.strip() for p in label.split("|")]
    return (parts + ["", "", ""])[:3]


def page_range(pages):
    """'pp. 22-23' -> (22, 23); 'p. 61' -> (61, 61); 'page n/a' -> None"""
    numbers = [int(n) for n in re.findall(r"\d+", pages)]
    if not numbers:
        return None
    return numbers[0], numbers[-1]


def claims_citing(answer, ref):
    """The answer lines that cite excerpt [ref]."""
    pattern = re.compile(rf"\[(?:\d+\s*,\s*)*{ref}(?:\s*,\s*\d+)*\]")
    return tuple(line for line in answer.splitlines() if pattern.search(line))


@st.cache_data(show_spinner=False)
def render_highlights(pdf_path, first, last, chunk_text, claims, company):
    return highlighted_pages(pdf_path, first, last, chunk_text, list(claims), company)


def show_source_in_pdf(file, company, pages, chunk_text, claims):
    pdf_path = os.path.join(PDF_DIR, file)
    span = page_range(pages)
    if not os.path.exists(pdf_path) or span is None:
        st.caption("The original PDF or its page numbers are not available for this source.")
        return
    sentences, images = render_highlights(pdf_path, span[0], span[1], chunk_text, claims, company)
    if not sentences:
        st.caption("Could not pinpoint the sentence; showing the first cited page.")
    for page_no, png in images:
        st.image(png, caption=f"{file}, page {page_no}", use_container_width=True)


def show_result(result):
    if result["refused"]:
        st.warning(result["answer"], icon="🚫")
        reasons = {
            "no_evidence": "No retrieved passage was relevant enough to use as evidence.",
            "no_answer_in_excerpts": "The retrieved passages don't contain the answer.",
            "failed_checks": "Most claims in the generated answer failed citation checks.",
        }
        st.caption(reasons.get(result["reason"], ""))
    else:
        st.markdown(result["answer"])

    if result["removed"]:
        with st.expander(f"⚠️ {len(result['removed'])} claim(s) removed by citation checks"):
            for claim, reason in result["removed"]:
                st.markdown(f"- **{reason}:** {claim}")

    if result["sources"]:
        st.markdown("**Sources**")
        for s in result["sources"]:
            file, company, pages = split_label(s["label"])
            relevance = f"{s['relevance']:.2f}" if isinstance(s["relevance"], (int, float)) else "n/a"
            with st.expander(f"[{s['ref']}]  {company} · {file} · {pages} · relevance {relevance}"):
                # contexts are in the same order as the numbered excerpts the LLM saw
                chunk_text = result["contexts"][s["ref"] - 1]
                in_pdf, as_text = st.tabs(["Highlighted in PDF", "Excerpt text"])
                with in_pdf:
                    show_source_in_pdf(file, company, pages, chunk_text,
                                       claims_citing(result["answer"], s["ref"]))
                with as_text:
                    st.text(chunk_text)


# ------------------------------------------------------------------ chat
if "history" not in st.session_state:
    st.session_state.history = []   # list of (question, result)

st.title("Ask the filings")

if not st.session_state.history:
    st.markdown("Try an example:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        show_result(result)

question = st.chat_input("Ask about the companies' 10-K filings...") or st.session_state.pop("pending", None)

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the filings and checking citations..."):
            try:
                result = answer_question(question)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        show_result(result)
    st.session_state.history.append((question, result))