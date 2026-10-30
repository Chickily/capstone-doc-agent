import streamlit as st

from src import guardrails as gr
from src.agents import AgentOrchestrator
from src.config import get_secret, settings
from src.ingestion import process_upload
from src.llm import LLM
from src.pipeline import simple_rag
from src.vector_store import VectorStore

st.set_page_config(page_title="Document Agent", layout="wide")

# ---------- session state (survives Streamlit reruns) ----------
ss = st.session_state
if "store" not in ss:
    ss.store = VectorStore()
for key, default in {"tables": {}, "indexed": {}, "messages": [], "q_count": 0}.items():
    ss.setdefault(key, default)

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    st.caption(f"Provider: {settings.provider} | Model: {settings.model}")
    user_key = st.text_input("Your API key (optional)", type="password",
                             help="Leave blank to use the app's shared key (limited questions).")
    api_key = user_key or get_secret(settings.api_key_name)
    using_shared_key = not user_key

    mode = st.radio("Answer mode", ["Agent (plan, retrieve, validate)", "Simple RAG"])
    show_trace = st.checkbox("Show agent trace", value=True)

    st.header("Documents")
    files = st.file_uploader("PDF, TXT, CSV, or Excel (.xlsx)", type=["pdf", "txt", "csv", "xlsx"],
                             accept_multiple_files=True)
    if st.button("Index documents", type="primary", disabled=not files):
        for f in files:
            key = f"{f.name}:{f.size}"
            if key in ss.indexed:
                continue
            try:
                gr.validate_upload(f.name, f.size)
                with st.spinner(f"Indexing {f.name}..."):
                    chunks, tables = process_upload(f.name, f.getvalue())
                    ss.store.add_chunks(chunks)
                    ss.tables.update(tables)
                    ss.indexed[key] = len(chunks)
                st.success(f"{f.name}: {len(chunks)} chunks")
            except (gr.ValidationError, ValueError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Failed to process {f.name}: {type(e).__name__}")

    if ss.indexed:
        st.caption(f"{len(ss.indexed)} file(s) | {ss.store.count()} chunks indexed")

    if st.button("Clear chat"):
        ss.messages = []
        st.rerun()

# ---------- main area ----------
st.title("Enterprise Document Agent")
st.caption("Upload documents, then ask questions. Answers are grounded in your files with citations.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + " | ".join(m["sources"]))
        if m.get("trace") and show_trace:
            with st.expander("Agent trace"):
                st.json(m["trace"])

if prompt := st.chat_input("Ask a question about your documents"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            question = gr.validate_question(prompt)
            if not api_key:
                raise gr.ValidationError("No API key available. Enter one in the sidebar.")
            if ss.store.count() == 0:
                raise gr.ValidationError("Upload and index at least one document first.")
            if using_shared_key and ss.q_count >= settings.max_questions_shared_key:
                raise gr.ValidationError("Shared-key question limit reached. Enter your own API key in the sidebar.")

            # last 6 turns as plain-text history (must start with a user turn)
            history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-6:]]
            while history and history[0]["role"] != "user":
                history.pop(0)

            llm = LLM(api_key, settings.model, settings.provider)
            with st.spinner("Thinking..."):
                if mode.startswith("Agent"):
                    result = AgentOrchestrator(llm, ss.store, ss.tables).run(question, history)
                    answer, sources = result["answer"], result["sources"]
                    trace = {"plan": result["plan"], "tool_calls": result["tool_calls"],
                             "validation": result["validation"]}
                else:
                    answer, sources = simple_rag(llm, ss.store, question)
                    trace = None

            ss.q_count += 1
            st.markdown(answer)
            if sources:
                st.caption("Sources: " + " | ".join(sources))
            if trace and show_trace:
                with st.expander("Agent trace"):
                    st.json(trace)

            ss.messages.append({"role": "user", "content": prompt})
            ss.messages.append({"role": "assistant", "content": answer, "sources": sources, "trace": trace})
        except gr.ValidationError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(gr.friendly_error(e))
