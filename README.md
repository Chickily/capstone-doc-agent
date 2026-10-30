# Enterprise Document Agent (RAG + Agentic AI)
Upload PDF / TXT / CSV / Excel files and ask questions. A Planner -> Researcher -> Validator
agent workflow retrieves evidence from ChromaDB and answers with Claude, with citations.
## Run locally (macOS, VS Code)
1. python3.13 -m venv edureka_rag_project and select the interpreter in VS Code
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. Copy .env.example to .env and add ANTHROPIC_API_KEY
5. streamlit run app.py