#DAG-LLM Pipeline (Reference Release)
From Fuzzy Worlds to Causal Graphs
This repository provides a minimal reference implementation of the DAG-LLM Pipeline — a method for converting unstructured text into causal directed acyclic graphs (DAGs). It demonstrates how large language models (LLMs), which encode fuzzy and probabilistic human world models, can be transformed into explicit, explainable, and structured causal graphs.
The codebase is designed for reproducibility and research.
________________


✨ Features
* Backend Service (FastAPI): Ingest journal text, extract entities & relations with GPT, and store results in SQLite.

* Graph Models: Structured database schema for agents, entities, edges, and journal entries.

* Visualization: Example DAGs rendered with NetworkX + Matplotlib.

* Frontend (ReactFlow demo): Minimal graph viewer in App.js.

* Open Reference Implementation: Lightweight, extensible, and easy to adapt for research.

________________


📂 Repository Structure
.
├── api/
│   ├── db.py                # Database schema (Agent, Entity, Edge, JournalEntry)
│   └── gpt_maindb.py        # FastAPI service for ingestion + DAG extraction
├── create_db.py             # Initialize SQLite database
├── send_journal_gpt_db.py   # Example script: ingest text + visualize DAG
└── sample_journal.txt       # Example text for testing


________________


🚀 Getting Started
1. Clone Repository
git clone https://github.com/your-username/dag-llm-pipeline.git
cd dag-llm-pipeline


2. Set Up Backend
Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Initialize the database:
python create_db.py


Run the FastAPI backend:
uvicorn api.gpt_maindb:app --reload --port 8001


3. Set OpenAI API Key
Export your API key (or edit .env):
export OPENAI_API_KEY=your_api_key_here


4. Visualize a Sample Journal
Run the demo script:
python send_journal_gpt_db.py


This will:
   * Ingest sample_journal.txt

   * Store entities & edges in SQLite

   * Render the resulting DAG with NetworkX/Matplotlib


📖 Example Output
Given the input text:
“I feel nervous because I failed my last interview. I want a strategy to increase confidence.”
The pipeline extracts nodes and edges such as:
Nodes: ["I", "nervousness", "last_interview_failure", "confidence", "strategy"]
Edges:
- last_interview_failure → nervousness (causes)
- nervousness → confidence (reduces)
- strategy → confidence (increases)


And renders a DAG showing the causal structure.
