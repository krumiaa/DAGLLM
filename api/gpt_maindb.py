# gpt_maindb_min.py  — white-paper minimal
from fastapi import FastAPI
from pydantic import BaseModel
import os, json
from openai import OpenAI

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "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")

app = FastAPI()

API_KEY = os.getenv("SOUL_MEMORY_API_KEY") or "supersecretdevkey"

SYSTEM_PROMPT = """
You are a causal extraction agent. Transform unstructured text into a **causal world model (DAG)** that shows how events, emotions, goals, and strategies are connected. 
Focus on the central observer ("I") as the hub of the graph.

### RULES

1. **Central Observer (CO)**
   - If text is first-person, set CO = "I".
   - Nearly every node must connect to CO within 2 hops.

2. **Entities (Nodes)**
   - Include only meaningful events, emotions, states, goals, strategies, and behaviors.
   - Examples: "job_interview", "last_interview_failure", "nervousness", "confidence", "calmness", "strategy".
   - Each node: { "name": ..., "sentiment_type": ..., "sentiment_value": ..., "tags": [...] }
   - Use tags from ["event","emotion","state","goal","behavior","concept","agent"].

3. **Relationships (Edges)**
   - Show causal/affective influence between nodes.
   - Use only the finite relations: ["causes","reduces","increases","enables","improves","plans_for","results_in_sentiment_change","influences"].
   - Each edge: { "source": ..., "target": ..., "relation": ..., "confidence": int(0–100), "sentiment_type": ..., "sentiment_value": ..., "tags": [...], "event_time_iso": null, "narrative_order": N }

4. **Coverage**
   - Always include explicit causal statements (e.g., "failed last interview → nervousness").
   - Infer plausible links when implied (e.g., "nervousness reduces confidence").
   - Connect strategies or interventions to their intended effects.

5. **Simplify**
   - Do NOT create sentence nodes or attention loops.
   - Do NOT decompose into ultra-micro steps unless necessary.
   - Output should be a compact causal DAG (5–12 nodes typical).


### OUTPUT FORMAT (EXACT)
Return only JSON shaped like:
{
  "nodes": ["I","job_interview","last_interview_failure","nervousness","confidence","calmness","strategy","answer_weakness_questions"],
  "edges": [
    {"source":"last_interview_failure","target":"nervousness","relation":"causes"},
    {"source":"nervousness","target":"confidence","relation":"reduces"},
    {"source":"nervousness","target":"calmness","relation":"reduces"},
    {"source":"strategy","target":"confidence","relation":"increases"},
    {"source":"strategy","target":"calmness","relation":"enables"},
    {"source":"strategy","target":"answer_weakness_questions","relation":"improves"},
    {"source":"I","target":"job_interview","relation":"plans_for"},
    {"source":"job_interview","target":"I","relation":"results_in_sentiment_change"}
  ],

  "node_meta": {
    "I": {"tags":["agent"]},
    "job_interview": {"tags":["event"]},
    "last_interview_failure": {"tags":["event"]},
    "nervousness": {"tags":["emotion"], "sentiment_type":"Anxiety","sentiment_value":0.7},
    "confidence": {"tags":["state"]},
    "calmness": {"tags":["state"]},
    "strategy": {"tags":["concept"]},
    "answer_weakness_questions": {"tags":["behavior"]}
  },

  "edge_meta": [
    {"source":"last_interview_failure","target":"nervousness","confidence":90},
    {"source":"nervousness","target":"confidence","confidence":80},
    {"source":"nervousness","target":"calmness","confidence":80},
    {"source":"strategy","target":"confidence","confidence":85},
    {"source":"strategy","target":"calmness","confidence":85},
    {"source":"strategy","target":"answer_weakness_questions","confidence":80},
    {"source":"I","target":"job_interview","confidence":70},
    {"source":"job_interview","target":"I","confidence":70}
  ]
}

"""

class AgentTextRequest(BaseModel):
    user_id: str | None = None  # optional in the minimal version
    text: str

def extract_entities_and_relations(text: str) -> dict:
    system_prompt = SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":f"Extract the complete structured entity and causal relationship graph from:\n\n{text}"}],
        temperature=0.35,
        max_tokens=8192,
        response_format={"type":"json_object"},
        seed=7
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        # last-balanced truncation (same as your current helper)
        stack, last_ok = [], None
        for i,ch in enumerate(raw):
            if ch in "{[": stack.append(ch)
            elif ch in "}]":
                if stack and ((stack[-1]=="{" and ch=="}") or (stack[-1]=="[" and ch=="]")):
                    stack.pop()
                    if not stack: last_ok = i
        return json.loads(raw[:last_ok+1]) if last_ok is not None else {"nodes": [], "edges": []}

def normalize_sidecar_dag(dag: dict) -> dict:
    node_meta = dag.get("node_meta", {}) or {}
    norm_nodes = []
    for n in dag.get("nodes", []):
        if isinstance(n, str):
            meta = node_meta.get(n, {})
            norm_nodes.append({"name": n, "tags": meta.get("tags", []) or [],
                               "sentiment_type": meta.get("sentiment_type"),
                               "sentiment_value": meta.get("sentiment_value")})
        elif isinstance(n, dict):
            n.setdefault("tags", [])
            norm_nodes.append(n)
    norm_edges = []
    for e in dag.get("edges", []):
        if isinstance(e, dict):
            norm_edges.append({
                "source": e.get("source"),
                "target": e.get("target"),
                "relation": e.get("relation"),
                "confidence": e.get("confidence"),
                "tags": e.get("tags", []) or [],
                "sentiment_type": e.get("sentiment_type"),
                "sentiment_value": e.get("sentiment_value"),
                "event_time_iso": e.get("event_time_iso"),
                "narrative_order": e.get("narrative_order"),
            })
    return {"nodes": norm_nodes, "edges": norm_edges}

@app.post("/store_agent_text")
async def store_agent_text(req: AgentTextRequest):
    dag = extract_entities_and_relations(req.text)
    dag = normalize_sidecar_dag(dag)
    return {"status": "stored", "nodes": dag.get("nodes", []), "edges": dag.get("edges", [])}
