"""
Orlando Real Estate — A2A Multi-Agent Dashboard
Run: streamlit run dashboard.py
"""
import multiprocessing
multiprocessing.freeze_support()

import ssl, os, sys, subprocess, json, threading, time, base64, tempfile
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# ── SSL fix (Windows) ──────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "orlando-a2a"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))

MCP_SERVER_PATH = os.path.abspath("mcp_server.py")
client = OpenAI()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orlando RE · A2A",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#0f0f0f; --surface:#1a1a1a; --border:#2a2a2a;
    --accent:#c8a96e; --accent2:#7eb8a4; --text:#e8e4dc;
    --muted:#666; --green:#4caf7d; --red:#e05c5c;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.main-header{font-family:'DM Serif Display',serif;font-size:2.4rem;color:var(--accent);letter-spacing:-0.5px;margin-bottom:0.1rem;}
.sub-header{font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:2rem;}
.agent-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:0.78rem;}
.agent-card.active{border-color:var(--accent);background:#201e18;}
.agent-card.done{border-color:var(--green);background:#141a16;}
.agent-card.error{border-color:var(--red);background:#1a1414;}
.agent-name{font-weight:500;color:var(--accent);font-size:0.82rem;margin-bottom:4px;}
.agent-port{color:var(--muted);font-size:0.7rem;}
.agent-result{color:var(--accent2);font-size:0.75rem;margin-top:6px;word-break:break-word;}
.flow-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;font-family:'DM Mono',monospace;font-size:0.78rem;color:var(--text);white-space:pre;margin-bottom:1rem;}
.report-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:24px 28px;margin-top:1rem;}
.hitl-bar{background:#1a1a12;border:1px solid var(--accent);border-radius:8px;padding:16px 20px;margin:1rem 0;font-family:'DM Mono',monospace;font-size:0.8rem;color:var(--accent);}
div[data-testid="stTextArea"] textarea{background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-size:0.9rem!important;}
div[data-testid="stTextInput"] input{background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-family:'DM Mono',monospace!important;font-size:0.85rem!important;}
.stButton>button{background:var(--accent)!important;color:#0f0f0f!important;border:none!important;border-radius:6px!important;font-family:'DM Mono',monospace!important;font-size:0.8rem!important;font-weight:500!important;letter-spacing:0.5px!important;padding:8px 20px!important;}
.stButton>button:hover{background:#d4b87a!important;}
[data-testid="stMarkdownContainer"]{color:var(--text)!important;}
h1,h2,h3{font-family:'DM Serif Display',serif!important;color:var(--accent)!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ── MCP client ─────────────────────────────────────────────────────────────
def call_mcp_tool(tool_name, params):
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "a2a", "version": "1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": tool_name, "arguments": {"params": params}}},
    ]
    stdin_data = "\n".join(json.dumps(m) for m in msgs) + "\n"
    result = subprocess.run(
        [sys.executable, MCP_SERVER_PATH],
        input=stdin_data.encode(),
        capture_output=True,
        timeout=60
    )
    stdout = result.stdout.decode().strip()
    stderr = result.stderr.decode().strip()

    for line in stdout.split("\n"):
        try:
            msg = json.loads(line)
            content = msg.get("result", {}).get("content", [])
            if content:
                text = content[0].get("text", "")
                # Check if it's an error response from MCP
                if msg.get("result", {}).get("isError"):
                    return {"error": text[:300]}
                try:
                    return json.loads(text)
                except:
                    return {"text": text}
        except:
            continue
    return {"error": f"No valid response. stderr: {stderr[:200]}"}

# ── Agent cards ────────────────────────────────────────────────────────────
AGENT_CARDS = {
    "zoning": {
        "name": "zoning", "icon": "🏛",
        "description": "Answers questions about Orlando zoning laws, setbacks, permits, and classifications using RAG over the municipal code.",
        "tool": "zoning_law_query", "port": 8001,
        "params_schema": {"query": "str", "top_k": "int default 5", "similarity_threshold": "float default 0.7"},
        "when_to_use": "zoning laws, setbacks, permits, land use classifications",
    },
    "fmv": {
        "name": "fmv", "icon": "💰",
        "description": "Predicts fair market value of an Orlando property using XGBoost trained on Orange County data.",
        "tool": "predict_fair_market_value", "port": 8002,
        "params_schema": {
            "latitude": "float", "longitude": "float",
            "living_area": "float (sqft)", "land_sqft": "float",
            "age": "int (years since built)", "structure_quality": "float 1-5",
            "SPEC_FEAT_VAL": "float default 0",
            "rail_dist": "float meters default 5000",
            "ocean_dist": "float meters default 100000",
            "water_dist": "float meters default 2000",
            "center_dist": "float meters default 15000",
            "subcenter_dist": "float meters default 5000",
            "highway_dist": "float meters default 1000",
            "month_sold": "int 1-12 default 6",
            "avno60plus": "int 0 or 1 default 0"
        },
        "when_to_use": "property valuation, fair market value, price estimate",
    },
    "walkability": {
        "name": "walkability", "icon": "🚶",
        "description": "Scores walkability 0-100 and lists nearby amenities via OpenStreetMap.",
        "tool": "assess_walkability", "port": 8003,
        "params_schema": {"latitude": "float", "longitude": "float", "radius_miles": "float default 1.0"},
        "when_to_use": "walkability, nearby amenities, transit, schools, restaurants",
    },
    "vision": {
        "name": "vision", "icon": "👁",
        "description": "Assesses property damage from images (analyze mode) or searches similar damage cases in vector DB (search mode).",
        "tool": "property_damage_assessment", "port": 8004,
        "params_schema": {"mode": "search or analyze", "search_query": "str (search mode)", "image_path": "str (analyze mode)", "top_k": "int default 3"},
        "when_to_use": "property damage, roof damage, water damage, structural issues",
    },
    "market_expert": {
        "name": "market_expert", "icon": "🎓",
        "description": "Fine-tuned LLM expert on Orlando neighborhoods, pricing trends, and investment insights.",
        "tool": "orlando_market_expert", "port": 8006,
        "params_schema": {"query": "str", "temperature": "float default 0.7", "max_tokens": "int default 500"},
        "when_to_use": "market trends, investment advice, neighborhood insights, price appreciation",
    },
}

# ── Agent HTTP servers (plain http.server, no uvicorn/FastAPI) ─────────────
@st.cache_resource
def start_agents():
    def make_handler(agent_name, tool_name):
        card = AGENT_CARDS[agent_name]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({"status": "ok", "agent": agent_name}).encode()
                elif self.path == "/agent-card":
                    body = json.dumps(card).encode()
                else:
                    body = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                req = json.loads(raw)
                result = call_mcp_tool(tool_name, req["params"])
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        return Handler

    for name, card in AGENT_CARDS.items():
        handler = make_handler(name, card["tool"])
        try:
            server = HTTPServer(("127.0.0.1", card["port"]), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except OSError:
            pass  # port already in use from previous run — that's fine

    time.sleep(1)

    status = {}
    for name, card in AGENT_CARDS.items():
        try:
            r = requests.get(f'http://127.0.0.1:{card["port"]}/health', timeout=3)
            status[name] = r.json().get("status") == "ok"
        except:
            status[name] = False
    return status

# ── Core functions ─────────────────────────────────────────────────────────
def supervisor(user_query, image_attached=False):
    agent_descriptions = json.dumps(
        {name: {k: v for k, v in card.items() if k in ("description", "when_to_use", "params_schema")}
         for name, card in AGENT_CARDS.items()},
        indent=2
    )
    image_hint = "\nNOTE: The user has attached a property image — include 'vision' agent with mode='analyze'." if image_attached else ""
    prompt = (
        "You are a supervisor for an Orlando Real Estate AI system.\n"
        "Available agents:\n\n" + agent_descriptions +
        "\n\nFor FMV, always include ALL required params. Use these defaults if not specified by user: "
        "SPEC_FEAT_VAL=0, rail_dist=5000, ocean_dist=100000, water_dist=2000, center_dist=15000, "
        "subcenter_dist=5000, highway_dist=1000, month_sold=6, avno60plus=0.\n"
        + image_hint +
        "\nIf the query involves illegal discrimination (race, religion, national origin) "
        'respond ONLY with: {"refused": true, "reason": "..."}\n'
        "Otherwise respond ONLY as JSON (no markdown):\n"
        '{"agents": ["fmv"], "params": {"fmv": {"latitude": 28.55, "longitude": -81.35, "living_area": 1800, "land_sqft": 7500, "age": 20, "structure_quality": 3.0, "SPEC_FEAT_VAL": 0, "rail_dist": 5000, "ocean_dist": 100000, "water_dist": 2000, "center_dist": 15000, "subcenter_dist": 5000, "highway_dist": 1000, "month_sold": 6, "avno60plus": 0}}}'
    )
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_query}]
    )
    raw = response.choices[0].message.content.strip().strip("```json").strip("```").strip()
    if not raw:
        return None
    parsed = json.loads(raw)
    if parsed.get("refused"):
        return {"refused": True, "reason": parsed["reason"]}
    return parsed

def dispatch(plan, image_path=None):
    results = {}
    for agent in plan["agents"]:
        port   = AGENT_CARDS[agent]["port"]
        params = plan["params"][agent]
        # Inject image path for vision agent if provided
        if agent == "vision" and image_path:
            params["mode"] = "analyze"
            params["image_path"] = image_path
        r = requests.post(f"http://127.0.0.1:{port}/invoke", json={"params": params}, timeout=60)
        results[agent] = r.json()
    return results

def synthesize(user_query, results, feedback=""):
    query = user_query
    system = "You are a real estate analyst. Write a concise well-structured Markdown report answering the user query. Cite which agent provided each insight."
    
    if feedback:
        query = user_query + f"\n\nHuman reviewer feedback (YOU MUST FOLLOW THIS EXACTLY): {feedback}"
        system = (
            "You are a real estate analyst. The human has reviewed your previous report and provided specific feedback. "
            "You MUST rewrite the report following their feedback precisely — change the structure, tone, focus, and length as instructed. "
            "Do not produce the same report with minor tweaks. The feedback is a directive, not a suggestion."
        )
    
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.4,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"User query: {query}\n\nAgent results:\n{json.dumps(results, indent=2)}"}
        ]
    )
    return response.choices[0].message.content

def result_summary(agent, result):
    if "error" in result:
        return f"✗ {result['error'][:80]}"
    elif agent == "fmv":
        fmv = result.get("predicted_fmv")
        return f"${fmv:,.2f}" if isinstance(fmv, (int, float)) else str(fmv)
    elif agent == "walkability":
        return f"{result.get('walkability_score', '?')}/100 — {result.get('interpretation', '')}"
    elif agent == "zoning":
        return f"{result.get('answer', '')[:120]}..."
    elif agent == "vision":
        return f"severity: {result.get('severity', '?')}/10 | {result.get('damage_type', '')}"
    elif agent == "market_expert":
        return f"{result.get('response', '')[:120]}..."
    return str(result)[:120]

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("stage", "idle"),
    ("plan", None),
    ("results", {}),
    ("report", ""),
    ("agent_status", {}),
    ("query", ""),
    ("image_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent_health = start_agents()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-header">Orlando RE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A2A · Multi-Agent System</div>', unsafe_allow_html=True)

    st.markdown("**Agent Registry**")
    for name, card in AGENT_CARDS.items():
        healthy = agent_health.get(name, False)
        status  = st.session_state.agent_status.get(name, "idle")
        if status == "running":
            css_class, dot = "active", "🟡"
        elif status == "done":
            css_class, dot = "done", "🟢"
        elif status == "error":
            css_class, dot = "error", "🔴"
        else:
            css_class, dot = "", "🟢" if healthy else "⚫"

        result_html = ""
        if status == "done" and name in st.session_state.results:
            summary = result_summary(name, st.session_state.results[name])
            result_html = f'<div class="agent-result">{summary}</div>'

        st.markdown(f"""
        <div class="agent-card {css_class}">
            <div class="agent-name">{dot} {card['icon']} {name.upper()}</div>
            <div class="agent-port">:{card['port']} · {card['tool']}</div>
            {result_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Example queries**")
    examples = {
        "🌳 Tree permit + FMV": "What size tree requires a permit in Orlando, and what is the fair market value for a property at lat 28.522, lon -81.418, 1950 sqft, built 1998, 8200 sqft lot, quality 3.5?",
        "🏘 Investment advice": "Is the Dr. Phillips neighborhood in Orlando a good investment right now? What are current market trends?",
        "🔍 Full analysis": "Complete analysis for lat 28.541, lon -81.396, 2200 sqft, built 2005, quality 4.0, 9500 sqft lot. Need FMV, zoning, walkability, and market outlook.",
        "🚫 Guardrail test": "Are there any neighborhoods in Orlando I should avoid because of the racial demographics?",
    }
    for label, q in examples.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state.query = q
            st.session_state.stage = "idle"
            st.session_state.plan = None
            st.session_state.results = {}
            st.session_state.report = ""
            st.session_state.agent_status = {}
            st.session_state.image_path = None
            st.rerun()

# ── Main panel ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Property Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Agent-to-Agent · Orlando Real Estate</div>', unsafe_allow_html=True)

query = st.text_area(
    "Query", value=st.session_state.query, height=100,
    placeholder="Ask about zoning, fair market value, walkability, damage assessment, or market trends...",
    label_visibility="collapsed"
)

# ── Image upload for vision agent ──────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📎 Attach property image for damage assessment (optional)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)

image_path = None
if uploaded_file:
    suffix = "." + uploaded_file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    image_path = tmp.name
    st.image(uploaded_file, caption="Attached image — vision agent will analyze this", width=300)

col1, col2 = st.columns([1, 6])
with col1:
    run_clicked = st.button("▶ Run", use_container_width=True)
with col2:
    if st.session_state.stage != "idle":
        if st.button("↺ Reset", use_container_width=False):
            st.session_state.stage = "idle"
            st.session_state.plan = None
            st.session_state.results = {}
            st.session_state.report = ""
            st.session_state.agent_status = {}
            st.session_state.image_path = None
            st.rerun()

st.markdown("---")

# ── Run pipeline ───────────────────────────────────────────────────────────
if run_clicked and query.strip():
    st.session_state.query = query
    st.session_state.image_path = image_path
    st.session_state.stage = "routing"
    st.session_state.plan = None
    st.session_state.results = {}
    st.session_state.report = ""
    st.session_state.agent_status = {}

if st.session_state.stage == "routing":
    with st.spinner("Supervisor reading agent cards and routing..."):
        plan = supervisor(query, image_attached=bool(st.session_state.image_path))
    if plan is None or plan.get("refused"):
        st.session_state.stage = "refused"
        st.session_state.plan = plan
    else:
        st.session_state.plan = plan
        st.session_state.agent_status = {a: "idle" for a in plan["agents"]}
        st.session_state.stage = "dispatching"
    st.rerun()

if st.session_state.stage == "dispatching":
    plan   = st.session_state.plan
    agents = plan["agents"]

    flow_lines = ["  SUPERVISOR"]
    for i, agent in enumerate(agents):
        port = AGENT_CARDS[agent]["port"]
        connector = "  ├──" if i < len(agents) - 1 else "  └──"
        flow_lines.append(f"{connector} HTTP POST ──► [{agent.upper()}] :{port}/invoke")
    st.markdown(f'<div class="flow-box">{chr(10).join(flow_lines)}</div>', unsafe_allow_html=True)

    results = {}
    for agent in agents:
        st.session_state.agent_status[agent] = "running"
        with st.spinner(f"Calling [{agent.upper()}] agent..."):
            port   = AGENT_CARDS[agent]["port"]
            params = plan["params"][agent]
            if agent == "vision" and st.session_state.image_path:
                params["mode"] = "analyze"
                params["image_path"] = st.session_state.image_path
            try:
                r = requests.post(f"http://127.0.0.1:{port}/invoke", json={"params": params}, timeout=60)
                results[agent] = r.json()
                st.session_state.agent_status[agent] = "done" if "error" not in results[agent] else "error"
            except Exception as e:
                results[agent] = {"error": str(e)}
                st.session_state.agent_status[agent] = "error"

    st.session_state.results = results
    st.session_state.stage = "synthesizing"
    st.rerun()

if st.session_state.stage == "synthesizing":
    with st.spinner("Synthesizer generating report..."):
        report = synthesize(query, st.session_state.results)
    st.session_state.report = report
    st.session_state.stage = "hitl"
    st.rerun()

if st.session_state.stage == "hitl":
    st.markdown("**Report**")
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="hitl-bar">⚑ HUMAN REVIEW — Approve this report or provide feedback to regenerate</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("✓ Approve", use_container_width=True):
            st.session_state.stage = "done"
            st.rerun()
    with col_b:
        feedback = st.text_area("Feedback", placeholder="e.g. Focus more on investment risk...", height=100, key="feedback_input")
        if st.button("↺ Regenerate with feedback", use_container_width=True) and feedback.strip():
            with st.spinner("Regenerating..."):
                st.session_state.report = synthesize(st.session_state.query, st.session_state.results, feedback)
            st.rerun()

if st.session_state.stage == "done":
    st.success("✓ Report approved")
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.stage == "refused":
    reason = st.session_state.plan.get("reason", "") if st.session_state.plan else ""
    st.error(f"🚫 Query refused by supervisor: {reason}")
