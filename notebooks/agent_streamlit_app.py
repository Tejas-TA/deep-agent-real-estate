"""
Orlando Real Estate AI — Streamlit UI  v3
Drop next to agent.py + mcp_server.py, then:  streamlit run streamlit_app.py
"""

import asyncio, json, threading, os
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Orlando RE · AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════
for k, v in {
    "messages": [],
    "agent_ready": False,
    "agent": None,
    "tools": [],
    "dark_mode": True,
    "auto_send": False,
    "chat_input_value": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
# THEME  — CSS vars driven by dark_mode toggle
# ══════════════════════════════════════════════════════════
D = st.session_state.dark_mode

THEME = {
    "bg":         "#0d0f14"  if D else "#f8fafc",
    "sidebar_bg": "#12151c"  if D else "#ffffff",
    "sidebar_bdr":"#1e2433"  if D else "#e2e8f0",
    "card_bg":    "#12151c"  if D else "#ffffff",
    "card_bdr":   "#1e2433"  if D else "#e2e8f0",
    "text":       "#e2e8f0"  if D else "#0f172a",
    "muted":      "#64748b"  if D else "#94a3b8",
    "accent":     "#4ade80"  if D else "#16a34a",
    "accent_bg":  "#052e16"  if D else "#f0fdf4",
    "accent_bdr": "#14532d"  if D else "#bbf7d0",
    "blue":       "#60a5fa"  if D else "#2563eb",
    "orange":     "#fb923c"  if D else "#ea580c",
    "hero_bg":    "linear-gradient(135deg,#0f172a 0%,#1e293b 100%)" if D
                  else "linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%)",
    "hero_bdr":   "#1e2d40"  if D else "#bbf7d0",
    "user_bg":    "#1e293b"  if D else "#f1f5f9",
    "user_bdr":   "#334155"  if D else "#cbd5e1",
    "ai_bg":      "#0f1e14"  if D else "#f0fdf4",
    "ai_bdr":     "#166534"  if D else "#86efac",
    "ai_text":    "#dcfce7"  if D else "#14532d",
    "trace_bg":   "#0c1220"  if D else "#f8fafc",
    "trace_bdr":  "#1e2d40"  if D else "#e2e8f0",
    "input_bg":   "#12151c"  if D else "#ffffff",
    "input_bdr":  "#1e2433"  if D else "#cbd5e1",
    "btn_bg":     "#12151c"  if D else "#ffffff",
    "btn_bdr":    "#1e2433"  if D else "#e2e8f0",
    "btn_text":   "#94a3b8"  if D else "#475569",
    "tile_bg":    "#12151c"  if D else "#f8fafc",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {THEME['bg']};
    color: {THEME['text']};
}}
section[data-testid="stSidebar"] {{
    background: {THEME['sidebar_bg']};
    border-right: 1px solid {THEME['sidebar_bdr']};
}}
.main .block-container {{
    background: {THEME['bg']};
    padding-top: 1.5rem;
    max-width: 1120px;
}}

.hero {{
    background: {THEME['hero_bg']};
    border: 1px solid {THEME['hero_bdr']};
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 1.2rem;
}}
.hero-icon {{ font-size: 2.4rem; line-height:1; }}
.hero-title {{
    font-family: 'Space Mono', monospace; font-size: 1.4rem;
    font-weight: 700; color: {THEME['text']}; margin: 0;
}}
.hero-sub {{ font-size: 0.84rem; color: {THEME['muted']}; margin: 0.2rem 0 0; }}

.pill {{
    display:inline-block; padding:.18rem .6rem; border-radius:999px;
    font-size:.68rem; font-family:'Space Mono',monospace;
    letter-spacing:.05em; font-weight:700; text-transform:uppercase;
}}
.pill-green  {{ background:{THEME['accent_bg']}; color:{THEME['accent']}; border:1px solid {THEME['accent_bdr']}; }}
.pill-orange {{ background:{'#2c1a00' if D else '#fff7ed'}; color:{THEME['orange']}; border:1px solid {'#5a3300' if D else '#fed7aa'}; }}

.bubble-user {{
    background:{THEME['user_bg']}; border:1px solid {THEME['user_bdr']};
    border-radius:12px 12px 4px 12px; padding:.8rem 1.1rem;
    margin:.5rem 0 .5rem 3rem; font-size:.94rem; color:{THEME['text']};
}}
.bubble-ai {{
    background:{THEME['ai_bg']}; border:1px solid {THEME['ai_bdr']};
    border-radius:12px 12px 12px 4px; padding:.8rem 1.1rem;
    margin:.5rem 3rem .5rem 0; font-size:.94rem; color:{THEME['ai_text']};
}}
.bubble-label {{
    font-family:'Space Mono',monospace; font-size:.64rem; letter-spacing:.1em;
    text-transform:uppercase; margin-bottom:.3rem; color:{THEME['muted']};
}}

.trace-card {{
    background:{THEME['trace_bg']}; border:1px solid {THEME['trace_bdr']};
    border-left:3px solid {THEME['blue']}; border-radius:8px;
    padding:.65rem 1rem; margin:.35rem 0;
    font-family:'Space Mono',monospace; font-size:.71rem; color:{THEME['blue']};
}}
.trace-result {{ border-left-color:{THEME['accent']}; color:{THEME['accent']}; }}

.result-card {{
    background:{'linear-gradient(135deg,#071a0f,#0c2318)' if D else 'linear-gradient(135deg,#f0fdf4,#dcfce7)'};
    border:1px solid {THEME['ai_bdr']}; border-radius:12px;
    padding:1.3rem 1.5rem; margin-top:.8rem;
}}
.result-value {{
    font-family:'Space Mono',monospace; font-size:2rem;
    font-weight:700; color:{THEME['accent']};
}}
.result-label {{
    font-size:.78rem; color:{THEME['muted']};
    text-transform:uppercase; letter-spacing:.1em;
}}

.metric-row {{ display:flex; gap:.75rem; flex-wrap:wrap; margin:.6rem 0; }}
.metric-tile {{
    flex:1; min-width:130px; background:{THEME['tile_bg']};
    border:1px solid {THEME['card_bdr']}; border-radius:10px; padding:.85rem 1rem;
}}
.metric-tile .val {{
    font-family:'Space Mono',monospace; font-size:1.2rem;
    font-weight:700; color:{THEME['text']};
}}
.metric-tile .lbl {{
    font-size:.7rem; color:{THEME['muted']};
    text-transform:uppercase; letter-spacing:.08em; margin-top:.2rem;
}}
.metric-tile.green .val  {{ color:{THEME['accent']}; }}
.metric-tile.blue  .val  {{ color:{THEME['blue']}; }}
.metric-tile.orange .val {{ color:{THEME['orange']}; }}

.qq-card {{
    background:{THEME['card_bg']}; border:1px solid {THEME['card_bdr']};
    border-radius:10px; padding:.75rem 1rem; margin:.3rem 0;
}}
.qq-title {{ font-weight:600; font-size:.82rem; color:{THEME['text']}; margin-bottom:.2rem; }}
.qq-desc  {{ font-size:.74rem; color:{THEME['muted']}; line-height:1.45; }}
.qq-tag {{
    display:inline-block; margin-top:.35rem; padding:.15rem .55rem;
    border-radius:999px; font-size:.64rem; font-family:'Space Mono',monospace;
    font-weight:700; text-transform:uppercase; letter-spacing:.05em;
}}
.qq-tag-green  {{ background:{THEME['accent_bg']}; color:{THEME['accent']}; }}
.qq-tag-blue   {{ background:{'#0c1a35' if D else '#eff6ff'}; color:{THEME['blue']}; }}
.qq-tag-orange {{ background:{'#2c1a00' if D else '#fff7ed'}; color:{THEME['orange']}; }}

.tool-card {{
    background:{THEME['card_bg']}; border:1px solid {THEME['card_bdr']};
    border-radius:10px; padding:.85rem 1rem; margin:.35rem 0;
}}
.tool-card-title {{ font-weight:600; font-size:.83rem; color:{THEME['text']}; }}
.tool-card-desc  {{ font-size:.75rem; color:{THEME['muted']}; margin-top:.2rem; line-height:1.5; }}

.stTextArea textarea {{
    background:{THEME['input_bg']} !important; border:1px solid {THEME['input_bdr']} !important;
    color:{THEME['text']} !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-size:.92rem !important;
}}
.stTextArea textarea:focus {{
    border-color:{THEME['accent']} !important;
    box-shadow:0 0 0 2px {'rgba(74,222,128,.12)' if D else 'rgba(22,163,74,.12)'} !important;
}}
.stNumberInput input {{
    background:{THEME['input_bg']} !important; color:{THEME['text']} !important;
    border-color:{THEME['input_bdr']} !important;
}}
.stSelectbox > div > div {{
    background:{THEME['input_bg']} !important; color:{THEME['text']} !important;
}}
.stButton > button {{
    background:{THEME['btn_bg']}; border:1px solid {THEME['btn_bdr']};
    color:{THEME['btn_text']}; border-radius:8px; font-size:.82rem;
    font-family:'DM Sans',sans-serif; transition:all .18s;
    width:100%; text-align:left; padding:.5rem .9rem;
}}
.stButton > button:hover {{
    background:{'#1e293b' if D else '#f1f5f9'};
    border-color:{THEME['blue']}; color:{THEME['text']};
}}
.stTabs [data-baseweb="tab-list"] {{
    background:{'#0d0f14' if D else '#f8fafc'};
    border-bottom:1px solid {THEME['card_bdr']};
}}
.stTabs [data-baseweb="tab"] {{
    color:{THEME['muted']}; font-family:'DM Sans',sans-serif; font-size:.88rem;
}}
.stTabs [aria-selected="true"] {{
    color:{THEME['accent']} !important;
    border-bottom:2px solid {THEME['accent']} !important;
}}
hr {{ border-color:{THEME['card_bdr']} !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")
MCP_CONFIG = {
    "orlando_realestate": {
        "command": "python",
        "args": [MCP_SERVER_PATH],
        "transport": "stdio",
        "env": {**os.environ},
    }
}

SYSTEM_PROMPT = """You are an expert Orlando real estate AI assistant with 5 specialized tools:
1. zoning_law_query           — Orlando municipal zoning codes (RAG)
2. property_damage_assessment — Vision AI property damage assessment
3. predict_fair_market_value  — XGBoost fair market value prediction
4. assess_walkability         — Live OpenStreetMap walkability scoring
5. orlando_market_expert      — Fine-tuned Orlando market trends expert

CRITICAL — property_damage_assessment calling convention:
ALL arguments MUST be nested inside a "params" object. Valid fields:
  - mode: "search" (find similar cases) or "analyze" (analyze an uploaded image)
  - search_query: text description of damage — use this field name, NOT "query"
  - image_path: absolute file path to image (analyze mode only)
  - top_k: number of results (default 3)

CORRECT search mode:  {"params": {"mode": "search", "search_query": "roof damage mold", "top_k": 3}}
CORRECT analyze mode: {"params": {"mode": "analyze", "image_path": "/tmp/uploaded_damage.jpg", "top_k": 3}}
WRONG (never do this): {"mode": "search", "query": "mold"}

When the user uploads an image, always use mode "analyze" with the provided image_path.
When no image is uploaded, use mode "search" with search_query.

Guidelines: use the most relevant tool(s). Combine FMV + walkability + market expert for full evals.
Format dollars with commas, scores as X/100. Be concise but thorough."""

TOOL_META = {
    "predict_fair_market_value": {
        "icon": "💰",
        "title": "XGBoost Fair Market Value",
        "desc": (
            "Predicts property value using gradient boosting trained on Orlando sales. "
            "Captures non-linear effects of location, size, age, transit proximity, "
            "water access, and seasonal timing — things linear regression misses entirely. "
            "Try Winter Park vs Kissimmee to see the model capture the full price gradient."
        ),
        "tag": ("🤖 ML MODEL", "green"),
    },
    "assess_walkability": {
        "icon": "🚶",
        "title": "Live Walkability Score",
        "desc": (
            "Pulls real-time amenity density from OpenStreetMap — shops, transit stops, "
            "restaurants, parks, schools — within 500m and converts to a 0–100 score. "
            "No estimates: actual POI counts, live data every time."
        ),
        "tag": ("📡 LIVE OSM", "blue"),
    },
    "zoning_law_query": {
        "icon": "📋",
        "title": "Orlando Zoning RAG",
        "desc": (
            "Semantic search over Orlando's full municipal code. Ask about ADU rules, "
            "setbacks, tree permits, short-term rentals, height limits — returns exact "
            "code sections, not generic summaries."
        ),
        "tag": ("🔍 RAG", "orange"),
    },
    "property_damage_assessment": {
        "icon": "🔍",
        "title": "Damage Assessment (Vision AI)",
        "desc": (
            "Describe or reference property damage — roof, foundation, water intrusion, "
            "storm damage. Returns repair cost ranges and severity classification using "
            "GPT-4o vision analysis."
        ),
        "tag": ("👁 VISION AI", "orange"),
    },
    "orlando_market_expert": {
        "icon": "📊",
        "title": "Fine-tuned Market Expert",
        "desc": (
            "A model fine-tuned specifically on Orlando neighborhood data, investor reports, "
            "and market trends. Ask about best zip codes, cap rates, rental demand, "
            "gentrification signals, and appreciation forecasts."
        ),
        "tag": ("🎯 FINE-TUNED", "green"),
    },
}

# ── QUICK QUERIES  (btn_label, query, title, desc, tag_text, tag_color)
QUICK_QUERIES = [
    (
        "💎 Winter Park Luxury Home",
        (
            "Predict the fair market value for a luxury property in Winter Park, Orlando "
            "at latitude 28.5997, longitude -81.3392, with 18000 sqft of land, 4200 sqft living area, "
            "8 years old, quality score 5.0, special features value $45000, "
            "rail distance 2200m, ocean distance 52000m, water distance 600m, "
            "city center distance 12000m, subcenter distance 3000m, highway distance 4500m, sold in March."
        ),
        "💎 Luxury Benchmark — Winter Park",
        (
            "High-end home in one of Orlando's most coveted zip codes. "
            "Winter Park's lakefront proximity, top quality score, and tight city-center distance "
            "should push the XGBoost prediction well above $1M. "
            "This is the model's luxury ceiling — watch it reflect the premium."
        ),
        "💎 LUXURY · WINTER PARK", "green",
    ),
    (
        "🏘️ Kissimmee Suburban Starter",
        (
            "Predict the fair market value for a suburban starter home near Kissimmee "
            "at latitude 28.2919, longitude -81.4078, with 6500 sqft of land, 1450 sqft living area, "
            "22 years old, quality score 2.5, special features value $0, "
            "rail distance 9000m, ocean distance 72000m, water distance 8000m, "
            "city center distance 30000m, subcenter distance 12000m, highway distance 1200m, sold in August."
        ),
        "🏘️ Suburban Baseline — Kissimmee",
        (
            "Older starter home in suburban Kissimmee — run this after Winter Park. "
            "Same market, completely different drivers: farther from water, older, lower quality, "
            "higher city-center distance. The gap between these two predictions proves "
            "XGBoost is capturing real location variance, not just square footage."
        ),
        "🏘️ SUBURBAN · KISSIMMEE", "orange",
    ),
    (
        "🏙️ Downtown Walkability",
        "Assess the walkability score for a property in downtown Orlando at latitude 28.5383, longitude -81.3792.",
        "🏙️ Urban Benchmark — Downtown",
        (
            "Downtown's dense grid of restaurants, transit, shops, and offices typically scores 70+. "
            "Use this as your high-water mark — then run Horizon West to see the suburban contrast."
        ),
        "🏙️ DOWNTOWN URBAN", "blue",
    ),
    (
        "🚗 Horizon West Walkability",
        "Assess the walkability score for a property in Horizon West, a master-planned suburb of Orlando, at latitude 28.4165, longitude -81.5918.",
        "🚗 Suburban Contrast — Horizon West",
        (
            "Master-planned but car-dependent — Horizon West typically scores 20–40. "
            "Side-by-side with downtown this shows exactly how the model quantifies "
            "urban vs suburban access from live OSM data."
        ),
        "🚗 SUBURBAN · HORIZON WEST", "orange",
    ),
    (
        "📋 ADU + Tree Permit Rules",
        "What are Orlando's zoning rules for building an ADU on a single-family lot? What size tree requires a removal permit?",
        "📋 Zoning RAG — ADU & Tree Code",
        (
            "Two of the most-asked Orlando zoning questions pulled directly from the municipal code "
            "via semantic RAG search — exact section references, no generic summaries."
        ),
        "📖 ZONING RAG", "blue",
    ),
    (
        "📊 Best Orlando Zip Codes 2026",
        "What are the best zip codes and neighborhoods in Orlando to invest in right now for long-term appreciation? Include cap rate expectations and rental demand signals.",
        "📊 Market Expert — Investment Intel",
        (
            "The fine-tuned market expert draws on Orlando-specific training to surface "
            "real investment signals — not generic advice you'd get from any LLM."
        ),
        "🎯 FINE-TUNED EXPERT", "green",
    ),
]

# ══════════════════════════════════════════════════════════
# ASYNC + AGENT HELPERS
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result(timeout=180)

async def _init_agent():
    client = MultiServerMCPClient(MCP_CONFIG)
    tools  = await client.get_tools()
    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent  = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    return agent, tools

async def _run_query(agent, history: list[dict], query: str, image_path: str = None):
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=query))

    traces, final = [], ""
    async for chunk in agent.astream({"messages": msgs}, stream_mode="updates"):
        for _, state in chunk.items():
            for msg in state.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            # ── VISION AI RUNTIME PATCH ───────────────────────────
                            if tc["name"] == "property_damage_assessment":
                                args = tc.get("args", {})
                                if "params" not in args:
                                    args = {"params": args}
                                p = args.get("params", {})
                                if "query" in p and "search_query" not in p:
                                    p["search_query"] = p.pop("query")
                                if image_path:
                                    p["mode"] = "analyze"
                                    p["image_path"] = image_path
                                else:
                                    if "mode" not in p:
                                        p["mode"] = "search"
                                allowed = {"mode", "image_path", "search_query", "top_k"}
                                args["params"] = {k: v for k, v in p.items() if k in allowed}
                                tc["args"] = args
                            # ─────────────────────────────────────────────────────
                            traces.append({"type": "call", "tool": tc["name"], "args": tc["args"]})
                    elif msg.content:
                        final = msg.content
                elif isinstance(msg, ToolMessage):
                    traces.append({"type": "result", "tool": msg.name, "content": msg.content})
    return final, traces

# ══════════════════════════════════════════════════════════
# EXTRACT HELPERS
# ══════════════════════════════════════════════════════════
def _extract_fmv(traces):
    for t in traces:
        if t["type"] == "result" and t["tool"] == "predict_fair_market_value":
            try:
                raw = t["content"]
                if isinstance(raw, list): raw = raw[0].get("text", "")
                return json.loads(raw).get("predicted_fmv")
            except Exception: pass
    return None

def _extract_walkability(traces):
    for t in traces:
        if t["type"] == "result" and t["tool"] == "assess_walkability":
            try:
                raw = t["content"]
                if isinstance(raw, list): raw = raw[0].get("text", "")
                d = json.loads(raw)
                return d.get("walkability_score") or d.get("score")
            except Exception: pass
    return None

def _fmt_dollar(v): return f"${v:,.0f}"

# ══════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════
def render_traces(traces):
    for t in traces:
        icon = TOOL_META.get(t["tool"], {}).get("icon", "🔧")
        if t["type"] == "call":
            args_str = json.dumps(t["args"], indent=2)[:300].replace("\n", " · ")
            st.markdown(
                f'<div class="trace-card">{icon} <b>{t["tool"]}</b> — calling'
                f'<br><span style="opacity:.7">{args_str}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            raw = t["content"]
            if isinstance(raw, list): raw = raw[0].get("text", str(raw)) if raw else ""
            st.markdown(
                f'<div class="trace-card trace-result">✅ <b>{t["tool"]}</b> — returned'
                f'<br><span style="opacity:.75">{str(raw)[:280].replace(chr(10)," ")}…</span></div>',
                unsafe_allow_html=True,
            )

def render_metrics(fmv, walk):
    parts = []
    if fmv:
        parts.append(f'<div class="metric-tile green"><div class="val">{_fmt_dollar(fmv)}</div><div class="lbl">Predicted FMV</div></div>')
    if walk is not None:
        cls = "green" if walk >= 70 else ("orange" if walk >= 40 else "")
        parts.append(f'<div class="metric-tile {cls}"><div class="val">{walk}/100</div><div class="lbl">Walkability</div></div>')
    if parts:
        st.markdown(f'<div class="metric-row">{"".join(parts)}</div>', unsafe_allow_html=True)

def render_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user"><div class="bubble-label">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-ai"><div class="bubble-label">🏠 AI Agent</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("traces"):
                with st.expander("🔍 Tool trace", expanded=False):
                    render_traces(msg["traces"])
            render_metrics(
                _extract_fmv(msg.get("traces", [])),
                _extract_walkability(msg.get("traces", [])),
            )

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:.5rem 0 .8rem">
      <span style="font-size:2.1rem">🏠</span>
      <div style="font-family:'Space Mono',monospace;font-size:.93rem;color:{THEME['text']};margin-top:.3rem;font-weight:700">Orlando RE · AI</div>
      <div style="font-size:.71rem;color:{THEME['muted']};margin-top:.15rem">LangGraph + GPT-5</div>
    </div>
    """, unsafe_allow_html=True)

    # ── DARK / LIGHT TOGGLE ──
    col_lbl, col_tog = st.columns([3, 1])
    with col_lbl:
        st.markdown(f'<span style="font-size:.82rem;color:{THEME["muted"]}">{"🌙 Dark Mode" if D else "☀️ Light Mode"}</span>', unsafe_allow_html=True)
    with col_tog:
        if st.button("Switch", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    if st.session_state.agent_ready:
        n = len(st.session_state.tools)
        st.markdown(f'<span class="pill pill-green">● {n} tools ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-orange">○ Initializing…</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── TOOL CARDS ──
    st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:.69rem;letter-spacing:.13em;text-transform:uppercase;color:{THEME["accent"]};margin-bottom:.5rem">🛠 What Each Tool Does</div>', unsafe_allow_html=True)
    for key, meta in TOOL_META.items():
        tag_text, tag_color = meta["tag"]
        st.markdown(f"""
        <div class="tool-card">
          <div class="tool-card-title">{meta['icon']} {meta['title']}</div>
          <div class="tool-card-desc">{meta['desc']}</div>
          <span class="qq-tag qq-tag-{tag_color}">{tag_text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── QUICK QUERY CARDS ──
    st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:.69rem;letter-spacing:.13em;text-transform:uppercase;color:{THEME["accent"]};margin-bottom:.5rem">⚡ Try These Queries</div>', unsafe_allow_html=True)

    for (btn_lbl, query, title, desc, tag_text, tag_color) in QUICK_QUERIES:
        st.markdown(f"""
        <div class="qq-card">
          <div class="qq-title">{title}</div>
          <div class="qq-desc">{desc}</div>
          <span class="qq-tag qq-tag-{tag_color}">{tag_text}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(btn_lbl, key=f"qq_{btn_lbl}"):
            st.session_state["chat_input_value"] = query
            st.session_state["auto_send"]  = True
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════
# INIT AGENT
# ══════════════════════════════════════════════════════════
if not st.session_state.agent_ready:
    with st.spinner("🔌 Connecting to MCP server & loading tools…"):
        try:
            agent, tools = run_async(_init_agent())
            st.session_state.agent       = agent
            st.session_state.tools       = tools
            st.session_state.agent_ready = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Agent init failed: {e}")
            st.stop()

# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-icon">🏠</div>
  <div>
    <p class="hero-title">Orlando Real Estate AI</p>
    <p class="hero-sub">XGBoost valuations · Live walkability · Zoning RAG · Damage vision AI · Fine-tuned market expert</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab_chat, tab_fmv, tab_walk = st.tabs(["💬 Chat", "💰 FMV Calculator", "🚶 Walkability"])

# ─────────────────────────────────────────
# TAB 1 · CHAT
# ─────────────────────────────────────────
with tab_chat:
    render_chat_history()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── IMAGE UPLOADER for Vision AI ──────────────────────────
    with st.expander("📸 Attach image for Vision AI damage analysis", expanded=False):
        uploaded_img = st.file_uploader(
            "Upload a property damage photo (roof, mold, foundation, water damage, etc.)",
            type=["jpg", "jpeg", "png", "webp"],
            key="damage_image",
            label_visibility="visible",
        )
        if uploaded_img:
            st.image(uploaded_img, caption=f"📎 {uploaded_img.name}", use_container_width=True)
            st.caption("✅ Image attached — your next message will trigger Vision AI analyze mode automatically.")

    # ── INPUT ROW ─────────────────────────────────────────────
    col_inp, col_btn = st.columns([6, 1])
    with col_inp:
        user_input = st.text_area(
            label="msg",
            value=st.session_state.get("chat_input_value", ""),
            height=85,
            label_visibility="collapsed",
            placeholder="Ask anything — or upload an image above to analyze property damage…",
        )
    with col_btn:
        st.markdown("<div style='padding-top:.4rem'></div>", unsafe_allow_html=True)
        clicked_send = st.button("Send ➤", key="send_btn", use_container_width=True)

    auto_send   = st.session_state.pop("auto_send", False)
    should_send = clicked_send or auto_send

    if should_send:
        query = user_input.strip()

        # If image uploaded but no query, provide a default prompt
        if not query and uploaded_img:
            query = "Please analyze this property damage image. Identify the damage type, severity, affected systems, and estimated repair cost."

        if query:
            # Save uploaded image to a temp file so the MCP tool can read it by path
            saved_image_path = None
            if uploaded_img:
                import tempfile
                suffix = Path(uploaded_img.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_img.getbuffer())
                    saved_image_path = tmp.name

            # Build display content — show thumbnail in chat history if image attached
            display_query = query
            if uploaded_img:
                display_query = f"📸 **[Image: {uploaded_img.name}]**\n\n{query}"

            history_snapshot = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            st.session_state.messages.append({"role": "user", "content": display_query})
            st.session_state["chat_input_value"] = ""

            with st.spinner("🤖 Agent thinking…" if not saved_image_path else "🤖 Agent thinking…"):
                try:
                    final, traces = run_async(
                        _run_query(st.session_state.agent, history_snapshot, query, image_path=saved_image_path)
                    )
                    if not final:
                        final = "⚠️ The agent returned an empty response. Try rephrasing, or check tool connectivity."
                except Exception as e:
                    final, traces = f"⚠️ Error: {e}", []

            # Clean up temp file
            if saved_image_path:
                try:
                    import os as _os
                    _os.unlink(saved_image_path)
                except Exception:
                    pass

            st.session_state.messages.append({"role": "assistant", "content": final, "traces": traces})
            st.rerun()

# ─────────────────────────────────────────
# TAB 2 · FMV CALCULATOR
# ─────────────────────────────────────────
with tab_fmv:
    st.markdown("### 💰 Fair Market Value Estimator")
    st.markdown(
        f'<p style="color:{THEME["muted"]};font-size:.84rem;margin-bottom:1rem">'
        f'Powered by XGBoost trained on Orlando sales data. '
        f'Select a preset location to populate the fields, then click Predict.</p>',
        unsafe_allow_html=True,
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1: load_wp    = st.button("💎 Winter Park Luxury",     use_container_width=True, key="preset_wp")
    with pc2: load_lake  = st.button("🌊 Lake Nona Modern",       use_container_width=True, key="preset_lake")
    with pc3: load_apopka = st.button("🏡 Apopka Starter Home",   use_container_width=True, key="preset_apopka")

    PRESET_WP     = dict(lat=28.5997, lon=-81.3392, land=18000, living=4200, age=8,  qual=5.0, spec=45000, month_idx=2,  rail=2200,  ocean=52000, water=600,  center=12000, sub=3000,  highway=4500,  name="wp")
    PRESET_LAKE   = dict(lat=28.3772, lon=-81.2459, land=9500,  living=2600, age=5,  qual=4.5, spec=15000, month_idx=4,  rail=8500,  ocean=65000, water=1200, center=22000, sub=6000,  highway=2000,  name="lake")
    PRESET_APOPKA = dict(lat=28.6934, lon=-81.5322, land=7200,  living=1650, age=18, qual=3.0, spec=0,     month_idx=8,  rail=12000, ocean=78000, water=4500, center=35000, sub=14000, highway=1800,  name="apopka")
    PRESET_DEF    = dict(lat=28.5383, lon=-81.3792, land=8500,  living=2000, age=15, qual=4.0, spec=0,     month_idx=5,  rail=5000,  ocean=55000, water=1500, center=40000, sub=8000,  highway=3000,  name="def")

    if load_wp:     st.session_state["fmv_preset"] = PRESET_WP
    if load_lake:   st.session_state["fmv_preset"] = PRESET_LAKE
    if load_apopka: st.session_state["fmv_preset"] = PRESET_APOPKA

    P      = st.session_state.get("fmv_preset", PRESET_DEF)
    pk     = P["name"]
    MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

    c1, c2, c3 = st.columns(3)
    with c1:
        lat    = st.number_input("Latitude",           value=float(P["lat"]),  format="%.4f", key=f"fmv_lat_{pk}")
        lon    = st.number_input("Longitude",          value=float(P["lon"]),  format="%.4f", key=f"fmv_lon_{pk}")
        land   = st.number_input("Land area (sqft)",   value=int(P["land"]),   step=100,       key=f"fmv_land_{pk}")
    with c2:
        living = st.number_input("Living area (sqft)", value=int(P["living"]), step=50,        key=f"fmv_living_{pk}")
        age    = st.number_input("Age (years)",        value=int(P["age"]),    step=1,         key=f"fmv_age_{pk}")
        qual   = st.slider("Structure quality (1–5)",  1.0, 5.0, float(P["qual"]), 0.5,        key=f"fmv_qual_{pk}")
    with c3:
        spec   = st.number_input("Special features ($)", value=int(P["spec"]), step=500,       key=f"fmv_spec_{pk}")
        month  = st.selectbox("Month sold", MONTHS, index=int(P["month_idx"]),                 key=f"fmv_month_{pk}")

    st.markdown(f'<div style="font-size:.82rem;font-weight:600;color:{THEME["text"]};margin:.6rem 0 .3rem">📍 Distance Features — spatial encoding inputs for the model</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        rail    = st.number_input("Rail distance (m)",    value=int(P["rail"]),    step=100, key=f"fmv_rail_{pk}")
        ocean   = st.number_input("Ocean distance (m)",   value=int(P["ocean"]),   step=500, key=f"fmv_ocean_{pk}")
    with d2:
        water   = st.number_input("Water distance (m)",   value=int(P["water"]),   step=100, key=f"fmv_water_{pk}")
        center  = st.number_input("City center dist (m)", value=int(P["center"]),  step=500, key=f"fmv_center_{pk}")
    with d3:
        sub     = st.number_input("Subcenter dist (m)",   value=int(P["sub"]),     step=100, key=f"fmv_sub_{pk}")
        highway = st.number_input("Highway dist (m)",     value=int(P["highway"]), step=100, key=f"fmv_highway_{pk}")

    if st.button("🔮 Predict Fair Market Value", use_container_width=True, key="fmv_predict"):
        q = (
            f"Predict the fair market value for a property at latitude {lat}, longitude {lon}, "
            f"with {land} sqft of land, {living} sqft living area, {age} years old, "
            f"quality score {qual}, special features value ${spec}, "
            f"rail distance {rail}m, ocean distance {ocean}m, water distance {water}m, "
            f"city center distance {center}m, subcenter distance {sub}m, "
            f"highway distance {highway}m, sold in {month}."
        )
        with st.spinner("Running XGBoost model via agent…"):
            try:
                final, traces = run_async(_run_query(st.session_state.agent, [], q))
            except Exception as e:
                final, traces = f"⚠️ {e}", []

        fmv = _extract_fmv(traces)
        if fmv:
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-label">Predicted Fair Market Value</div>'
                f'<div class="result-value">{_fmt_dollar(fmv)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if final:
            st.markdown(
                f'<div class="bubble-ai" style="margin-top:.8rem">'
                f'<div class="bubble-label">🏠 AI Analysis</div>{final}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔍 Tool trace", expanded=False):
            render_traces(traces)

        st.session_state.messages.append({"role": "user",      "content": q})
        st.session_state.messages.append({"role": "assistant", "content": final, "traces": traces})

# ─────────────────────────────────────────
# TAB 3 · WALKABILITY
# ─────────────────────────────────────────
with tab_walk:
    st.markdown("### 🚶 Walkability Score")
    st.markdown(
        f'<p style="color:{THEME["muted"]};font-size:.84rem;margin-bottom:1rem">'
        f'Live amenity density from OpenStreetMap — actual POI counts within 500m, not estimates. '
        f'Select a preset or enter any Orlando coordinates.</p>',
        unsafe_allow_html=True,
    )

    wc1, wc2, wc3 = st.columns(3)
    with wc1: w_dt   = st.button("🏙️ Downtown Orlando",   use_container_width=True, key="w_pre_dt")
    with wc2: w_ln   = st.button("🏥 Lake Nona",           use_container_width=True, key="w_pre_ln")
    with wc3: w_mill = st.button("🎢 International Drive",  use_container_width=True, key="w_pre_id")

    W_PRESETS = {
        "downtown": (28.5383, -81.3792, "downtown"),
        "lakenona": (28.3772, -81.2459, "lakenona"),
        "idrive":   (28.4312, -81.4706, "idrive"),
    }
    if w_dt:   st.session_state["walk_preset"] = W_PRESETS["downtown"]
    if w_ln:   st.session_state["walk_preset"] = W_PRESETS["lakenona"]
    if w_mill: st.session_state["walk_preset"] = W_PRESETS["idrive"]

    wp_def = st.session_state.get("walk_preset", W_PRESETS["downtown"])
    wk = wp_def[2]
    wl, wr = st.columns(2)
    with wl: w_lat = st.number_input("Latitude",  value=float(wp_def[0]), format="%.4f", key=f"w_lat_{wk}")
    with wr: w_lon = st.number_input("Longitude", value=float(wp_def[1]), format="%.4f", key=f"w_lon_{wk}")

    if st.button("🚶 Check Walkability", use_container_width=True, key="w_check"):
        q = f"Assess the walkability score for a property at latitude {w_lat}, longitude {w_lon} in Orlando, FL."
        with st.spinner("Querying OpenStreetMap via agent…"):
            try:
                final, traces = run_async(_run_query(st.session_state.agent, [], q))
            except Exception as e:
                final, traces = f"⚠️ {e}", []

        walk = _extract_walkability(traces)
        if walk is not None:
            color = THEME["accent"] if walk >= 70 else (THEME["orange"] if walk >= 40 else "#f87171")
            label = ("Highly Walkable 🟢" if walk >= 70 else ("Somewhat Walkable 🟡" if walk >= 40 else "Car Dependent 🔴"))
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-label">Walkability Score</div>'
                f'<div class="result-value" style="color:{color}">{walk}/100</div>'
                f'<div style="color:{color};font-size:.85rem;margin-top:.3rem;font-weight:500">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if final:
            st.markdown(
                f'<div class="bubble-ai" style="margin-top:.8rem">'
                f'<div class="bubble-label">🏠 AI Analysis</div>{final}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔍 Tool trace", expanded=False):
            render_traces(traces)

        st.session_state.messages.append({"role": "user",      "content": q})
        st.session_state.messages.append({"role": "assistant", "content": final, "traces": traces})