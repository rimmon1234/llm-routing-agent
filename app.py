import streamlit as st
import time
import json
import plotly.graph_objects as go
from src.client import LLMClient
from src.router import HybridRouter

# Set page configuration with dark-themed styling options
st.set_page_config(
    page_title="Hybrid LLM Routing Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for the web UI
st.markdown("""
<style>
    /* Main styling overrides */
    .reportview-container {
        background: #0f111a;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #ff3b30;
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #f5f5f7;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-rocm {
        background-color: rgba(255, 59, 48, 0.15);
        color: #ff453a;
        border: 1px solid rgba(255, 59, 48, 0.3);
    }
    .badge-cpu {
        background-color: rgba(142, 142, 147, 0.15);
        color: #aeaeae;
        border: 1px solid rgba(142, 142, 147, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "total_saved" not in st.session_state:
    st.session_state.total_saved = 0.0
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.shields.io/badge/AMD-Developer_Hackathon_ACT_II-red?style=for-the-badge&logo=amd", use_container_width=True)
st.sidebar.title("Configuration")

# Credentials section
st.sidebar.subheader("API Access")
import os
# Try to get Fireworks key from environment first, then fallback to Streamlit secrets
default_key = os.getenv("FIREWORKS_API_KEY", "")
if not default_key:
    try:
        default_key = st.secrets.get("FIREWORKS_API_KEY", "")
    except Exception:
        default_key = ""

fireworks_key = st.sidebar.text_input("Fireworks API Key", type="password", value=default_key)
ollama_host = st.sidebar.text_input("Ollama Host URL", value="http://localhost:11434")

# Model configuration
st.sidebar.subheader("Model Selection")
local_model = st.sidebar.selectbox("Local Model", ["llama3.2:3b", "llama3.2:1b", "gemma2:2b", "custom"])
if local_model == "custom":
    local_model = st.sidebar.text_input("Custom Local Model Name", value="llama3.2:3b")

remote_model = st.sidebar.selectbox("Remote Model", [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/qwen3p7-plus",
    "accounts/fireworks/models/gpt-oss-120b"
])

# Save keys dynamically to env variables
import os
os.environ["OLLAMA_HOST"] = ollama_host
os.environ["LOCAL_MODEL"] = local_model
os.environ["REMOTE_MODEL"] = remote_model
if fireworks_key:
    os.environ["FIREWORKS_API_KEY"] = fireworks_key

# Instantiate client and router
try:
    client = LLMClient()
    router = HybridRouter(client=client)
    gpu_info = client.gpu_info
except Exception as e:
    # Safe fallback if initialization fails on hosted servers
    client = None
    router = None
    gpu_info = {"available": False, "name": "CPU/Cloud Environment", "backend": "cpu"}

# GPU status indicator in sidebar
st.sidebar.subheader("System Status")
if gpu_info.get("available", False):
    st.sidebar.markdown(f'<div class="status-badge badge-rocm">🔴 AMD ROCm: {gpu_info.get("name")}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge badge-cpu">⚪ CPU Mode / Cloud Host</div>', unsafe_allow_html=True)

# ----------------- MAIN LAYOUT -----------------
st.title("🧠 Hybrid LLM Routing Agent")
st.markdown("### AMD Instinct™ & ROCm-Accelerated Local/Remote Model Router")
st.markdown("---")

# Main Dashboard Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{st.session_state.total_queries}</div>
        <div class="metric-lbl">Total Queries Routed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    local_count = sum(1 for item in st.session_state.history if item['route'] == 'local')
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{local_count}</div>
        <div class="metric-lbl">Handled Locally (Free)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    remote_count = sum(1 for item in st.session_state.history if 'remote' in item['route'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{remote_count}</div>
        <div class="metric-lbl">Routed to Remote</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">${st.session_state.total_saved:.5f}</div>
        <div class="metric-lbl">Estimated Token Cost Saved</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Playground and Analytics Split
tab_playground, tab_analytics = st.tabs(["🎮 Playground & Router Demo", "📊 Performance Analytics"])

with tab_playground:
    st.subheader("Interactive Query Routing")
    
    # Text input area
    query_text = st.text_area("Enter your query:", placeholder="e.g., Write a python function to compute the factorial of a number using recursion.")
    
    col_str, col_fmt, col_sch = st.columns([1, 1, 2])
    with col_str:
        strategy = st.selectbox("Routing Strategy", ["fallback", "predictive", "always_local", "always_remote"])
    with col_fmt:
        expected_format = st.selectbox("Expected Output Format", ["text", "json", "python"])
    with col_sch:
        schema_input = st.text_input("Expected JSON Fields (comma separated)", placeholder="e.g., fruit_name,calories", disabled=(expected_format != "json"))

    # Convert schema input to format expected by evaluator
    schema = None
    if expected_format == "json" and schema_input:
        schema = {field.strip(): str for field in schema_input.split(",")}

    run_btn = st.button("🚀 Route & Run Query", type="primary")

    if run_btn and query_text:
        if not client or not router:
            st.error("Router could not be initialized. Please check configuration.")
        else:
            with st.spinner("Routing and executing query..."):
                start_time = time.time()
                
                # Perform the routing call
                try:
                    metrics = router.route_and_execute(
                        query=query_text,
                        strategy=strategy,
                        response_format=expected_format,
                        schema=schema
                    )
                    response = metrics["response"]
                    latency = metrics["latency_sec"]
                    
                    # Compute tokens
                    local_tokens = metrics["prompt_tokens_local"] + metrics["completion_tokens_local"]
                    remote_tokens = metrics["prompt_tokens_remote"] + metrics["completion_tokens_remote"]
                    
                    # Compute dollar savings
                    remote_price = client.remote_price_per_1m_tokens if client else 0.90
                    estimated_savings = (local_tokens / 1_000_000.0) * remote_price

                    # Store to history
                    st.session_state.history.append({
                        "query": query_text,
                        "strategy": strategy,
                        "route": metrics["route_chosen"],
                        "latency": latency,
                        "cost_saved": estimated_savings,
                        "actual_cost": metrics["cost_dollars"],
                        "local_tokens": local_tokens,
                        "remote_tokens": remote_tokens
                    })
                    st.session_state.total_queries += 1
                    st.session_state.total_saved += estimated_savings

                    # Display success message with routing result
                    st.success(f"Successfully completed route! Chosen path: **{metrics['route_chosen'].upper()}**")
                    
                    # Display metrics summary
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Latency", f"{latency:.3f}s")
                    m_col2.metric("Route Chosen", metrics["route_chosen"].upper())
                    m_col3.metric("Token Savings", f"{metrics['cost_saved']*100:.1f}%")
                    m_col4.metric("Remote Cost", f"${metrics['cost_dollars']:.6f}")

                    # Columns for Response and Token Details
                    res_col, debug_col = st.columns([3, 2])
                    with res_col:
                        st.subheader("Response Output")
                        if expected_format == "python":
                            st.code(response, language="python")
                        elif expected_format == "json":
                            try:
                                parsed = json.loads(response)
                                st.json(parsed)
                            except Exception:
                                st.text(response)
                        else:
                            st.write(response)
                            
                    with debug_col:
                        st.subheader("Route Trace & Validation Logs")
                        st.json(metrics)
                        
                except Exception as ex:
                    st.error(f"Execution Error: {str(ex)}")

with tab_analytics:
    st.subheader("Routing Efficiency & Cost Analytics")
    if not st.session_state.history:
        st.info("No queries have been routed yet. Go to the playground to run queries!")
    else:
        # Create Plots
        labels = [item['query'][:20] + '...' for item in st.session_state.history]
        latencies = [item['latency'] for item in st.session_state.history]
        savings = [item['cost_saved'] for item in st.session_state.history]
        routes = [item['route'] for item in st.session_state.history]

        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Cumulative Cost Savings ($)")
            cumulative_savings = []
            current_sum = 0
            for s in savings:
                current_sum += s
                cumulative_savings.append(current_sum)
                
            fig_savings = go.Figure()
            fig_savings.add_trace(go.Scatter(
                x=list(range(1, len(savings) + 1)), 
                y=cumulative_savings, 
                mode='lines+markers',
                line=dict(color='#ff3b30', width=3),
                marker=dict(size=8)
            ))
            fig_savings.update_layout(
                title="Accumulated Savings vs. Query Count",
                xaxis_title="Query Number",
                yaxis_title="Savings ($)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_savings, use_container_width=True)

        with col_chart2:
            st.markdown("### Routing Distribution")
            route_types = {}
            for r in routes:
                route_types[r] = route_types.get(r, 0) + 1
                
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(route_types.keys()), 
                values=list(route_types.values()),
                hole=.3,
                marker=dict(colors=['#ff3b30', '#30d158', '#0a84ff'])
            )])
            fig_pie.update_layout(
                title="Proportion of Queries by Target Route",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Raw Data Table
        st.subheader("Query History")
        st.dataframe(st.session_state.history)
