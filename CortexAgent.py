import streamlit as st
import pandas as pd
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import _snowflake
from snowflake.snowpark.context import get_active_session
import re

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
API_ENDPOINT = "/api/v2/cortex/agent:run"
MAX_DATAFRAME_ROWS = 1_000
APP_VERSION = "2025‑04‑23"

SERVICES = [
    {
        "name": "Survey Search",
        "location": "DEMO_SNOWMOBILE.SURVEYS.SURVEY_SEARCH",
        "title_column": "FILE_NAME", #optional
        #"id_column": "FILE_NAME", #optional
        "type": "search",
        "emoji": "🔍",
        "max_results": 3,
        "description": "Search across the raw text of survey results" #optional
    },
    {
        "name": "Sales Data",
        "location": "LLM_DEMO.ANALYST.semantic/snowmobile_sales.yaml",
        "type": "analyst",
        "emoji": "📊",
        "description": "Transctional data related to sales including dealerships and geo data." #optional
    
    },
        #add/remove if required, using the format above for your tools
        #{
        #"name": "sales_data",
        #"location": "LLM_DEMO.ANALYST.semantic/snowmobile_sales.yaml",
        #"type": "analyst",
        #"emoji": "📊",
        #}
]

CUSTOM_INSTRUCTION = (
    """You are a helpful agent called MobyOne. Use the full conversation content when answering. 
       A tool isn't required. Be concise and use citations in your response. 
       When you cite information, always use the format [DOC_ID] where DOC_ID is the source_id of the document. 
       If no source_id exists, use [DOC1], [DOC2], etc. consistently."""
)

onboarding_q = '"What are the top customer complaints in recent surveys?" or "Show me the split of customers across households"'

#Proceed with caution here :) 
AGENT_EXPERIMENTAL_PARAMS = {
    #"EnableRelatedQueries": True,
    "OnlyUseToolsToAnswerQuestions": False
}


session = get_active_session()

# ────────────────────────────────────────────────────────────────────────────────
# MODELS & STATE
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Message:
    role: str  
    content: str
    type: str = "text"
    ts: datetime = field(default_factory=datetime.utcnow)

    # tool payloads
    search_results: List[Dict[str, Any]] | None = None
    sql: str | None = None
    sql_df: pd.DataFrame | None = None
    viz: go.Figure | None = None
    viz_type: str | None = None
    suggestions: List[str] | None = None

    def to_api(self) -> Dict[str, Any]:
        """Convert to Cortex agent message format."""
        if self.role not in {"user", "assistant", "system"}:
            return {}
        out: Dict[str, Any] = {"role": self.role, "content": []}
        if self.content:
            out["content"].append({"type": "text", "text": self.content})

        # bundle tool results when assistant
        if self.role == "assistant":
            if self.search_results is not None:
                out["content"].append(
                    {
                        "type": "tool_results",
                        "tool_results": {
                            "tool_use_id": f"sr_{id(self)}",
                            "name": self._find_search_service_name(),
                            "status": "success",
                            "content": [
                                {
                                    "type": "json",
                                    "json": {
                                        "searchResults": self.search_results,
                                        "text": "",
                                    },
                                }
                            ],
                        },
                    })
            if self.sql is not None:
                out["content"].append(
                    {
                        "type": "tool_results",
                        "tool_results": {
                            "tool_use_id": f"sql_{id(self)}",
                            "name": self._find_analyst_service_name(),
                            "status": "success",
                            "content": [
                                {
                                    "type": "json",
                                    "json": {
                                        "sql": self.sql,
                                        "text": f"I ran the following SQL query: {self.sql}",
                                    },
                                }
                            ],
                        },
                    })
        return out
    
    def _find_search_service_name(self) -> str:
        """Find first enabled search service name"""
        for service in SERVICES:
            if service["type"] == "search" and state.tool_enabled(service["name"]):
                return service["name"]
        return "search"  # fallback
    
    def _find_analyst_service_name(self) -> str:
        """Find first enabled analyst service name"""
        for service in SERVICES:
            if service["type"] == "analyst" and state.tool_enabled(service["name"]):
                return service["name"]
        return "analyst"  # fallback


class StateManager:
    """Typed wrapper around st.session_state to reduce key‑sprawl."""

    def __init__(self):
        s = st.session_state
        # single lists
        s.setdefault("messages", [])
        s.setdefault("api_log", [])
        # toggles for each service
        for service in SERVICES:
            s.setdefault(f"tool_{service['name']}", True)
        # debug toggle
        s.setdefault("debug", False)
        # model
        s.setdefault("model", "claude-3-5-sonnet")
        # animation 
        s.setdefault("new_message", False)
        self._s = s

    # collection helpers
    @property
    def messages(self) -> List[Message]:
        return self._s["messages"]

    def add_msg(self, msg: Message):
        self.messages.append(msg)
        # Set animation flag to true when a new assistant message is added
        if msg.role == "assistant":
            self._s["new_message"] = True

    # log helpers
    def log(self, kind: str, payload: Any):
        self._s["api_log"].append({kind: payload})

    # toggles
    def tool_enabled(self, name: str) -> bool:
        return self._s.get(f"tool_{name}", False)

    def set_tool(self, name: str, value: bool):
        self._s[f"tool_{name}"] = value
    
    # Check if any service of a specific type is enabled
    def has_enabled_service_type(self, service_type: str) -> bool:
        for service in SERVICES:
            if service["type"] == service_type and self.tool_enabled(service["name"]):
                return True
        return False


state = StateManager()

# ────────────────────────────────────────────────────────────────────────────────
# DATA + VISUALISATION GATEWAY
# ────────────────────────────────────────────────────────────────────────────────
class DataGateway:

    COLOR_PALETTE = px.colors.qualitative.Plotly
    CHART_TYPES = ("bar", "line", "scatter")

    def __init__(self, snowpark_session):
        self.snowpark_session = snowpark_session

    @st.cache_data(show_spinner=False, ttl=600, max_entries=128)
    def run_sql(_self, sql: str) -> pd.DataFrame:  # type: ignore[arg-type]
        try:
            df = pd.DataFrame(_self.snowpark_session.sql(sql).collect())
            if len(df) > MAX_DATAFRAME_ROWS:
                df = df.head(MAX_DATAFRAME_ROWS)
            return df
        except Exception as e:  # noqa: BLE001
            return pd.DataFrame({"Error": [str(e)]})

    # ───── plotting helpers ─────
    def auto_viz(self, df: pd.DataFrame) -> Tuple[go.Figure | None, str]:
        if df.empty:
            return None, "error"
        if len(df) == 1 and len(df.columns) == 1:
            return None, "metric"
        x, y = df.columns[0], df.columns[1] if len(df.columns) > 1 else df.columns[0]
        return self.create_viz(df, x, y, "bar"), "bar"

    def create_viz(
        self,
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        chart_type: str = "bar",
        color: str | None = None,
    ) -> go.Figure:
        chart_type = chart_type if chart_type in self.CHART_TYPES else "bar"
        args: Dict[str, Any] = {
            "data_frame": df,
            "x": x_axis,
            "y": y_axis,
            "color_discrete_sequence": self.COLOR_PALETTE,
        }
        if color:
            args["color"] = color
        fig = {
            "bar": px.bar,
            "line": px.line,
            "scatter": px.scatter,
        }[chart_type](**args)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(226,232,240,0.6)")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(226,232,240,0.6)")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>")
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(250,250,250,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig


dg = DataGateway(session)

# ────────────────────────────────────────────────────────────────────────────────
# API SERVICE (CORTEX)
# ────────────────────────────────────────────────────────────────────────────────
class APIService:
    @staticmethod
    def _tool_resources() -> Dict[str, Any]:
        resources = {}
        
        for service in SERVICES:
            if state.tool_enabled(service["name"]):
                if service["type"] == "search":
                    resource = {
                        "name": service["location"],
                        "max_results": service.get("max_results", 3),
                    }
                    
                    # Add optional column configurations
                    if "id_column" in service:
                        resource["id_column"] = service["id_column"]
                    if "title_column" in service:
                        resource["title_column"] = service["title_column"]
                        
                    resources[service["name"]] = resource
                    
                elif service["type"] == "analyst":
                    resources[service["name"]] = {
                        "semantic_model_file": f"@{service['location']}",
                    }
        
        return resources

    @staticmethod
    def _tools() -> List[Dict[str, Any]]:
        tools = []
        
        for service in SERVICES:
            if state.tool_enabled(service["name"]):
                if service["type"] == "search":
                    tools.append({
                        "tool_spec": {
                            "type": "cortex_search", 
                            "name": service["name"]
                        }
                    })
                elif service["type"] == "analyst":
                    tools.append({
                        "tool_spec": {
                            "type": "cortex_analyst_text_to_sql", 
                            "name": service["name"]
                        }
                    })
        
        return tools

    def build_payload(self, user_msg: str) -> Dict[str, Any]:
        # ensure last message is user
        if not state.messages or state.messages[-1].role != "user":
            state.add_msg(Message("user", user_msg))
        else:
            state.messages[-1].content = user_msg
        payload = {
            "model": state._s["model"],
            "response_instruction": CUSTOM_INSTRUCTION,
            "tools": self._tools(),
            "tool_resources": self._tool_resources(),
            "experimental": AGENT_EXPERIMENTAL_PARAMS,
            "messages": [m.to_api() for m in state.messages],
        }
        return payload

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return _snowflake.send_snow_api_request(
            "POST",
            API_ENDPOINT,
            {},
            {"stream": True},
            payload,
            None,
            60_000,
        )

api = APIService()

# ────────────────────────────────────────────────────────────────────────────────
# CHAT SERVICE
# ────────────────────────────────────────────────────────────────────────────────
class ChatService:
    """Processes user prompts and translates agent responses into Message objects."""

    def handle_user_prompt(self, prompt: str):
        with st.spinner('MobyOne is thinking...'):
            payload = api.build_payload(prompt)
            state.log("Request", payload)
            response = api.call(payload)
            state.log("Response", response)
            self._process_agent_response(response)

    # ── helpers ──
    @staticmethod
    def _decode_chunks(raw: Dict[str, Any] | List[Any]) -> List[Dict[str, Any]]:
        """Ensure we always work with a *list* of delta events.
        If the top-level `content` key is a JSON string, parse it.
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and isinstance(raw.get("content"), str):
            try:
                return json.loads(raw["content"])
            except Exception:  # fallback to wrapper
                pass
        return [raw]

    def _process_agent_response(self, raw_response):
        chunks = self._decode_chunks(raw_response)
        assistant = Message("assistant", "")
        txt: List[str] = []
    
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
                
            data = ch.get("data", {})
            if not isinstance(data, dict):
                continue
                
            for item in data.get("delta", {}).get("content", []):
                typ = item.get("type")
                if typ == "text":
                    txt.append(item.get("text", ""))
                elif typ == "tool_results":
                    jr = item["tool_results"]["content"][0]["json"]
                    # Capture the text interpretation from tool results
                    if text := jr.get("text"):
                        txt.append(text)
                    if sr := jr.get("searchResults"):
                        assistant.search_results = sr
                    if sql := jr.get("sql"):
                        assistant.sql = sql
                        df = dg.run_sql(sql)
                        assistant.sql_df = df
                        if not df.empty:
                            fig, vt = dg.auto_viz(df)
                            assistant.viz, assistant.viz_type = fig, vt
                    if sug := jr.get("suggestions"):
                        assistant.suggestions = sug
    
        assistant.content = "".join(txt).strip() or "I've processed your request."
        state.add_msg(assistant)



chat = ChatService()


# ────────────────────────────────────────────────────────────────────────────────
# UI HELPERS 
# ────────────────────────────────────────────────────────────────────────────────

def render_search_results(results: List[Dict[str, Any]]):
    """Render search results using native Streamlit components."""
    st.markdown("### 📄 Source Documents")
    
    for i, doc in enumerate(results):
        doc_id = doc.get('source_id', f"DOC{i+1}")
        doc_title = doc.get('doc_title', doc.get('doc_title', doc_id))
        doc_text = doc.get('text', doc.get('content', '—'))
        
        with st.container():
            # Create a card-like container with border
            st.markdown(f"""
                <div style="padding: 1px 1px 1px 10px; border-left: 4px solid #0066cc; 
                            background-color: #f9f9f9; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="margin: 10px 0 5px 0;"><span style="background-color: #0066cc; color: white; 
                              padding: 2px 8px; border-radius: 4px; font-size: 14px; margin-right: 10px;">
                              {doc_id}</span> {doc_title}</h4>
                </div>
            """, unsafe_allow_html=True)
            st.caption(doc_text)


def render_viz(msg: Message, message_index: int):
    chart_settings_key = f"chart_settings_{message_index}" # Make sure message_index is passed

    if chart_settings_key not in st.session_state:
        # Initialize settings using auto_viz defaults if first time
        initial_x = msg.sql_df.columns[0]
        initial_y = msg.sql_df.columns[1] if len(msg.sql_df.columns) > 1 else msg.sql_df.columns[0]
        initial_chart_type = msg.viz_type if msg.viz_type in dg.CHART_TYPES else "bar" # Use initial type
        initial_color = None
        st.session_state[chart_settings_key] = {
            "x": initial_x,
            "y": initial_y,
            "chart_type": initial_chart_type,
            "color": initial_color,
        }

    # Get current settings
    current_settings = st.session_state[chart_settings_key]
    cols = msg.sql_df.columns.tolist()
    chart_types = list(dg.CHART_TYPES)
    color_options = [None] + cols # Allow 'None' for no color grouping

    # --- Add Interactive Widgets ---
    with st.expander("📊 Chart Configuration"):
        c1, c2 = st.columns(2)
        with c1:
            # Use callbacks or direct assignment to update session state on change
            selected_x = st.selectbox(
                "X-axis", options=cols,
                index=cols.index(current_settings["x"]), # Set default from state
                key=f"{chart_settings_key}_x" # Unique key for widget
            )
            selected_chart_type = st.selectbox(
                "Chart Type", options=chart_types,
                index=chart_types.index(current_settings["chart_type"]),
                key=f"{chart_settings_key}_chart_type"
            )
        with c2:
            selected_y = st.selectbox(
                "Y-axis", options=cols,
                index=cols.index(current_settings["y"]),
                key=f"{chart_settings_key}_y"
            )
            selected_color = st.selectbox(
                "Color by (Optional)", options=color_options,
                index=color_options.index(current_settings["color"]),
                 key=f"{chart_settings_key}_color"
            )

    # --- Update State ---
    # Update the session state dictionary with new selections
    st.session_state[chart_settings_key] = {
        "x": selected_x,
        "y": selected_y,
        "chart_type": selected_chart_type,
        "color": selected_color,
     }

    # --- Dynamically Generate and Render Plot ---
    try:
        fig = dg.create_viz(
            df=msg.sql_df,
            x_axis=selected_x,
            y_axis=selected_y,
            chart_type=selected_chart_type,
            color=selected_color if selected_color != None else None # Pass None explicitly if needed by create_viz
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{message_index}")
        
    except Exception as e:
        st.error(f"Failed to generate visualization: {e}")


def render_sql_block(msg: Message, message_index: int):
    if msg.sql_df is None or msg.sql_df.empty:
        return
    
    tab1, tab2, tab3 = st.tabs([ "📋 Data", "📊 Visualisation", "🔍 SQL"])
    
    with tab1:
        st.dataframe(msg.sql_df, use_container_width=True, hide_index=True)
        
    with tab2:
        if msg.viz is not None: 
            render_viz(msg, message_index)
        else:
            st.info("No visualisation available for this data")
    
    
    with tab3:
        st.code(msg.sql or "", language="sql")


def render_debug_panel():
    with st.expander("🪲 Debug", expanded=False):
        st.json(state._s["api_log"][-2:] if state._s["api_log"] else [])


def render_message_content(msg: Message):
        st.markdown(msg.content)


def render_suggestions(suggestions):
    """Render interactive suggestion buttons."""
    if not suggestions:
        return
    
    # Create a horizontal container for suggestions
    st.markdown("#### Suggested follow-ups:")
    cols = st.columns(len(suggestions))
    
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}_{hash(suggestion)}", 
                        use_container_width=True):
                chat.handle_user_prompt(suggestion)
                st.rerun()


# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ────────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="MobyOne", page_icon="❄️", layout="wide")

    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # ── sidebar ──
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Snowflake_Logo.svg/2560px-Snowflake_Logo.svg.png", width=150)
        st.markdown("")
        
        if st.button("New chat", use_container_width=True):
            state._s.clear()
            st.rerun()
        st.markdown("")
        
        st.markdown("### ⚙️ MobyOne Settings")
        state._s["model"] = st.selectbox(
            "Pick your LLM:",
            ["claude-3-5-sonnet", "mistral-large2", "llama3.3-70b"],
            index=["claude-3-5-sonnet", "mistral-large2", "llama3.3-70b"].index(state._s["model"]),
        )
        
        st.markdown("---")
        st.markdown("### Tool Configuration")
        
        # Dynamically render toggles for each service
        for service in SERVICES:
            help_msg = service.get("description") or f"Tool type: {service['type']}"
            state.set_tool(
                service["name"], 
                st.toggle(
                    f"{service['emoji']} {service['name']}", 
                    value=state.tool_enabled(service["name"]),
                    help=help_msg
                )
            )
        
        st.markdown("---")
        st.markdown("### Advanced")
        state._s["debug"] = st.toggle("Debug", value=state._s["debug"])
        
        st.markdown("---")
        st.markdown("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 10px; font-size: 12px; color: #666;">
        MobyOne v1.0<br>
        • Powered by Snowflake Cortex •
        </div>
        """, unsafe_allow_html=True)

    # ── welcome message ──
    if not state.messages:
        st.markdown(
            f"""
            <div class="welcome-container">
                <div class="welcome-emoji">👋</div>
                <div class="welcome-title">Welcome to MobyOne</div>
                <div class="welcome-subtitle">Your interactive data assistant powered by Snowflake Cortex and the Data Agents framework. Feel free to update this message to suit your own application!</div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Cortex Search</div>
                        <div class="feature-description">High quality search for best-in-class retrieval accuracy .</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Cortex Analyst</div>
                        <div class="feature-description">Ask questions of structured data with leading accuracy for text-to-sql.</div>
                    </div>
                </div>
                <div style="margin-top: 30px; font-size: 15px; color: #666;">
                    Try asking: {onboarding_q}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    # ── chat history ──
    messages_container = st.container()
    
    with messages_container:
        for i, msg in enumerate(state.messages):
            avatar = "👤" if msg.role == "user" else "❄️"
            
            # Determine if this is a new message (for animations)
            is_new_message = state._s.get("new_message", False) and i == len(state.messages) - 1 and msg.role == "assistant"
            
            # Add message
            with st.chat_message(msg.role, avatar=avatar):
                render_message_content(msg)
                
                # SQL and visualization first (prioritize data)
                if msg.sql:
                    render_sql_block(msg, i)
                
                # Search results - only if not already handled by citations
                if msg.search_results and msg.role == "assistant" and not re.search(r'\[[A-Za-z0-9_-]+\]', msg.content):
                    with st.expander("View Source Documents", expanded=False):
                        render_search_results(msg.search_results)
                
                # Suggestions
                if msg.suggestions:
                    render_suggestions(msg.suggestions)
                
            # Reset animation flag after displaying
            if is_new_message:
                state._s["new_message"] = False

    # ── user input ──
    if prompt := st.chat_input("Help me MobyOne, you're my only hope..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        chat.handle_user_prompt(prompt)
        st.rerun()

    if state._s["debug"]:
        render_debug_panel()


if __name__ == "__main__":
    main()
