import streamlit as st
import pandas as pd
import json
import _snowflake
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, lit, concat_ws, lower
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# ----- CONFIGURATION -----
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 60000  # in milliseconds
MAX_DATAFRAME_ROWS = 1000
APP_VERSION = "2.0.0"
session = get_active_session()



# Custom plot template
custom_template = pio.templates["plotly_white"]
custom_template.layout.update(
    font_family="Inter, sans-serif",
    title_font_family="Inter, sans-serif",
    title_font_size=16,
    plot_bgcolor="rgba(250, 250, 252, 0.95)",
    paper_bgcolor="rgba(255, 255, 255, 0)",
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'color': '#333333'}
    },
    margin=dict(l=40, r=40, t=60, b=40)
)
pio.templates["custom_template"] = custom_template


# ----- MODELS -----
class AnalystService:
    """Data class for Cortex Analyst services."""
    def __init__(self, name: str, active: bool, database: str, schema: str, stage: str, file: str):
        self.name = name
        self.active = active
        self.database = database
        self.schema = schema
        self.stage = stage
        self.file = file
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Active': self.active,
            'Name': self.name,
            'Database': self.database,
            'Schema': self.schema,
            'Stage': self.stage, 
            'File': self.file
        }
    
    @classmethod
    def from_dataframe_row(cls, row) -> 'AnalystService':
        return cls(
            name=row['Name'],
            active=row['Active'],
            database=row['Database'],
            schema=row['Schema'],
            stage=row['Stage'],
            file=row['File']
        )


class SearchService:
    """Data class for Cortex Search services."""
    def __init__(self, name: str, database: str, schema: str, full_name: str, 
                active: bool = False, max_results: int = 1):
        self.name = name
        self.database = database
        self.schema = schema
        self.full_name = full_name
        self.active = active
        self.max_results = max_results
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Active': self.active,
            'Name': self.name,
            'Database': self.database,
            'Schema': self.schema,
            'Max Results': self.max_results,
            'Full Name': self.full_name
        }


class CustomTool:
    """Data class for custom tools."""
    def __init__(self, name: str, tool_type: str, active: bool = True):
        self.name = name
        self.type = tool_type
        self.active = active
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Active': self.active,
            'Name': self.name,
            'Type': self.type
        }


class Message:
    """Represents a chat message."""
    def __init__(self, role: str, content: str, msg_type: str = "text"):
        self.role = role
        self.content = content
        self.type = msg_type
        self.timestamp = datetime.now()
        
        # Additional properties for rich messages
        self.sql = None
        self.sql_df = None
        self.searchResults = None
        self.suggestions = None
        self.visualization = None
        self.viz_type = None
        self.message_index = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        result = {
            'role': self.role,
            'text': self.content,
            'type': self.type,
            'timestamp': self.timestamp
        }
        
        # Add extra properties if they exist
        for prop in ['sql', 'sql_df', 'searchResults', 'suggestions', 'visualization', 'viz_type', 'message_index']:
            value = getattr(self, prop, None)
            if value is not None:
                result[prop] = value
                
        return result
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to format expected by API."""
        if self.role == 'user':
            return {
                'role': 'user',
                'content': [{'type': 'text', 'text': self.content}]
            }
        elif self.role == 'assistant':
            return {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': self.content}]
            }
        # Skip other message types for API
        return None
        
    def __eq__(self, other):
        """Compare messages for equality."""
        if not isinstance(other, Message):
            return False
        return (self.role == other.role and 
                self.content == other.content and 
                self.type == other.type)
                
    def __str__(self):
        """String representation for debugging."""
        return f"Message({self.role}, {self.type}, content_length={len(self.content)})"


# ----- DATA ACCESS LAYER -----
class DataService:
    """Handles all data operations and caching."""
    def __init__(self, session):
        self.session = session
        
    def get_stages(self):
        """Get all available stages in the account."""
        try:
            stages = (
                self.session.sql('SHOW STAGES IN ACCOUNT')
                .filter(col('"type"') == 'INTERNAL NO CSE')
                .select(
                    col('"database_name"').alias('"Database"'),
                    col('"schema_name"').alias('"Schema"'),
                    col('"name"').alias('"Stage"')
                )
                .distinct()
                .order_by(['"Database"','"Schema"','"Stage"'])
            ).to_pandas()
            return stages
        except Exception as e:
            return pd.DataFrame(columns=['Database', 'Schema', 'Stage'])
    
    def get_files_from_stage(self, database, schema, stage):
        """Get YAML files from a specific stage."""
        try:
            files = (
                self.session.sql(f'LS @"{database}"."{schema}"."{stage}"')
                .filter(col('"size"') < 1000000)
                .filter(
                    (lower(col('"name"')).endswith('.yaml')) | 
                    (lower(col('"name"')).endswith('.yml'))
                )
                .select(
                    col('"name"').alias('"File Name"'),
                )
                .distinct()
                .order_by(['"File Name"'])
            ).to_pandas()
            return files
        except Exception as e:
            return pd.DataFrame(columns=['File Name'])
    
    def get_search_services(self):
        """Get all available Cortex Search services."""
        try:
            available_services = (
                self.session.sql('SHOW CORTEX SEARCH SERVICES IN ACCOUNT')
                .select(
                    col('"database_name"').alias('"Database"'),
                    col('"schema_name"').alias('"Schema"'),
                    col('"name"').alias('"Name"')
                )
                .with_column('"Full Name"', concat_ws(lit('.'), col('"Database"'), col('"Schema"'), col('"Name"')))
            ).to_pandas()
            available_services['Active'] = False
            available_services['Max Results'] = 3
            available_services = available_services[['Active','Name','Database','Schema','Max Results','Full Name']]
            return available_services
        except Exception as e:
            None
            return pd.DataFrame(columns=['Active', 'Name', 'Database', 'Schema', 'Max Results', 'Full Name'])
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return results as DataFrame."""
        try:
            sql = sql.strip()
            # Remove trailing semicolon if present
            if sql.endswith(';'):
                sql = sql[:-1]
                
            return self.session.sql(sql).limit(MAX_DATAFRAME_ROWS).to_pandas()
        except Exception as e:
            st.error(f"Error executing SQL: {str(e)}")
            return pd.DataFrame()


# ----- VISUALIZATION SERVICE -----
class VisualizationService:
    """Handles all visualization operations."""
    
    CHART_TYPES = ["bar", "line", "scatter", "pie", "histogram", "box", "area", "heatmap"]
    COLOR_PALETTES = {
        "default": px.colors.qualitative.Plotly,
        "pastel": px.colors.qualitative.Pastel,
        "vibrant": px.colors.qualitative.Vivid,
        "dark": px.colors.qualitative.Dark24,
        "light": px.colors.sequential.Tealgrn,
        "diverging": px.colors.diverging.RdBu
    }
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        
    def get_chart_suggestions(self, df: pd.DataFrame, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Get visualization suggestions using LLM."""
        if df.empty:
            return self._get_default_suggestions(df)
        
        try:
            # Get column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Set x and y axis defaults based on data types
            x_axis = datetime_cols[0] if datetime_cols else categorical_cols[0] if categorical_cols else df.columns[0]
            y_axis = numeric_cols[0] if numeric_cols else None
            
            # Determine chart type based on data
            chart_type = "line" if datetime_cols and numeric_cols else \
                         "bar" if categorical_cols and numeric_cols else \
                         "scatter" if len(numeric_cols) >= 2 else \
                         "histogram" if numeric_cols else "bar"
            
            # Try to get LLM suggestions
            if self.llm_service is not None:
                suggestions = self.llm_service.get_chart_suggestions(df, prompt)
                if suggestions:
                    # Merge with our smart defaults
                    if "chart_type" not in suggestions or not suggestions["chart_type"] in self.CHART_TYPES:
                        suggestions["chart_type"] = chart_type
                    if "x_axis" not in suggestions or not suggestions["x_axis"] in df.columns:
                        suggestions["x_axis"] = x_axis
                    if "y_axis" not in suggestions or (suggestions["y_axis"] and suggestions["y_axis"] not in df.columns):
                        suggestions["y_axis"] = y_axis
                    return suggestions
            
            # If LLM fails, use smart defaults
            return {
                "chart_type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "color": None,
                "title": "Data Visualization"
            }
        except Exception as e:
            return self._get_default_suggestions(df)
    
    def _get_default_suggestions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get default visualization suggestions."""
        return {
            "chart_type": "bar",
            "x_axis": df.columns[0] if not df.empty and len(df.columns) > 0 else "",
            "y_axis": df.columns[1] if not df.empty and len(df.columns) > 1 else None,
            "color": None,
            "title": "Data Visualization" 
        }
    
    def create_visualization(self, df: pd.DataFrame, suggestions: Dict[str, Any], 
                            message_index: int) -> go.Figure:
        """Create an interactive visualization."""
        # Get parameters from suggestions
        chart_type = suggestions.get("chart_type", "bar")
        x_axis = suggestions.get("x_axis", df.columns[0] if len(df.columns) > 0 else None)
        y_axis = suggestions.get("y_axis", df.columns[1] if len(df.columns) > 1 else None)
        color = suggestions.get("color", None)
        title = suggestions.get("title", "Data Visualization")
        
        # Choose appropriate chart function
        chart_map = {
            "bar": px.bar,
            "line": px.line,
            "scatter": px.scatter,
            "pie": px.pie,
            "histogram": px.histogram,
            "box": px.box,
            "area": px.area,
            "heatmap": px.imshow
        }
        
        # Handle empty dataframe
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available to visualize",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Prepare chart arguments
        args = {
            "data_frame": df,
            "title": title,
            "template": "custom_template",
        }
        
        # Configure based on chart type
        if chart_type == "pie":
            args.update({
                "names": x_axis,
                "values": y_axis if y_axis else df.select_dtypes(include=['number']).columns[0] 
                          if not df.select_dtypes(include=['number']).empty else df.columns[0],
                "color_discrete_sequence": self.COLOR_PALETTES["vibrant"]
            })
        elif chart_type == "histogram":
            args.update({
                "x": x_axis,
                "color": color,
                "opacity": 0.8,
                "nbins": min(20, len(df[x_axis].unique()) if x_axis in df else 20),
                "color_discrete_sequence": [self.COLOR_PALETTES["default"][0]]
            })
        elif chart_type == "heatmap":
            # Pivot data if needed for heatmap
            if len(df.columns) >= 3 and x_axis and y_axis:
                value_col = [c for c in df.columns if c != x_axis and c != y_axis][0]
                pivot_df = df.pivot_table(index=y_axis, columns=x_axis, values=value_col, aggfunc='mean')
                fig = px.imshow(
                    pivot_df,
                    title=title,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    template="custom_template"
                )
                return fig
            else:
                # Fallback to heatmap of correlation matrix for numeric data
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Matrix" if title == "Data Visualization" else title,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        template="custom_template"
                    )
                    return fig
                else:
                    # If no numeric data, fall back to bar chart
                    chart_type = "bar"
                    args = {
                        "data_frame": df,
                        "x": x_axis,
                        "title": title,
                        "template": "custom_template",
                    }
        else:
            args.update({
                "x": x_axis,
                "y": y_axis,
                "color": color,
                "color_discrete_sequence": self.COLOR_PALETTES["default"]
            })
        
        # Create chart
        try:
            fig = chart_map[chart_type](**args)
            
            # Add grid for most chart types
            if chart_type not in ["pie", "heatmap"]:
                fig.update_yaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor="rgba(226, 232, 240, 0.6)"
                )
                fig.update_xaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor="rgba(226, 232, 240, 0.6)"
                )
            
            # Add hover template improvements
            if chart_type in ["bar", "line", "scatter", "area"]:
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
                )
            
            # Add better legend placement
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization:<br>{str(e)}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
    
    def auto_visualize(self, df: pd.DataFrame, prompt: Optional[str] = None) -> Tuple[go.Figure, str]:
        """Automatically visualize dataframe with the best chart type."""
        if df.empty or len(df) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data to visualize",
                showarrow=False,
                font=dict(size=16)
            )
            return fig, "none"
        
        try:
            # Get chart suggestions
            suggestions = self.get_chart_suggestions(df, prompt)
            
            # Generate visualization
            message_index = len(st.session_state.get('messages', []))
            visualization = self.create_visualization(df, suggestions, message_index)
            
            return visualization, suggestions.get("chart_type", "bar")
        except Exception as e:
            
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Visualization error:<br>{str(e)}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig, "error"


# ----- LLM SERVICE -----
class LLMService:
    """Handles all LLM operations."""
    def __init__(self):
        pass
        
    def get_chart_suggestions(self, df: pd.DataFrame, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Use LLM to suggest chart parameters."""
        try:
            suggestion_prompt = f"""
            Analyze this dataframe structure and sample data to suggest visualization parameters using visual best practices.
            
            Columns: {df.columns.tolist()}
            Data Types: {df.dtypes.to_dict()}
            Sample: {df.head(3).to_dict()}
            
            Your response should be a JSON object in the following format only:
            {{
                "chart_type": "bar",     // Most appropriate chart type: bar, line, scatter, pie, histogram, box, area, or heatmap
                "x_axis": "{df.columns[0]}",        // X-axis column name
                "y_axis": "{df.columns[1] if len(df.columns) > 1 else ''}",        // Y-axis column name (if applicable)
                "color": "",         // Color grouping column (optional)
                "title": "Data Visualization"          // Chart title suggestion
            }}
            """
            
            # Add user query context if provided
            if prompt:
                suggestion_prompt = f"User query: {prompt}\n\n{suggestion_prompt}"
            
            # Make API call
            payload = {
                "model": st.session_state.get('agent_model', 'claude-3-5-sonnet'),
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": suggestion_prompt}]
                    }
                ]
            }
            
            # Call LLM API
            resp = _snowflake.send_snow_api_request(
                "POST",
                "/api/v2/cortex/llm:complete",
                {},  # headers
                {},  # query params
                payload,
                None,
                30000,  # timeout in milliseconds
            )
            
            # Parse response
            if resp and isinstance(resp, dict) and "content" in resp:
                content = resp["content"]
                if isinstance(content, str):
                    response_json = json.loads(content)
                    if isinstance(response_json, dict) and "content" in response_json:
                        suggestions = json.loads(response_json["content"])
                        return suggestions
                    return response_json
                elif isinstance(content, dict):
                    return content
            
            # If we couldn't parse properly, return empty dict
            return {}
                
        except Exception as e:
            return {}


# ----- API SERVICE -----
class APIService:
    """Handles all API calls to Cortex."""
    def __init__(self):
        pass
    
    def format_messages_for_api(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the API with proper alternation."""
        api_messages = []
        expected_role = 'user'  # Start with user
        
        # Group messages by conversation turn
        grouped_messages = []
        current_group = []
        
        for message in messages:
            # Start a new group if role changes
            if current_group and message.role != current_group[0].role:
                grouped_messages.append(current_group)
                current_group = [message]
            else:
                current_group.append(message)
                
        # Add the last group
        if current_group:
            grouped_messages.append(current_group)
        
        # Process each group as a single turn
        for group in grouped_messages:
            # Only process if this group matches the expected role in the alternating sequence
            if group and group[0].role == expected_role:
                # For user, just take the first message
                if expected_role == 'user':
                    first_msg = group[0]
                    api_message = {
                        'role': 'user',
                        'content': [{'type': 'text', 'text': first_msg.content}]
                    }
                    api_messages.append(api_message)
                    
                # For assistant, combine all text content
                elif expected_role == 'assistant':
                    # Only include actual text messages from assistant (not tool use/results)
                    content_list = []
                    for msg in group:
                        if msg.type == 'text':
                            content_list.append({'type': 'text', 'text': msg.content})
                    
                    # Only add if there's actual content
                    if content_list:
                        api_message = {
                            'role': 'assistant',
                            'content': content_list
                        }
                        api_messages.append(api_message)
                
                # Switch expected role
                expected_role = 'assistant' if expected_role == 'user' else 'user'
        
        # Ensure we have proper alternation
        if len(api_messages) > 1:
            for i in range(1, len(api_messages)):
                if api_messages[i]['role'] == api_messages[i-1]['role']:
                    # Keep only the last message of the same role
                    api_messages[i-1] = None
            
            api_messages = [m for m in api_messages if m is not None]
            
        return api_messages
    
    def get_tool_resources(self) -> Dict[str, Any]:
        """Build tool resources for API payload - simplified version."""
        tool_resources = {}
        
        # Add search services
        active_search = st.session_state.search_services[st.session_state.search_services['Active']]
        for _, row in active_search.iterrows():
            tool_resources[row['Name']] = {
                'name': row['Full Name'],
                'max_results': row['Max Results']
            }
        
        # Add analyst services
        active_analyst = st.session_state.analyst_services[st.session_state.analyst_services['Active']]
        for _, row in active_analyst.iterrows():
            # Direct file reference without stage in path
            tool_resources[row['Name']] = {
                'semantic_model_file': f"@{row['Database']}.{row['Schema']}.{row['File']}"
            }
            
        return tool_resources
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Build tools list for API payload - simplified version."""
        tools = st.session_state.tools.copy()
        
        # Add search services
        active_search = st.session_state.search_services[st.session_state.search_services['Active']]
        for _, row in active_search.iterrows():
            tools.append({
                'tool_spec': {
                    'type': 'cortex_search',
                    'name': row['Name']
                }
            })
        
        # Add analyst services
        active_analyst = st.session_state.analyst_services[st.session_state.analyst_services['Active']]
        for _, row in active_analyst.iterrows():
            tools.append({
                'tool_spec': {
                    'type': 'cortex_analyst_text_to_sql',
                    'name': row['Name']
                }
            })
        
        # Add custom tools
        active_custom = st.session_state.custom_tools[st.session_state.custom_tools['Active']]
        for _, row in active_custom.iterrows():
            tools.append({
                'tool_spec': {
                    'type': row['Type'],
                    'name': row['Name']
                }
            })
            
        return tools
    
    def generate_payload(self, message: str) -> Dict[str, Any]:
        """Generate API payload - simplified."""
        # Make a copy of messages for processing
        messages_copy = st.session_state.formatted_messages.copy()
        
        # Check if we need to add the current message (if it's not already the last user message)
        need_to_add_message = True
        if messages_copy and messages_copy[-1].role == 'user' and messages_copy[-1].content == message:
            need_to_add_message = False
            
        # Add current message if needed
        if need_to_add_message:
            # Create a new message object for the current message
            current_msg = Message('user', message)
            messages_copy.append(current_msg)
        
        # Format messages for API with proper alternation
        api_messages = self.format_messages_for_api(messages_copy)
        
        # Build payload
        payload = {
            "model": st.session_state.agent_model,
            "tools": self.get_tools(),
            "tool_resources": self.get_tool_resources(),
            "messages": api_messages
        }
        
        return payload
    
    def call_agent_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Cortex Agent API."""
        try:
            resp = _snowflake.send_snow_api_request(
                "POST",
                API_ENDPOINT,
                {},  # headers
                {'stream': True},  # query params
                payload,
                None,
                API_TIMEOUT
            )
            
            # Parse response - handle both string and dict returns
            if isinstance(resp, dict) and "content" in resp:
                content = resp["content"]
                if isinstance(content, str):
                    return json.loads(content)
                return content
            
            return resp
            
        except Exception as e:
            raise RuntimeError(f"Error calling Cortex Agent API: {str(e)}")


# ----- CHAT SERVICE -----
class ChatService:
    """Handles chat operations and message processing."""
    def __init__(self, data_service, api_service, viz_service):
        self.data_service = data_service
        self.api_service = api_service
        self.viz_service = viz_service
    
    def process_message(self, user_prompt: str) -> None:
        """Process a user message and get response."""
        try:
            # Create API payload
            payload = self.api_service.generate_payload(user_prompt)
            
            # Log API request
            st.session_state.api_history.append({'Request': payload})
            
            # Call API
            response = self.api_service.call_agent_api(payload)
            
            # Log API response
            st.session_state.api_history.append({'Response': response})
            
            # Process response
            self.format_bot_message(response, user_prompt)
            
            return True
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            return False
    
    def format_bot_message(self, data: List[Dict[str, Any]], user_query: str) -> None:
        """Format the bot's response from API data."""
        # Create a main assistant response message to hold all content
        main_response = Message('assistant', "", 'text')
        has_tool_use = False
        has_tool_results = False
        bot_text_message = ''
        
        # Process all message parts
        for message in data:
            if message.get('event') == 'error':
                # Handle errors separately
                self.handle_error_message(message.get('data', {}))
                return  # Exit early on error
            
            elif message.get('event') == 'done':
                break
            
            elif 'data' in message and 'delta' in message['data'] and 'content' in message['data']['delta']:
                for content in message['data']['delta']['content']:
                    if content['type'] == 'tool_use':
                        has_tool_use = True
                        if 'tool_use' in content and isinstance(content['tool_use'], dict):
                            tool_name = content['tool_use'].get('name', 'Unknown tool')
                            tool_message = f"I used the following tool to serve your request: **{tool_name}**\n\n"
                            bot_text_message += tool_message
                    
                    elif content['type'] == 'tool_results':
                        has_tool_results = True
                        # Extract and add tool results to the main message
                        tool_data = self.extract_tool_results(content, user_query)
                        
                        # Add extracted data to main response
                        if tool_data.get('searchResults'):
                            main_response.searchResults = tool_data['searchResults']
                            if not main_response.content:
                                main_response.content = 'I found the following relevant documents:'
                        
                        if tool_data.get('sql'):
                            main_response.sql = tool_data['sql']
                            main_response.sql_df = tool_data.get('sql_df')
                            main_response.visualization = tool_data.get('visualization')
                            main_response.viz_type = tool_data.get('viz_type')
                            main_response.message_index = len(st.session_state.messages)
                        
                        if tool_data.get('suggestions'):
                            main_response.suggestions = tool_data['suggestions']
                        
                        # Add any tool result text to the bot message
                        if tool_data.get('text'):
                            bot_text_message += tool_data['text'] + "\n\n"
                    
                    elif content['type'] == 'text':
                        bot_text_message += content['text']
        
        # Finalize the main response
        if bot_text_message.strip():
            main_response.content = bot_text_message.strip()
            
            # Add to display messages (just once)
            st.session_state.messages.append(main_response.to_dict())
            
            # Add to formatted messages for API (just once)
            st.session_state.formatted_messages.append(main_response)
    
    def handle_error_message(self, content: Dict[str, Any]) -> None:
        """Handle error messages from the API."""
        error_msg = 'Your query returned the following error:\n\n'
        
        if isinstance(content, dict):
            error_msg += f"**Error Code:** {content.get('code', 'Unknown')} \n\n"
            error_msg += f"**Error Message:** {content.get('message', 'Unknown error')} \n\n"
        else:
            error_msg += f"**Error:** {str(content)}"
        
        # Create message object
        msg = Message('assistant', error_msg, 'error')
        
        # Add to display messages
        st.session_state.messages.append(msg.to_dict())
        
        # Add to formatted messages
        st.session_state.formatted_messages.append(msg)
    
    def extract_tool_results(self, content: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Extract data from tool results content."""
        result = {
            'text': '',
            'searchResults': None,
            'sql': None,
            'sql_df': None,
            'visualization': None,
            'viz_type': None,
            'suggestions': None
        }
        
        if 'tool_results' not in content or not isinstance(content['tool_results'], dict):
            return result
            
        # Extract content from tool results
        tool_content = content['tool_results'].get('content', [{}])
        if not tool_content or not isinstance(tool_content, list):
            return result
            
        # Get JSON data
        json_data = tool_content[0].get('json', {}) if len(tool_content) > 0 else {}
        if not isinstance(json_data, dict):
            return result
        
        # Extract text content
        result['text'] = json_data.get('text', '')
        
        # Extract search results
        result['searchResults'] = json_data.get('searchResults', None)
        
        # Extract SQL and suggestions
        result['sql'] = json_data.get('sql', None)
        result['suggestions'] = json_data.get('suggestions', None)
        
        # Handle SQL results
        if result['sql'] and len(result['sql']) > 1:
            try:
                # Execute SQL
                result['sql_df'] = self.data_service.execute_sql(result['sql'])
                
                # Generate visualization if we have results
                if not result['sql_df'].empty:
                    result['visualization'], result['viz_type'] = self.viz_service.auto_visualize(
                        result['sql_df'], user_query
                    )
                
            except Exception as e:
                result['text'] = f"Error executing SQL query: {str(e)}"
        
        # Set default text based on content type if not already set
        if result['searchResults'] and not result['text']:
            result['text'] = 'I found the following relevant documents:'
        elif result['suggestions'] and not result['sql'] and not result['text']:
            result['text'] = json_data.get('text', 'You might want to try these questions:')
            
        return result


# ----- UI COMPONENTS -----
class UIComponents:
    """UI component definitions."""
    
    @staticmethod
    def load_css():
        """Load CSS styles and return logo URL."""
        vLogo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Snowflake_Logo.svg/1024px-Snowflake_Logo.svg.png'
        st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        /* Overall App Container */
        .stApp {
            background-color: #f7f9fc; 
            font-family: 'Inter', sans-serif;
        }
        /* Main content area */
        .main-container {
            max-width: 950px; 
            margin: auto;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.07);
            background-color: white;
        }
        /* Chat Messages */
        .chat-message {
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: flex-start;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        .chat-message:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        }
        .user-message {
            background-color: #ebf5ff;
            color: #0a4b9c;
            justify-content: flex-end;
            margin-left: 20%;
            border: 1px solid #d5e8ff;
        }
        .bot-message {
            background-color: #ffffff;
            color: #333333;
            justify-content: flex-start;
            margin-right: 20%;
            border: 1px solid #f0f0f0;
        }
        .avatar {
            font-size: 1.6rem;
            margin-right: 1rem;
            background-color: #f0f5ff;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .message-text {
            flex-grow: 1;
            word-wrap: break-word;
            margin-top: 0.2rem;
        }
        /* Input Box */
        .stTextInput>div>div>input {
            border-radius: 24px;
            padding: 0.8rem 1.3rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
            transition: all 0.3s;
            font-size: 1rem;
        }
        .stTextInput>div>div>input:focus {
            border-color: #2fb8ec;
            box-shadow: 0 0 0 3px rgba(47, 184, 236, 0.15);
            outline: none;
        }
        /* Send Button */
        .stButton>button {
            background-color: #2fb8ec;
            color: white;
            border-radius: 24px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0d8ecf;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
        }
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 16px;
            border-radius: 6px 6px 0 0;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ebf5ff !important;
            font-weight: 600;
        }
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #333;
            background-color: #f8f9fa;
            border-radius: 6px;
        }
        /* Data visualization */
        .chart-container {
            border-radius: 10px;
            background-color: white;
            padding: 16px;
            margin-top: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        /* Titles and Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            color: #333;
        }
        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: white;
            margin-right: 8px;
        }
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-active {
            background-color: #10b981;
        }
        .status-inactive {
            background-color: #d1d5db;
        }
        /* Welcome screen */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .welcome-emoji {
            font-size: 60px;
            margin-bottom: 2rem;
        }
        .welcome-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #333;
        }
        .welcome-subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 2rem;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            width: 100%;
            margin-top: 1.5rem;
        }
        .feature-card {
            background-color: #f8fafc;
            border-radius: 8px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .feature-icon {
            font-size: 28px;
            margin-bottom: 1rem;
        }
        .feature-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .feature-description {
            font-size: 14px;
            color: #666;
        }
        /* Search results enhancements */
        .search-results-container {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .search-results-header {
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .search-result-item {
            border-left: 3px solid #2fb8ec;
            padding-left: 12px;
            margin-bottom: 10px;
        }
        /* SQL results enhancements */
        .sql-results-container {
            margin-top: 16px;
        }
        .sql-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }
        .sql-stat-card {
            background-color: #f8fafc;
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .sql-stat-value {
            font-size: 20px;
            font-weight: 700;
            color: #2fb8ec;
        }
        .sql-stat-label {
            font-size: 12px;
            color: #666;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
        return vLogo
    
    @staticmethod
    def render_welcome_screen():
        """Render welcome screen for new chats."""
        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-emoji">👋</div>
                <div class="welcome-title">Welcome to Cortex Agent</div>
                <div class="welcome-subtitle">Your interactive data assistant powered by Snowflake</div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Search Services</div>
                        <div class="feature-description">Find information across your documentation and knowledge bases</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Data Analysis</div>
                        <div class="feature-description">Ask questions about your data and get visualized results</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">💬</div>
                        <div class="feature-title">Natural Conversations</div>
                        <div class="feature-description">Maintain context across multi-turn conversations</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">⚙️</div>
                        <div class="feature-title">Custom Tools</div>
                        <div class="feature-description">Connect specialized tools to extend capabilities</div>
                    </div>
                </div>
                <div style="margin-top: 2rem;">
                    <p style="color: #666;">Start by configuring services in the sidebar and asking a question below</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def display_search_results(results, expanded=False):
        """Display search results in an improved format."""
        with st.expander(f"📄 Search Results ({len(results)} documents)", expanded=expanded):
            for i, doc in enumerate(results):
                with stylable_container(
                    f"search_result_{i}",
                    css_styles="""
                    {
                        border: 1px solid #e0e7ff;
                        border-radius: 8px;
                        padding: 14px;
                        margin-bottom: 12px;
                        background-color: #f5f8ff;
                    }
                    """
                ):
                    # Header with source and score
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{doc.get('source_id', doc.get('title', 'Document'))}**")
                    with col2:
                        if 'score' in doc:
                            st.markdown(f"<span style='color:#666;font-size:12px;float:right;'>Relevance: {float(doc.get('score', 0)):.2f}</span>", unsafe_allow_html=True)
                    
                    # Content
                    st.markdown(doc.get('text', doc.get('content', 'No content available')))
                    
                    # Metadata
                    metadata_items = {k: v for k, v in doc.items() 
                                    if k not in ['source_id', 'text', 'title', 'content', 'score']}
                    
                    if metadata_items:
                        for key, value in metadata_items.items():
                            st.markdown(f"**{key}**: {value}")
    
    @staticmethod
    def display_sql_visualization(message, df):
        """Display SQL visualization with improved UI."""
        # Create tabs for data, visualization, and SQL
        tabs = st.tabs(["📋 Data Table", "📊 Visualization", "🔍 SQL Query"])
        
        with tabs[0]:  # Data Table tab
            # Stats above the dataframe
            row_count = len(df)
            col_count = len(df.columns)
            
            # Determine if there are numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_stats = ""
            if len(numeric_cols) > 0:
                first_col = numeric_cols[0]
                total = df[first_col].sum()
                avg = df[first_col].mean()
                numeric_stats = f"Sum of {first_col}: {total:,.2f} | Avg: {avg:,.2f}"
            
            st.markdown(f"<span style='color:#666;font-size:13px;'>Showing {row_count:,} rows and {col_count} columns. {numeric_stats}</span>", unsafe_allow_html=True)
            
            # Display dataframe with styling
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        with tabs[1]:  # Visualization tab
            if message.get('visualization') is not None:
                # Get message index for the interactive controls
                message_index = message.get('message_index', 0)
                
                # Display stats in a nicer format
                if len(df) > 0:
                    stats_cols = st.columns(3)
                    with stats_cols[0]:
                        st.markdown(
                            f"""
                            <div class="sql-stat-card">
                                <div class="sql-stat-value">{len(df):,}</div>
                                <div class="sql-stat-label">Rows</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    with stats_cols[1]:
                        st.markdown(
                            f"""
                            <div class="sql-stat-card">
                                <div class="sql-stat-value">{len(df.columns)}</div>
                                <div class="sql-stat-label">Columns</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    with stats_cols[2]:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            col_sum = df[numeric_cols[0]].sum()
                            st.markdown(
                                f"""
                                <div class="sql-stat-card">
                                    <div class="sql-stat-value">{col_sum:,.1f}</div>
                                    <div class="sql-stat-label">Sum of {numeric_cols[0]}</div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                
                # Display chart
                chart_container = st.container()
                with chart_container:
                    if f"custom_chart_{message_index}" not in st.session_state:
                        st.plotly_chart(message['visualization'], use_container_width=True, key=f"vis_chart_{message_index}")
                    else:
                        st.plotly_chart(st.session_state[f"custom_chart_{message_index}"], use_container_width=True)
                
                # Chart customization
                with st.expander("✨ Customize Visualization", expanded=False):
                    st.markdown("""
                                    <div style="margin-bottom: 15px; font-size: 0.95rem; color: #334155;">
                                        Adjust the chart parameters to explore your data in different ways
                                    </div>
                                """, unsafe_allow_html=True)
                                        
                    # Chart controls in two columns with modern styling
                    col1, col2 = st.columns(2)
                    with col1:
                        chart_type = st.selectbox(
                            "Chart Type",
                            ["bar", "line", "scatter", "pie", "histogram", "box", "area"],
                            index=["bar", "line", "scatter", "pie", "histogram", "box", "area"].index(
                                st.session_state.get(f"chart_type_{message_index}", 
                                message.get('viz_type', 'bar')) if st.session_state.get(f"chart_type_{message_index}", 
                                message.get('viz_type', 'bar')) in 
                                ["bar", "line", "scatter", "pie", "histogram", "box", "area"] 
                                else "bar"
                            ),
                            key=f"chart_type_{message_index}"
                        )
                        
                        x_columns = df.columns.tolist()
                        default_x = st.session_state.get(f"x_axis_{message_index}", x_columns[0] if x_columns else None)
                        x_axis = st.selectbox(
                            "X-Axis",
                            x_columns,
                            index=x_columns.index(default_x) if default_x in x_columns else 0,
                            key=f"x_axis_{message_index}"
                        )
                        
    
                    
                    with col2:
                        color_options = [None] + df.columns.tolist()
                        default_color = st.session_state.get(f"color_{message_index}", None)
                        color = st.selectbox(
                            "Color By",
                            color_options,
                            index=color_options.index(default_color) if default_color in color_options else 0,
                            key=f"color_{message_index}"
                        )
                        
                        # Y-axis selector (conditional based on chart type)
                        y_columns = [None] + df.columns.tolist()
                        if chart_type not in ["histogram", "pie"]:
                            selectbox_key = f"y_axis_{message_index}"
                            
                            # If the key doesn't exist in session state yet, initialize with a default
                            if selectbox_key not in st.session_state:
                                default_y = y_columns[1] if len(y_columns) > 1 else None
                                st.session_state[selectbox_key] = default_y
                            
                            # Now use the selectbox with just the key, no index parameter
                            y_axis = st.selectbox(
                                "Y-Axis",
                                options=y_columns,
                                key=selectbox_key
                            )
                        else:
                            # For histogram and pie, no Y-axis selection needed
                            y_axis = None if chart_type == "histogram" else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        
                            
                    # Apply changes button
                    if st.button("Apply Changes", use_container_width=True, key=f"apply_chart_{message_index}"):
                        # Configure chart parameters based on selections
                        chart_map = {
                            "bar": px.bar,
                            "line": px.line,
                            "scatter": px.scatter,
                            "pie": px.pie,
                            "histogram": px.histogram,
                            "box": px.box,
                            "area": px.area
                        }
    
                        # Basic arguments for the chart
                        args = {
                            "data_frame": df,
                            "template": "plotly_white",
                        }
    
                        # Add conditional parameters based on chart type
                        if chart_type == "pie":
                            args.update({
                                "names": x_axis,
                                "values": y_axis if chart_type != "histogram" else None,
                                "color": x_axis,
                                "color_discrete_sequence": px.colors.sequential.Viridis
                            })
                        elif chart_type == "histogram":
                            args.update({
                                "x": x_axis,
                                "color": color,
                                "opacity": 0.8,
                                "nbins": 20
                            })
                        else:
                            args.update({
                                "x": x_axis,
                                "y": y_axis,
                                "color": color,
                                "labels": {
                                    x_axis: x_axis,
                                    y_axis: y_axis if y_axis else ""
                                }
                            })
                        
                        # Create the chart
                        try:
                            fig = chart_map[chart_type](**args)
                            
                            # Apply consistent styling
                            fig.update_layout(
                                font_family="Inter, sans-serif",
                                title_font_family="Plus Jakarta Sans, sans-serif",
                                title_font_size=16,
                                plot_bgcolor="rgba(250, 250, 252, 0.9)",
                                paper_bgcolor="rgba(255, 255, 255, 0)",
                                title={
                                    'x': 0.5,
                                    'xanchor': 'center'
                                },
                                margin=dict(l=40, r=40, t=60, b=40),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Add grid lines for most chart types
                            if chart_type not in ["pie"]:
                                fig.update_yaxes(
                                    showgrid=True, 
                                    gridwidth=1, 
                                    gridcolor="rgba(226, 232, 240, 0.6)"
                                )
                                fig.update_xaxes(
                                    showgrid=True, 
                                    gridwidth=1, 
                                    gridcolor="rgba(226, 232, 240, 0.6)"
                                )
                            
                            # Store the chart in session state
                            st.session_state[f"custom_chart_{message_index}"] = fig
                            # Force a rerun to update the visualization
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error generating chart: {e}")
                
                # Add context about the visualization type
                if 'viz_type' in message:
                    viz_type = message['viz_type']
                    viz_descriptions = {
                        "time_series": "📈 Time Series visualization showing trends over time",
                        "scatter": "📊 Scatter plot showing the relationship between variables",
                        "bar": "📊 Bar chart showing key metrics from your query",
                        "categorical_bar": "📊 Bar chart comparing values across categories",
                        "histogram": "📊 Histogram showing the distribution of values",
                        "count": "📊 Count plot showing frequency of categories",
                        "pie": "🥧 Pie chart showing proportion of categories",
                        "box": "📦 Box plot showing distribution statistics",
                        "area": "📊 Area chart showing cumulative values over a dimension"
                    }
                    st.caption(viz_descriptions.get(viz_type, f"📊 {viz_type.title()} chart based on your data"))
            else:
                st.info("No visualization available for this data.", icon="ℹ️")
        
        with tabs[2]:  # SQL Query tab
            st.code(message['sql'], language='sql')
    
    @staticmethod
    def display_suggestions(suggestions, message_index=0):
        """Display suggested questions in an improved format."""
        with st.expander("❓ Suggested Questions", expanded=True):
            for i, suggestion in enumerate(suggestions):
                suggest_key = f"suggest_{message_index}_{i}"
                if st.button(suggestion, key=suggest_key, use_container_width=True):
                    return suggestion
        return None


# ----- APP STATE -----
# Initialize session state variables
def init_session_state():
    """Initialize all session state variables."""
    # Chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'formatted_messages' not in st.session_state:
        st.session_state.formatted_messages = []
    
    # API history
    if 'api_history' not in st.session_state:
        st.session_state.api_history = []
    
    # Services & Tools
    if 'analyst_services' not in st.session_state:
        st.session_state.analyst_services = pd.DataFrame(
            columns=['Active', 'Name', 'Database', 'Schema', 'Stage', 'File']
        )
    
    if 'search_services' not in st.session_state:
        st.session_state.search_services = pd.DataFrame(
            columns=['Active', 'Name', 'Database', 'Schema', 'Max Results', 'Full Name']
        )
    
    if 'tools' not in st.session_state:
        st.session_state.tools = []
    
    if 'custom_tools' not in st.session_state:
        st.session_state.custom_tools = pd.DataFrame(
            columns=['Active', 'Name', 'Type']
        )
    
    # Data stores
    if 'stages' not in st.session_state:
        st.session_state.stages = pd.DataFrame()
    
    # Configuration
    if 'agent_model' not in st.session_state:
        st.session_state.agent_model = 'claude-3-5-sonnet'
    
    # UI state
    if 'active_suggestion' not in st.session_state:
        st.session_state.active_suggestion = None

    # Feature toggles
    if 'enable_animations' not in st.session_state:
        st.session_state.enable_animations = True
        
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
        
    # Performance tracking
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []

def reset_chat():
    """Reset chat but keep configuration."""
    st.session_state.messages = []
    st.session_state.formatted_messages = []
    st.session_state.api_history = []
    st.session_state.active_suggestion = None
    st.session_state.response_times = []

def ensure_valid_message_sequence():
    """Ensure message sequence is valid for API (alternate user/assistant) by combining consecutive messages."""
    if len(st.session_state.formatted_messages) > 0:
        # First pass: combine consecutive messages with the same role
        combined_messages = []
        current_group = []
        
        for msg in st.session_state.formatted_messages:
            if not current_group or msg.role == current_group[0].role:
                # Add to current group if roles match
                current_group.append(msg)
            else:
                # Process the completed group
                if current_group:
                    # Create a combined message from the group
                    combined_msg = Message(
                        role=current_group[0].role,
                        content="\n\n".join([m.content for m in current_group if m.content]),
                        msg_type=current_group[0].type
                    )
                    
                    # For assistant messages, preserve all metadata from messages in the group
                    if combined_msg.role == 'assistant':
                        # Merge data properties from all messages in the group
                        for prop in ['sql', 'searchResults', 'suggestions', 'viz_type']:
                            for m in current_group:
                                val = getattr(m, prop, None)
                                if val is not None:
                                    setattr(combined_msg, prop, val)
                        
                        # Take the most recent SQL DataFrame and visualization if available
                        for m in reversed(current_group):
                            if hasattr(m, 'sql_df') and m.sql_df is not None:
                                combined_msg.sql_df = m.sql_df
                            if hasattr(m, 'visualization') and m.visualization is not None:
                                combined_msg.visualization = m.visualization
                            if hasattr(m, 'message_index') and m.message_index is not None:
                                combined_msg.message_index = m.message_index
                                break
                    
                    combined_messages.append(combined_msg)
                # Start a new group
                current_group = [msg]
        
        # Add the last group if any
        if current_group:
            combined_msg = Message(
                role=current_group[0].role,
                content="\n\n".join([m.content for m in current_group if m.content]),
                msg_type=current_group[0].type
            )
            
            # For assistant messages, preserve metadata
            if combined_msg.role == 'assistant':
                for prop in ['sql', 'searchResults', 'suggestions', 'viz_type']:
                    for m in current_group:
                        val = getattr(m, prop, None)
                        if val is not None:
                            setattr(combined_msg, prop, val)
                
                # Take the most recent SQL DataFrame and visualization
                for m in reversed(current_group):
                    if hasattr(m, 'sql_df') and m.sql_df is not None:
                        combined_msg.sql_df = m.sql_df
                    if hasattr(m, 'visualization') and m.visualization is not None:
                        combined_msg.visualization = m.visualization
                    if hasattr(m, 'message_index') and m.message_index is not None:
                        combined_msg.message_index = m.message_index
                        break
            
            combined_messages.append(combined_msg)
        
        # Second pass: Ensure alternating roles (user/assistant/user...)
        valid_sequence = []
        expected_role = 'user'  # Start with user
        
        for msg in combined_messages:
            if msg.role == expected_role:
                valid_sequence.append(msg)
                # Switch expected role
                expected_role = 'assistant' if expected_role == 'user' else 'user'
            elif not valid_sequence or valid_sequence[-1].role != msg.role:
                # If we have an unexpected role but it doesn't cause consecutive same roles,
                # we can still include it to preserve more of the conversation
                valid_sequence.append(msg)
                # Reset expected role based on what we just added
                expected_role = 'assistant' if msg.role == 'user' else 'user'
        
        # Update the session state with our cleaned sequence
        st.session_state.formatted_messages = valid_sequence
        
        # Also update display messages to match
        st.session_state.messages = [msg.to_dict() for msg in valid_sequence]
    
    # If we have no messages, ensure we're ready to start with a user message
    elif len(st.session_state.messages) > 0:
        # Rebuild formatted_messages from display messages
        st.session_state.formatted_messages = []
        
        expected_role = 'user'
        for msg in st.session_state.messages:
            role = msg.get('role')
            if role == expected_role:
                formatted_msg = Message(
                    role=role,
                    content=msg.get('text', ''),
                    msg_type=msg.get('type', 'text')
                )
                st.session_state.formatted_messages.append(formatted_msg)
                expected_role = 'assistant' if expected_role == 'user' else 'user'


# ----- DIALOGS -----
# Dialog management functions are simplified with direct access to session state
@st.dialog("Manage Cortex Search Services", width='large')
def manage_search_services():
    """Dialog to manage Cortex Search services."""
    st.subheader('Manage Cortex Search Services', anchor=False)
    st.markdown('Activate or deactivate Cortex Search Services for your Agent.')
    st.divider()
    
    # Refresh data if needed
    if st.session_state.search_services.empty:
        try:
            # Direct query without caching
            services = session.sql('SHOW CORTEX SEARCH SERVICES IN ACCOUNT').select(
                col('"database_name"').alias('"Database"'),
                col('"schema_name"').alias('"Schema"'),
                col('"name"').alias('"Name"')
            ).with_column(
                '"Full Name"', concat_ws(lit('.'), col('"Database"'), col('"Schema"'), col('"Name"'))
            ).to_pandas()
            
            services['Active'] = False
            services['Max Results'] = 1
            services = services[['Active','Name','Database','Schema','Max Results','Full Name']]
            st.session_state.search_services = services
        except Exception as e:
            st.error(f"Error fetching search services: {e}")
            st.session_state.search_services = pd.DataFrame(
                columns=['Active','Name','Database','Schema','Max Results','Full Name']
            )
    
    services_df = st.data_editor(
        st.session_state.search_services, 
        use_container_width=True,
        column_config={
            "Active": st.column_config.CheckboxColumn(
                "Active",
                help="Enable/disable this service",
                width="small",
            ),
            "Max Results": st.column_config.NumberColumn(
                "Max Results",
                help="Maximum number of search results to return",
                min_value=1,
                max_value=10,
                step=1,
                width="small",
            ),
        },
        disabled=["Database", "Schema", "Full Name"]
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button('Update Services', use_container_width=True):
            st.session_state.search_services = services_df
            st.rerun()

@st.dialog("Manage Cortex Analyst Services", width='large')
def manage_analyst_services():
    """Dialog to manage Cortex Analyst services."""
    st.subheader('Manage Cortex Analyst Services', anchor=False)
    st.markdown('Add new Semantic Models or manage existing ones.')
    
    task = st.radio("Select action:", ['Add a new Service', 'Manage existing Services'], horizontal=True)
    st.divider()
    
    # Add new service
    if task == 'Add a new Service':
        # Ensure stages are loaded
        if st.session_state.stages.empty:
            try:
                # Direct query without caching
                stages = session.sql('SHOW STAGES IN ACCOUNT').filter(
                    col('"type"') == 'INTERNAL NO CSE'
                ).select(
                    col('"database_name"').alias('"Database"'),
                    col('"schema_name"').alias('"Schema"'),
                    col('"name"').alias('"Stage"')
                ).distinct().order_by(
                    ['"Database"','"Schema"','"Stage"']
                ).to_pandas()
                
                st.session_state.stages = stages
            except Exception as e:
                st.error(f"Error fetching stages: {e}")
                st.session_state.stages = pd.DataFrame(columns=['Database', 'Schema', 'Stage'])
        
        stages = st.session_state.stages
        
        if stages.empty:
            st.warning("No stages found in your account. Please create a stage first.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            database = st.selectbox('Database:', sorted(set(stages['Database'])))
        with col2:
            schema_options = sorted(set(stages[stages['Database'] == database]['Schema']))
            schema = st.selectbox('Schema:', schema_options)
        
        stage_options = sorted(set(stages[(stages['Database'] == database) & (stages['Schema'] == schema)]['Stage']))
        stage = st.selectbox('Stage:', stage_options)
        
        # Fetch files directly without caching
        try:
            files = session.sql(f'LS @"{database}"."{schema}"."{stage}"').filter(
                col('"size"') < 1000000
            ).filter(
                (lower(col('"name"')).endswith('.yaml')) | 
                (lower(col('"name"')).endswith('.yml'))
            ).select(
                col('"name"').alias('"File Name"'),
            ).distinct().order_by(
                ['"File Name"']
            ).to_pandas()
        except Exception as e:
            st.error(f"Error fetching files: {e}")
            files = pd.DataFrame(columns=['File Name'])
        
        if not files.empty:
            file = st.selectbox('YAML File:', files['File Name'])
            
            col1, col2 = st.columns(2)
            with col1:
                default_name = file.split('/')[-1].split('.')[0] if '/' in file else file.split('.')[0]
                name = st.text_input('Service Name:', value=default_name)
            
            with col2:
                st.write("")
                st.write("")
                if st.button('Add Service', use_container_width=True):
                    # Check if service with same name already exists
                    if name in st.session_state.analyst_services['Name'].values:
                        st.error(f"Service with name '{name}' already exists!")
                    else:
                        new_service = {'Active': True, 'Name': name, 'Database': database, 'Schema': schema, 'Stage': stage, 'File': file}
                        st.session_state.analyst_services.loc[len(st.session_state.analyst_services)] = new_service
                        st.success(f"Added service '{name}'")
                        st.rerun()
        else:
            st.info('No YAML files smaller than 1MB found in selected stage.', icon="ℹ️")
    
    # Manage existing services
    if task == 'Manage existing Services':
        if st.session_state.analyst_services.empty:
            st.info('No analyst services have been added yet.', icon="ℹ️")
        else:
            services_df = st.data_editor(
                st.session_state.analyst_services, 
                use_container_width=True,
                column_config={
                    "Active": st.column_config.CheckboxColumn(
                        "Active",
                        help="Enable/disable this service",
                        width="small",
                    ),
                    "Name": st.column_config.TextColumn(
                        "Name",
                        help="Service name",
                        width="medium",
                    ),
                },
                disabled=["Database", "Schema", "Stage", "File"]
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button('Update Services', use_container_width=True):
                    st.session_state.analyst_services = services_df
                    st.rerun()

@st.dialog("API History", width='large')
def display_api_call_history():
    st.subheader("API Call History", anchor=False)
    st.markdown("View all API requests and responses from this session.")
    st.divider()
    
    if not st.session_state.api_history:
        st.info("No API calls have been made yet.", icon="ℹ️")
        return
        
    for i, message in enumerate(st.session_state.api_history):
        if 'Request' in message:
            st.subheader(f"Request #{(i//2)+1}", anchor=False)
            with st.expander("View request payload", expanded=False):
                st.json(message['Request'])
        
        if 'Response' in message:
            st.subheader(f"Response #{(i//2)+1}", anchor=False)
            with st.expander("View response data", expanded=False):
                st.json(message['Response'])
            
        if i < len(st.session_state.api_history) - 1:
            st.divider()

@st.dialog("Manage Custom Tools", width='large')
def manage_custom_tools():
    """Dialog to manage custom tools."""
    st.subheader('Custom Tools Management', anchor=False)
    st.info('Other tools besides Cortex Search and Cortex Analyst are in preview!', icon="ℹ️")
    
    task = st.radio("Select action:", ['Add a new Tool', 'Manage existing Tools'], horizontal=True)
    st.divider()
    
    # Add new tool
    if task == 'Add a new Tool':
        tool_type = st.selectbox('Tool Type:', ['SQL Execution', 'Custom'])
        
        if tool_type == 'Custom':
            col1, col2 = st.columns(2)
            with col1:
                type_value = st.text_input('Tool Type:')
            with col2:
                name_value = st.text_input('Tool Name:')
        else:  # SQL Execution
            col1, col2 = st.columns(2)
            with col1:
                type_value = st.text_input('Tool Type:', value='cortex_analyst_sql_exec', disabled=True)
            with col2:
                name_value = st.text_input('Tool Name:')
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button('Add Tool', use_container_width=True):
                if not name_value or not type_value:
                    st.error("Tool name and type are required!")
                else:
                    new_tool = {'Active': True, 'Name': name_value, 'Type': type_value}
                    st.session_state.custom_tools.loc[len(st.session_state.custom_tools)] = new_tool
                    st.success(f"Added tool '{name_value}'")
                    st.rerun()
    
    # Manage existing tools
    if task == 'Manage existing Tools':
        if st.session_state.custom_tools.empty:
            st.info('No custom tools have been added yet.', icon="ℹ️")
        else:
            tools_df = st.data_editor(
                st.session_state.custom_tools, 
                use_container_width=True,
                column_config={
                    "Active": st.column_config.CheckboxColumn(
                        "Active",
                        help="Enable/disable this tool",
                        width="small",
                    ),
                }
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button('Update Tools', use_container_width=True):
                    st.session_state.custom_tools = tools_df
                    st.rerun()
    
    @staticmethod
    @st.dialog("API Payload Preview", width='large')
    def display_payload(api_service, state):
        """Dialog to display API payload preview."""
        st.subheader("API Payload Configuration", anchor=False)
        st.markdown("Preview the current payload that will be sent to the API.")
        st.divider()
        
        prompt = st.text_input('Sample Question:', value='Tell me about our data')
        
        payload = api_service.generate_payload(
            message=prompt,
            messages=state.formatted_messages,
            model=state.agent_model,
            search_services=state.search_services,
            analyst_services=state.analyst_services,
            custom_tools=state.custom_tools,
            base_tools=state.tools
        )
        
        st.json(payload, expanded=True)
    
    @staticmethod
    @st.dialog("API History", width='large')
    def display_api_history(state):
        """Dialog to display API call history."""
        st.subheader("API Call History", anchor=False)
        st.markdown("View API requests and responses from this session.")
        st.divider()
        
        if not state.api_history:
            st.info("No API calls have been made yet.", icon="ℹ️")
            return
        
        with st.expander("Download API history as JSON", expanded=False):
            st.download_button(
                "Download API History",
                data=json.dumps(state.api_history, indent=2),
                file_name=f"cortex_api_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
        for i, message in enumerate(state.api_history):
            if 'Request' in message:
                st.subheader(f"Request #{(i//2)+1}", anchor=False)
                with st.expander("View request payload", expanded=False):
                    st.json(message['Request'])
            
            if 'Response' in message:
                st.subheader(f"Response #{(i//2)+1}", anchor=False)
                with st.expander("View response data", expanded=False):
                    st.json(message['Response'])
                
            if i < len(state.api_history) - 1:
                st.divider()
    
    @staticmethod
    @st.dialog("Message History", width='large')
    def display_message_history(state):
        """Dialog to display message history in detail."""
        st.subheader("Chat Message History", anchor=False)
        st.markdown("View the internal message objects in the conversation.")
        st.divider()
        
        if not state.messages:
            st.info("No messages have been exchanged yet.", icon="ℹ️")
            return
        
        tab1, tab2 = st.tabs(["Display Messages", "API-Formatted Messages"])
        
        with tab1:
            for i, message in enumerate(state.messages):
                role = message.get('role')
                if role == 'assistant':
                    st.subheader(f"Assistant Message #{i+1}", anchor=False)
                    with st.expander(f"View message ({message.get('type', 'text')})", expanded=False):
                        st.json(message)
                
                elif role == 'user':
                    st.subheader(f"User Message #{i+1}", anchor=False)
                    with st.expander("View message", expanded=False):
                        st.json(message)
                
                elif role == '❗':
                    st.subheader(f"Alert Message #{i+1}", anchor=False)
                    with st.expander("View message", expanded=False):
                        st.write(message)
                
                if i < len(state.messages) - 1:
                    st.divider()
        
        with tab2:
            if not state.formatted_messages:
                st.info("No API-formatted messages available.", icon="ℹ️")
            else:
                for i, message in enumerate(state.formatted_messages):
                    st.subheader(f"Message #{i+1}: {message.role}", anchor=False)
                    with st.expander(f"View message ({message.type})", expanded=False):
                        st.write(f"Role: {message.role}")
                        st.write(f"Type: {message.type}")
                        st.write(f"Timestamp: {message.timestamp}")
                        st.write("Content:")
                        st.markdown(message.content)
                    
                    if i < len(state.formatted_messages) - 1:
                        st.divider()
    
    @staticmethod
    @st.dialog("App Settings", width='medium')
    def display_settings():
        """Dialog to display and modify app settings."""
        st.subheader("Application Settings", anchor=False)
        st.markdown("Customize your Cortex Agent experience.")
        st.divider()
        
        st.toggle("Enable animations", value=st.session_state.get('enable_animations', True), 
                 key="enable_animations")
        
        st.divider()
        st.subheader("Danger Zone", anchor=False)
        
        if st.button("Reset All Settings", type="primary", use_container_width=True):
            # Reset will happen on rerun
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()


# ----- MAIN APPLICATION -----
def main():
    """Main application entry point."""
        
    # Create state manager and ensure it's initialized
    if 'initialized' not in st.session_state:
        init_session_state()
        st.session_state.initialized = True
    
    # Initialize services (with minimal dependencies, avoiding caching issues)
    data_service = DataService(session)
    llm_service = LLMService()
    viz_service = VisualizationService(llm_service)
    api_service = APIService()
    chat_service = ChatService(data_service, api_service, viz_service)
    
    # Load UI components
    ui = UIComponents()
    vLogo = ui.load_css()
    
    # Fix message alternation issues if needed
    ensure_valid_message_sequence()
    
    #################
    # SIDEBAR UI
    #################
    with st.sidebar:
        st.image(vLogo, width=200)
        
        st.markdown("---")
        
        # Model selector
        model_options = ['claude-3-5-sonnet', 'mistral-large2', 'llama3.3-70b']
        
        st.session_state.agent_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(st.session_state.agent_model) if st.session_state.agent_model in model_options else 0
        )
        
        # Actions section
        st.markdown("### Actions")
        
        if st.button('New Chat', use_container_width=True, icon='🔄'):
            reset_chat()
            st.rerun()
        
        # Configuration container
        with stylable_container(
            "config_container",
            css_styles="""
                {
                    border-radius: 10px;
                    background-color: white;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                    padding: 15px;
                    margin-top: 20px;
                }
            """
        ):
            st.markdown("### Configuration")
            
            # Service controls with icons and tooltips
            col1, col2 = st.columns(2)
            with col1:
                search_count = sum(st.session_state.search_services['Active']) if not st.session_state.search_services.empty else 0
                search_button = st.button(
                    f'🔍 Search ({search_count})', 
                    use_container_width=True,
                    help="Manage Cortex Search Services"
                )
                if search_button:
                    manage_search_services()
            
            with col2:
                analyst_count = sum(st.session_state.analyst_services['Active']) if not st.session_state.analyst_services.empty else 0
                analyst_button = st.button(
                    f'📊 Analyst ({analyst_count})', 
                    use_container_width=True,
                    help="Manage Cortex Analyst Services"
                )
                if analyst_button:
                    manage_analyst_services()
            
            col1, col2 = st.columns(2)
            with col1:
                tools_count = sum(st.session_state.custom_tools['Active']) if not st.session_state.custom_tools.empty else 0
                tools_button = st.button(
                    f'🧰 Tools ({tools_count})', 
                    use_container_width=True,
                    help="Manage Custom Tools"
                )
                if tools_button:
                    manage_custom_tools()

            with col2:
                history_button = st.button(
                    '📚 API History', 
                    use_container_width=True,
                    help="View API History"
                )
                if history_button:
                    display_api_call_history()

        
        # Status indicators
        st.markdown("---")
        st.markdown("### Status")
        
        # Active services and tools
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "#2fb8ec" if search_count else "#d3d3d3"
            st.markdown(f"<div class='badge' style='background-color: {status};'>Search</div>", unsafe_allow_html=True)
        with col2:
            status = "#2fb8ec" if analyst_count else "#d3d3d3"
            st.markdown(f"<div class='badge' style='background-color: {status};'>Analyst</div>", unsafe_allow_html=True)
        with col3:
            status = "#2fb8ec" if tools_count else "#d3d3d3"
            st.markdown(f"<div class='badge' style='background-color: {status};'>Tools</div>", unsafe_allow_html=True)
        
        # Credits and version
        st.markdown("---")
        st.markdown(f"<div style='text-align: center; color: #888; font-size: 0.8em;'>Snowflake Cortex Agent v{APP_VERSION}</div>", unsafe_allow_html=True)
    
    #################
    # MAIN CHAT UI
    #################
    with stylable_container(
        "main_container",
        css_styles="""
            {
                background-color: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
                margin-bottom: 20px;
            }
        """
    ):
        # Display chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Welcome screen for new chats
            if not st.session_state.messages:
                ui.render_welcome_screen()
            else:
                # Display chat history
                for message in st.session_state.messages:
                    # Get message properties
                    role = message.get("role")
                    
                    # Determine avatar
                    avatar = "👤" if role == "user" else "❄️" if role == 'assistant' else None
                    
                    # Display message with appropriate styling
                    with st.chat_message(role, avatar=avatar):
                        if role in ("user", "assistant"):
                            # Display text content
                            st.markdown(message.get("text", ""))
                            
                            # Handle search results if present
                            if role == 'assistant' and message.get('searchResults') and len(message.get('searchResults', [])) > 0:
                                ui.display_search_results(message['searchResults'])
                            
                            # Handle SQL results with visualization
                            if role == 'assistant' and message.get('sql') and message.get('sql_df') is not None:
                                if not message.get('sql_df', pd.DataFrame()).empty:
                                    st.markdown("""
                                        <div style="margin: 15px 0 10px 0; font-size: 1rem; color: #334155; font-weight: 600;">
                                            Query Results
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    ui.display_sql_visualization(message, message['sql_df'])
                            
                            # Handle suggestions
                            if role == 'assistant' and message.get('suggestions') and len(message.get('suggestions', [])) > 0:
                                suggestion = ui.display_suggestions(message['suggestions'], message_index=st.session_state.messages.index(message))
                                if suggestion:
                                    st.session_state.active_suggestion = suggestion
                                    st.rerun()
                        
                        elif role == "❗" and message.get('type') == 'hint':
                            st.warning(message.get('text', ""))
        
        # Check for active suggestion
        if st.session_state.active_suggestion:
            # Create user message
            user_message = {"role": "user", "text": st.session_state.active_suggestion}
            st.session_state.messages.append(user_message)
            
            # Add to formatted messages
            formatted_msg = Message("user", st.session_state.active_suggestion)
            st.session_state.formatted_messages.append(formatted_msg)
            
            # Clear suggestion
            suggestion = st.session_state.active_suggestion
            st.session_state.active_suggestion = None
            
            # Process message
            start_time = time.time()
            chat_service.process_message(suggestion)
            end_time = time.time()
            
            # Track performance
            st.session_state.response_times.append(end_time - start_time)
            
            st.rerun()
    
    # Chat input container
    with stylable_container(
        key='chat_input',
        css_styles="""{
            border-radius: 14px; 
            padding: 18px 22px; 
            background-color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.06);
            margin-bottom: 10px;
        }"""
    ):
        c1, c2 = st.columns([1000, 1])  # Use small dummy second column
        
        with c1:
            prompt_placeholder = "What would you like to know about your data?"
            if len(api_service.get_tool_resources()) == 0:
                prompt_placeholder = "Configure services in the sidebar first, then ask a question..."
            
            if prompt := st.chat_input(prompt_placeholder):
                # Create user message
                user_message = {"role": "user", "text": prompt}
                st.session_state.messages.append(user_message)
                
                # Add to formatted messages
                formatted_msg = Message("user", prompt)
                st.session_state.formatted_messages.append(formatted_msg)
                
                # Check if services are configured
                if len(api_service.get_tool_resources()) == 0:
                    st.session_state.messages.append({
                        "role": "❗", 
                        "type": "hint", 
                        "text": 'You are using Cortex Agent without any active services or tools. Configure them in the sidebar first.'
                    })
                    st.rerun()
                
                # Process the prompt
                try:
                    with st.spinner('Processing your request...'):
                        start_time = time.time()
                        chat_service.process_message(prompt)
                        end_time = time.time()
                        
                        # Track performance
                        st.session_state.response_times.append(end_time - start_time)
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")
    
    # Footer with usage tips
    with st.expander("ℹ️ Usage Tips", expanded=False):
        st.markdown("""
        ### 💡 Tips for best results
        
        - **Be specific** in your questions for more accurate answers
        - Enable **Cortex Search** services to query documentation
        - Add **Semantic Models** to analyze your data with natural language
        - Try asking for **visualizations** of your data
        - The agent maintains context across multiple messages
        """)


if __name__ == "__main__":
    main()
