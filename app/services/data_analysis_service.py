import pandas as pd
import plotly.express as px
import json
import logging
from typing import List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from app.services.react_agent_context import get_current_context
from app.database import SessionLocal
from app.models import Document, DocumentType
import os

logger = logging.getLogger(__name__)


def _extract_data_from_description(description: str) -> tuple:
    """
    Extract x_data and y_data from data description text.

    Args:
        description: Text description containing data like "produk Laptop (10), Mouse (50), Keyboard (25)"

    Returns:
        tuple: (x_data, y_data) lists
    """
    import re

    try:
        # Pattern to match "name (number)" or "name: number"
        pattern = r"(\w+)\s*[\(:]\s*(\d+)\s*[)\)]?"
        matches = re.findall(pattern, description)

        if matches:
            x_data = [match[0] for match in matches]
            y_data = [int(match[1]) for match in matches]
            return x_data, y_data

        # Alternative pattern for "name = number" or "name - number"
        pattern2 = r"(\w+)\s*[=-]\s*(\d+)"
        matches2 = re.findall(pattern2, description)

        if matches2:
            x_data = [match[0] for match in matches2]
            y_data = [int(match[1]) for match in matches2]
            return x_data, y_data

    except Exception as e:
        logger.error(f"Error extracting data from description: {e}")

    return None, None


@tool
def analyze_dataframe(query: str) -> str:
    """
    Analyze CSV/Excel data using pandas based on natural language query.
    Uses LangChain's pandas dataframe agent for real data analysis.

    Args:
        query: Natural language question about the data

    Returns:
        Analysis results as formatted string
    """
    try:
        # Get document context from ReactAgent runtime context
        context = get_current_context()

        if not context:
            return "Error: No document context available. Please ensure documents are selected in the chat."

        user_id = context.get("user_id")
        selected_document_ids = context.get("selected_document_ids", [])

        if not user_id:
            return "Error: No user context available."

        db = SessionLocal()
        try:
            # Filter by selected document IDs if provided, otherwise get recent files
            query_filter = [
                Document.user_id == user_id,
                Document.file_type.in_(
                    [DocumentType.CSV.value, DocumentType.XLSX.value]
                ),
                Document.processing_status == "completed",
            ]

            if selected_document_ids:
                query_filter.append(Document.id.in_(selected_document_ids))
                docs = db.query(Document).filter(*query_filter).all()
                logger.info(
                    f"Filtering by selected document IDs: {selected_document_ids}"
                )
            else:
                docs = (
                    db.query(Document)
                    .filter(*query_filter)
                    .order_by(Document.created_at.desc())
                    .limit(3)
                    .all()
                )
                logger.info(
                    "No specific documents selected, using recent CSV/Excel files"
                )

            # Load first available file
            df = None
            data_source = "no data"

            for doc in docs:
                try:
                    if os.path.exists(str(doc.file_path)):
                        if str(doc.file_type) == DocumentType.CSV.value:
                            df = pd.read_csv(str(doc.file_path))
                        else:  # Excel file
                            df = pd.read_excel(str(doc.file_path))
                        data_source = f"file: {doc.original_filename}"
                        logger.info(
                            f"Loaded DataFrame from {doc.original_filename} with shape: {df.shape}"
                        )
                        break
                except Exception as load_error:
                    logger.warning(
                        f"Failed to load {doc.original_filename}: {load_error}"
                    )
                    continue

        finally:
            db.close()

        if df is None or df.empty:
            return "No CSV/Excel files found. Please upload and select CSV or Excel files in the document context."

        # Use LangChain pandas agent for analysis
        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.1,
        )

        agent_executor = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="tool-calling",
            max_iterations=5,
            early_stopping_method="generate",
        )

        # Execute query
        df_info = f"""
        DataFrame Info:
        - Data Source: {data_source}
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        
        Sample Data:
        {df.head().to_string()}
        
        User Query: {query}
        """

        result = agent_executor.invoke({"input": df_info})

        analysis_result = f"""
        📊 Data Analysis Results for: "{query}"
        
        Dataset: {data_source}
        Rows: {len(df)} | Columns: {len(df.columns)}
        
        {result.get("output", "No output generated")}
        """

        logger.info(f"Successfully analyzed dataframe with query: {query}")
        return analysis_result

    except Exception as e:
        logger.error(f"Error in analyze_dataframe: {e}")
        return f"Error analyzing data: {str(e)}"


@tool
def generate_chart(
    data_description: str, chart_type: str = "bar", chart_config: str = "{}"
) -> str:
    """
    Generate interactive Plotly chart from data analysis results.

    Args:
        data_description: Description of the data to visualize
        chart_type: Type of chart (bar, line, scatter, histogram, pie, box). Defaults to "bar"
        chart_config: JSON string with chart configuration. Defaults to "{}"

    Returns:
        JSON string containing Plotly chart configuration
    """
    try:
        # Parse chart configuration
        config = json.loads(chart_config) if chart_config else {}

        # Extract actual data from data_description
        x_data, y_data = _extract_data_from_description(data_description)

        # Use extracted data or fallback to config data
        chart_x_data = (
            config.get("x_data", x_data)
            if x_data
            else config.get("x_data", ["Category A", "Category B", "Category C"])
        )
        chart_y_data = (
            config.get("y_data", y_data)
            if y_data
            else config.get("y_data", [10, 20, 15])
        )

        if chart_type.lower() == "bar":
            # Build labels from config xaxis/yaxis or use defaults
            labels = config.get("labels", {})
            if "xaxis" in config and "title" in config["xaxis"]:
                labels["x"] = config["xaxis"]["title"]
            if "yaxis" in config and "title" in config["yaxis"]:
                labels["y"] = config["yaxis"]["title"]
            if not labels:
                labels = {"x": "Categories", "y": "Values"}

            fig = px.bar(
                x=chart_x_data,
                y=chart_y_data,
                title=config.get("title", "Bar Chart"),
                labels=labels,
            )

        elif chart_type.lower() == "line":
            line_x_data = (
                chart_x_data
                if (chart_x_data and isinstance(chart_x_data[0], (int, float)))
                else list(range(len(chart_x_data) if chart_x_data else 3))
            )
            fig = px.line(
                x=config.get("x_data", line_x_data),
                y=config.get("y_data", chart_y_data),
                title=config.get("title", "Line Chart"),
                labels=config.get("labels", {"x": "Time", "y": "Values"}),
            )

        elif chart_type.lower() == "scatter":
            scatter_x_data = (
                chart_x_data
                if (chart_x_data and isinstance(chart_x_data[0], (int, float)))
                else list(range(len(chart_x_data) if chart_x_data else 3))
            )
            fig = px.scatter(
                x=config.get("x_data", scatter_x_data),
                y=config.get("y_data", chart_y_data),
                title=config.get("title", "Scatter Plot"),
                labels=config.get("labels", {"x": "X Values", "y": "Y Values"}),
            )

        elif chart_type.lower() == "pie":
            fig = px.pie(
                values=config.get("values", chart_y_data),
                names=config.get("names", chart_x_data),
                title=config.get("title", "Pie Chart"),
            )

        else:
            # Default to bar chart with extracted data
            fig = px.bar(
                x=chart_x_data, y=chart_y_data, title=config.get("title", "Chart")
            )

        # Convert to JSON
        chart_json = fig.to_json()

        # Wrap in response format expected by client
        response = {
            "chart_data": json.loads(chart_json) if chart_json else None,
            "chart_type": chart_type,
            "description": data_description,
            "config": config,
        }

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error in generate_chart: {e}")
        return json.dumps(
            {"error": f"Error generating chart: {str(e)}", "chart_data": None}
        )


class DataAnalysisService:
    """Service for managing data analysis operations"""

    @staticmethod
    def get_analysis_tools() -> List:
        """Get list of available analysis tools for the React Agent"""
        return [analyze_dataframe, generate_chart]
