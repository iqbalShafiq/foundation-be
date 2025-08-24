import pandas as pd
import plotly.express as px
import json
import logging
from typing import List, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from app.services.react_agent_context import get_current_context
from app.database import SessionLocal
from app.models import Document, DocumentType
import os

logger = logging.getLogger(__name__)


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
            base_url="https://openrouter.ai/api/v1",
            model="gpt-4.1-mini",
            temperature=0,
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
    x_data: List,
    y_data: List,
    chart_type: str = "bar",
    title: str = "Chart",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    names: Optional[List] = None,
    values: Optional[List] = None,
) -> str:
    """
    Generate interactive Plotly chart with specified data and configuration.

    Args:
        x_data: List of x-axis data points (categories, labels, or numeric values)
        y_data: List of y-axis data points (numeric values)
        chart_type: Type of chart - "bar", "line", "scatter", "pie", "histogram", "box"
        title: Chart title
        x_label: Label for x-axis
        y_label: Label for y-axis
        names: List of names for pie chart segments (optional, defaults to x_data)
        values: List of values for pie chart (optional, defaults to y_data)

    Returns:
        JSON string containing Plotly chart configuration
    """
    try:
        labels = {"x": x_label, "y": y_label}

        if chart_type.lower() == "bar":
            fig = px.bar(
                x=x_data,
                y=y_data,
                title=title,
                labels=labels,
            )

        elif chart_type.lower() == "line":
            fig = px.line(
                x=x_data,
                y=y_data,
                title=title,
                labels=labels,
            )

        elif chart_type.lower() == "scatter":
            fig = px.scatter(
                x=x_data,
                y=y_data,
                title=title,
                labels=labels,
            )

        elif chart_type.lower() == "pie":
            pie_names = names if names is not None else x_data
            pie_values = values if values is not None else y_data
            fig = px.pie(
                values=pie_values,
                names=pie_names,
                title=title,
            )

        elif chart_type.lower() == "histogram":
            fig = px.histogram(
                x=x_data,
                title=title,
                labels={"x": x_label, "count": "Frequency"},
            )

        elif chart_type.lower() == "box":
            fig = px.box(
                y=y_data,
                title=title,
                labels={"y": y_label},
            )

        else:
            fig = px.bar(
                x=x_data,
                y=y_data,
                title=title,
                labels=labels,
            )

        chart_json = fig.to_json()

        response = {
            "chart_data": json.loads(chart_json) if chart_json else None,
            "chart_type": chart_type,
            "title": title,
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
