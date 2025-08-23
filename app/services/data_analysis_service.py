import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from app.models import ContextSource
import io
import os

logger = logging.getLogger(__name__)


@tool
def analyze_dataframe(query: str) -> str:
    """
    Analyze CSV/Excel data using pandas based on natural language query.
    
    Args:
        query: Natural language question about the data
        
    Returns:
        Analysis results as formatted string
    """
    try:
        # For now, simulate data analysis since we don't have real file loading
        # In production, this would load actual CSV/Excel files
        
        # Return simulated analysis result
        return f"""
Data Analysis Results for query: "{query}"

📊 Simulated Analysis:
- Query processed successfully
- Found sample data patterns
- Statistical summary: 
  * Mean: 150.5
  * Count: 100 rows
  * Categories: Sales, Marketing, Development

📈 Key Insights:
- Sales trend shows 15% growth
- Top performing category: Sales (45%)
- Recommended action: Focus on high-performing segments

Note: This is a simulated response. In production, this tool would:
1. Load actual CSV/Excel files from document context
2. Create pandas DataFrame
3. Execute data analysis based on the query
4. Return real statistical results
"""
        
    except Exception as e:
        logger.error(f"Error in analyze_dataframe: {e}")
        return f"Error analyzing data: {str(e)}"


@tool  
def generate_chart(data_description: str, chart_type: str = "bar", chart_config: str = "{}") -> str:
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
        
        # Generate sample chart based on type
        # In production, this would use actual data from analyze_dataframe
        
        if chart_type.lower() == "bar":
            fig = px.bar(
                x=config.get("x_data", ["Category A", "Category B", "Category C"]),
                y=config.get("y_data", [10, 20, 15]),
                title=config.get("title", "Bar Chart"),
                labels=config.get("labels", {"x": "Categories", "y": "Values"})
            )
        
        elif chart_type.lower() == "line":
            fig = px.line(
                x=config.get("x_data", [1, 2, 3, 4, 5]),
                y=config.get("y_data", [10, 15, 12, 18, 20]),
                title=config.get("title", "Line Chart"),
                labels=config.get("labels", {"x": "Time", "y": "Values"})
            )
            
        elif chart_type.lower() == "scatter":
            fig = px.scatter(
                x=config.get("x_data", [1, 2, 3, 4, 5]),
                y=config.get("y_data", [10, 15, 12, 18, 20]),
                title=config.get("title", "Scatter Plot"),
                labels=config.get("labels", {"x": "X Values", "y": "Y Values"})
            )
            
        elif chart_type.lower() == "pie":
            fig = px.pie(
                values=config.get("values", [30, 25, 45]),
                names=config.get("names", ["A", "B", "C"]),
                title=config.get("title", "Pie Chart")
            )
            
        else:
            # Default to bar chart
            fig = px.bar(
                x=["Sample A", "Sample B", "Sample C"],
                y=[10, 20, 15],
                title="Sample Chart"
            )
        
        # Convert to JSON
        chart_json = fig.to_json()
        
        # Wrap in response format expected by client
        response = {
            "chart_data": json.loads(chart_json),
            "chart_type": chart_type,
            "description": data_description,
            "config": config
        }
        
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"Error in generate_chart: {e}")
        return json.dumps({
            "error": f"Error generating chart: {str(e)}",
            "chart_data": None
        })


class DataAnalysisService:
    """Service for managing data analysis operations"""
    
    @staticmethod
    def get_analysis_tools() -> List:
        """Get list of available analysis tools for the React Agent"""
        return [analyze_dataframe, generate_chart]
    
    @staticmethod
    def load_dataframe_from_file(file_path: str, file_type: str) -> pd.DataFrame:
        """
        Load pandas dataframe from file
        
        Args:
            file_path: Path to the file
            file_type: Type of file (csv, xlsx, xls)
            
        Returns:
            pandas DataFrame
        """
        try:
            if file_type.lower() == "csv":
                return pd.read_csv(file_path)
            elif file_type.lower() in ["xlsx", "xls"]:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error loading dataframe from {file_path}: {e}")
            raise
    
    @staticmethod
    def format_dataframe_info(df: pd.DataFrame) -> str:
        """
        Format dataframe information for context
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Formatted string with dataframe info
        """
        try:
            # Get basic info
            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            info_str = info_buffer.getvalue()
            
            # Get sample data
            sample_data = df.head().to_string()
            
            # Get basic statistics
            stats = df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else "No numeric columns for statistics"
            
            formatted_info = f"""
DataFrame Information:
{info_str}

Sample Data (first 5 rows):
{sample_data}

Statistical Summary:
{stats}
"""
            return formatted_info
            
        except Exception as e:
            logger.error(f"Error formatting dataframe info: {e}")
            return f"Error getting dataframe info: {str(e)}"