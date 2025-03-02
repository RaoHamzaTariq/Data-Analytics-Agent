import re
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from scipy import stats
import numpy as np

# Add required imports
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# agent.invoke("how many rows are there?")


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)

class Tools:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        return cls._instance
    
    def __init__(self, df: pd.DataFrame):
        # Initialize with either cleaned data if available or original data
        if st.session_state.cleaned_df is not None:
            self.df = st.session_state.cleaned_df.copy()
        else:
            self.df = df.copy()
        self.original_df = df.copy()  # Backup of original data
        self.cleaning_stats = {}  # Track cleaning operations
        self.tools = [
            self.convert_query_to_pandas,
            self.clean_data,
            self.get_cleaned_data,
            self.reset_data,
            self.get_summary_statistics
        ]
        Tools._instance = self  # Store the instance

    def format_table(self, data, headers=None, format_type='grid'):
        """Format data into a readable table format"""
        if tabulate:
            return tabulate(data, headers=headers, tablefmt=format_type)
        else:
            # Fallback formatting if tabulate is not available
            if isinstance(data, pd.DataFrame):
                return data.to_string()
            elif isinstance(data, dict):
                return '\n'.join([f"{k}: {v}" for k, v in data.items()])
            else:
                return str(data)

    def get_summary_statistics(self) -> str:
        """Generate comprehensive summary statistics for the dataset.

        This tool analyzes the dataset and provides detailed statistical information including:
        - Basic dataset metrics (rows, columns, memory usage)
        - Column types and data distributions
        - Descriptive statistics for numerical columns (mean, std, min/max etc.)
        - Missing value analysis
        - Categorical column summaries
        
        Returns:
            str: A formatted string containing the comprehensive statistical summary
        
        Note:
            The analysis adapts based on the data types present in the dataset,
            providing relevant statistics for both numerical and categorical data.
        """
        try:
            if self.df is None or self.df.empty:
                return "No data available for analysis."

            summary = []
            summary.append("=== Dataset Summary Statistics ===\n")

            try:
                # Basic Dataset Info
                summary.append("📊 Dataset Overview:")
                summary.append(f"• Total Rows: {len(self.df):,}")
                summary.append(f"• Total Columns: {len(self.df.columns):,}")
                
                # Memory Usage (with error handling)
                try:
                    total_memory = self.df.memory_usage(deep=True).sum()
                    memory_mb = total_memory / 1024 / 1024  # Convert to MB
                    summary.append(f"• Memory Usage: {memory_mb:.2f} MB")
                except:
                    pass  # Skip memory usage if it fails
                summary.append("")

                # Column Types Summary
                summary.append("📋 Column Types:")
                for col, dtype in self.df.dtypes.items():
                    summary.append(f"• {col}: {dtype}")
                summary.append("")

                # Basic Statistics
                summary.append("📈 Basic Statistics:")
                
                # Numerical Columns
                num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                if len(num_cols) > 0:
                    summary.append("\nNumerical Columns:")
                    for col in num_cols:
                        stats = self.df[col].describe()
                        summary.append(f"\n• {col}:")
                        summary.append(f"  - Count: {stats['count']:,.0f}")
                        summary.append(f"  - Mean: {stats['mean']:,.2f}")
                        summary.append(f"  - Std: {stats['std']:,.2f}")
                        summary.append(f"  - Min: {stats['min']:,.2f}")
                        summary.append(f"  - Max: {stats['max']:,.2f}")
                        
                        # Calculate missing values
                        missing = self.df[col].isnull().sum()
                        if missing > 0:
                            summary.append(f"  - Missing Values: {missing:,} ({(missing/len(self.df)*100):.1f}%)")

                # Categorical Columns
                cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    summary.append("\nCategorical Columns:")
                    for col in cat_cols:
                        unique_vals = self.df[col].nunique()
                        summary.append(f"\n• {col}:")
                        summary.append(f"  - Unique Values: {unique_vals:,}")
                        
                        # Show top 5 most frequent values
                        value_counts = self.df[col].value_counts().head(5)
                        if not value_counts.empty:
                            summary.append("  - Top 5 Values:")
                            for val, count in value_counts.items():
                                val_str = str(val) if pd.notna(val) else "NaN"
                                summary.append(f"    * {val_str}: {count:,} ({(count/len(self.df)*100):.1f}%)")
                        
                        # Calculate missing values
                        missing = self.df[col].isnull().sum()
                        if missing > 0:
                            summary.append(f"  - Missing Values: {missing:,} ({(missing/len(self.df)*100):.1f}%)")

                # Overall Missing Values Summary
                total_missing = self.df.isnull().sum().sum()
                if total_missing > 0:
                    summary.append("\n⚠️ Overall Missing Values:")
                    summary.append(f"• Total Missing: {total_missing:,} values")
                    summary.append(f"• Percentage: {(total_missing/(len(self.df)*len(self.df.columns))*100):.1f}% of all data points")

                # Duplicate Rows
                duplicates = self.df.duplicated().sum()
                if duplicates > 0:
                    summary.append(f"\n⚠️ Duplicate Rows: {duplicates:,} ({(duplicates/len(self.df)*100):.1f}% of data)")

                return "\n".join(summary)

            except Exception as e:
                return f"Error while generating summary statistics: {str(e)}\nPlease check if the dataset is properly loaded and accessible."

        except Exception as e:
            return f"Critical error in summary statistics: {str(e)}"

    def convert_query_to_pandas(self, query: str) -> str:
        """Convert a natural language query into a Pandas operation"""
        try:
            agent = create_pandas_dataframe_agent(
                ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash"),
                self.df,
                verbose=True,
                allow_dangerous_code=True
            )
            return agent.invoke(query)['output']
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def reset_data(self) -> str:
        """Reset data to original state"""
        try:
            self.df = self.original_df.copy()
            st.session_state.cleaned_df = None
            self.cleaning_stats = {}
            return "Data has been reset to original state"
        except Exception as e:
            return f"Error resetting data: {str(e)}"

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame"""
        try:
            if self.df is None:
                return self.original_df.copy()
            # Update session state with cleaned data
            st.session_state.cleaned_df = self.df.copy()
            return self.df.copy()
        except Exception as e:
            print(f"Error getting cleaned data: {str(e)}")
            return self.original_df.copy()

    def clean_data(self, operations: str = None, columns: list[str] = None, advanced_options: dict = None) -> str:
        """Clean the dataset based on specified operations and columns
        
        Args:
            operations (str): Cleaning operation to perform ('all', 'missing', 'duplicates', 'strings', 'outliers')
            columns (list[str]): List of column names to clean. If None, applies to all applicable columns
            advanced_options (dict): Advanced cleaning options like:
                - fill_method: {'mean', 'median', 'mode', 'value', 'forward', 'backward'}
                - fill_value: Custom value to fill missing data
                - string_operations: ['strip', 'lower', 'upper', 'title']
                - outlier_method: {'iqr', 'zscore', 'percentile'}
                - outlier_threshold: Threshold for outlier detection
                - format_type: {'datetime', 'numeric', 'string'}
        
        Returns:
            str: A summary of the cleaning operations performed
        """
        try:
            # Use the latest cleaned data if available
            if st.session_state.cleaned_df is not None:
                self.df = st.session_state.cleaned_df.copy()
            
            initial_rows = len(self.df)
            initial_nulls = self.df.isnull().sum().sum()

            # Store initial stats
            stats = {
                'initial_rows': initial_rows,
                'initial_null_values': initial_nulls,
                'operations_performed': []
            }

            # Set default advanced options if not provided
            if advanced_options is None:
                advanced_options = {}
            
            # Handle missing values
            if operations in ['all', 'missing']:
                # Get columns to process
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                categorical_cols = self.df.select_dtypes(include=['object']).columns
                
                # Filter columns if specific columns are requested
                if columns:
                    numeric_cols = [col for col in numeric_cols if col in columns]
                    categorical_cols = [col for col in categorical_cols if col in columns]
                
                fill_method = advanced_options.get('fill_method', 'auto')
                fill_value = advanced_options.get('fill_value', None)
                
                # Process numeric columns
                for col in numeric_cols:
                    if fill_method == 'mean':
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
                    elif fill_method == 'median':
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    elif fill_method == 'value' and fill_value is not None:
                        self.df[col] = self.df[col].fillna(fill_value)
                    elif fill_method == 'forward':
                        self.df[col] = self.df[col].fillna(method='ffill')
                    elif fill_method == 'backward':
                        self.df[col] = self.df[col].fillna(method='bfill')
                    else:  # auto/default
                        self.df[col] = self.df[col].fillna(self.df[col].median())

                # Process categorical columns
                for col in categorical_cols:
                    if fill_method == 'mode':
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown")
                    elif fill_method == 'value' and fill_value is not None:
                        self.df[col] = self.df[col].fillna(fill_value)
                    elif fill_method == 'forward':
                        self.df[col] = self.df[col].fillna(method='ffill')
                    elif fill_method == 'backward':
                        self.df[col] = self.df[col].fillna(method='bfill')
                    else:  # auto/default
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown")

                stats['operations_performed'].append({
                    'operation': 'handle_missing_values',
                    'columns_processed': list(numeric_cols) + list(categorical_cols),
                    'fill_method': fill_method,
                    'nulls_filled': initial_nulls - self.df.isnull().sum().sum()
                })

            # Handle outliers
            if operations in ['all', 'outliers']:
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                if columns:
                    numeric_cols = [col for col in numeric_cols if col in columns]
                
                outlier_method = advanced_options.get('outlier_method', 'iqr')
                outlier_threshold = advanced_options.get('outlier_threshold', 1.5)
                
                outliers_handled = 0
                for col in numeric_cols:
                    if outlier_method == 'iqr':
                        Q1 = self.df[col].quantile(0.25)
                        Q3 = self.df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - outlier_threshold * IQR
                        upper_bound = Q3 + outlier_threshold * IQR
                        outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                        self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                        outliers_handled += outliers
                    elif outlier_method == 'zscore':
                        z_scores = abs(stats.zscore(self.df[col].fillna(self.df[col].median())))
                        outliers = (z_scores > outlier_threshold).sum()
                        self.df.loc[z_scores > outlier_threshold, col] = self.df[col].median()
                        outliers_handled += outliers
                    elif outlier_method == 'percentile':
                        lower = np.percentile(self.df[col].dropna(), outlier_threshold)
                        upper = np.percentile(self.df[col].dropna(), 100 - outlier_threshold)
                        outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                        self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                        outliers_handled += outliers

                if outliers_handled > 0:
                    stats['operations_performed'].append({
                        'operation': 'handle_outliers',
                        'method': outlier_method,
                        'columns_processed': list(numeric_cols),
                        'outliers_handled': outliers_handled
                    })

            # Remove duplicates
            if operations in ['all', 'duplicates']:
                if not columns:
                    initial_duplicates = len(self.df)
                    self.df = self.df.drop_duplicates()
                    stats['operations_performed'].append({
                        'operation': 'remove_duplicates',
                        'rows_removed': initial_duplicates - len(self.df)
                    })
                else:
                    initial_duplicates = len(self.df)
                    self.df = self.df.drop_duplicates(subset=columns)
                    stats['operations_performed'].append({
                        'operation': 'remove_duplicates',
                        'columns': columns,
                        'rows_removed': initial_duplicates - len(self.df)
                    })

            # Clean string values
            if operations in ['all', 'strings']:
                string_cols = self.df.select_dtypes(include=['object']).columns
                if columns:
                    string_cols = [col for col in string_cols if col in columns]
                
                string_operations = advanced_options.get('string_operations', ['strip', 'lower'])
                
                for col in string_cols:
                    if 'strip' in string_operations:
                        self.df[col] = self.df[col].str.strip() if self.df[col].dtype == 'object' else self.df[col]
                    if 'lower' in string_operations:
                        self.df[col] = self.df[col].str.lower() if self.df[col].dtype == 'object' else self.df[col]
                    if 'upper' in string_operations:
                        self.df[col] = self.df[col].str.upper() if self.df[col].dtype == 'object' else self.df[col]
                    if 'title' in string_operations:
                        self.df[col] = self.df[col].str.title() if self.df[col].dtype == 'object' else self.df[col]

                stats['operations_performed'].append({
                    'operation': 'clean_strings',
                    'columns_processed': list(string_cols),
                    'operations_applied': string_operations
                })

            # Format standardization
            if 'format_type' in advanced_options and columns:
                format_type = advanced_options['format_type']
                for col in columns:
                    if format_type == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif format_type == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif format_type == 'string':
                        self.df[col] = self.df[col].astype(str)
                
                stats['operations_performed'].append({
                    'operation': 'format_standardization',
                    'columns_processed': columns,
                    'format_type': format_type
                })

            # Update session state with cleaned data
            st.session_state.cleaned_df = self.df.copy()
            
            # Final stats
            stats['final_rows'] = len(self.df)
            stats['final_null_values'] = self.df.isnull().sum().sum()
            
            # Generate summary message
            summary = "Data cleaning completed:\n"
            summary += f"- Initial rows: {initial_rows}\n"
            summary += f"- Final rows: {len(self.df)}\n"
            
            if operations in ['all', 'missing']:
                summary += f"- Null values filled: {initial_nulls - self.df.isnull().sum().sum()}\n"
            
            summary += "Operations performed:\n"
            for op in stats['operations_performed']:
                if op['operation'] == 'handle_missing_values':
                    summary += f"  • Handled missing values in columns: {', '.join(op['columns_processed'])}\n"
                    summary += f"    - Filled {op['nulls_filled']} missing values using {op['fill_method']} method\n"
                elif op['operation'] == 'handle_outliers':
                    summary += f"  • Handled outliers using {op['method']} method\n"
                    summary += f"    - Processed columns: {', '.join(op['columns_processed'])}\n"
                    summary += f"    - Handled {op['outliers_handled']} outliers\n"
                elif op['operation'] == 'remove_duplicates':
                    if 'columns' in op:
                        summary += f"  • Removed {op['rows_removed']} duplicate rows based on columns: {', '.join(op['columns'])}\n"
                    else:
                        summary += f"  • Removed {op['rows_removed']} duplicate rows\n"
                elif op['operation'] == 'clean_strings':
                    summary += f"  • Cleaned string values in columns: {', '.join(op['columns_processed'])}\n"
                    summary += f"    - Applied operations: {', '.join(op['operations_applied'])}\n"
                elif op['operation'] == 'format_standardization':
                    summary += f"  • Standardized format to {op['format_type']} for columns: {', '.join(op['columns_processed'])}\n"
            
            self.cleaning_stats = stats
            return summary

        except Exception as e:
            return f"Error during data cleaning: {str(e)}"

def create_agent(df: pd.DataFrame):
    tools = Tools(df).tools
    llm_with_tools = llm.bind_tools(tools)
    columns = df.columns.tolist()

    def assistant(state: MessagesState) -> MessagesState:
        """Processes user messages with LLM and Pandas tools."""
        system_content = """
        You are BI Data Agent, an advanced AI Data Assistant powered by BI Structure and developed by Rao Hamza Tariq. Your expertise lies in data analysis, cleaning, and providing insightful answers about datasets.

        Core Capabilities:
        1. Data Analysis & Insights
           - Answer detailed questions about the data
           - Provide statistical summaries and trends
           - Generate meaningful insights and patterns
        
        2. Comprehensive Data Cleaning
           - Handle missing values intelligently (use operation='missing')
           - Remove duplicate entries (use operation='duplicates')
           - Clean and standardize string data (use operation='strings')
           - Perform all cleaning operations together (use operation='all')
           - Support column-specific cleaning by passing column names
        
        3. Data Management
           - Reset dataset to original state when needed
           - Provide access to cleaned dataset
           - Track cleaning operations and their impact
        
        Operational Guidelines:
        - For data cleaning requests:
          * Use clean_data tool with operations=['all', 'missing', 'duplicates', 'strings']
          * For column-specific cleaning, pass the columns parameter
          * Examples:
            - clean_data(operations='missing', columns=['column1', 'column2'])
            - clean_data(operations='duplicates')
            - clean_data(operations='all')
        - For accessing cleaned data: Use get_cleaned_data tool
        - For data reset requests: Use reset_data tool
        - For analysis queries: Use convert_query_to_pandas tool
        
        Communication Style:
        - Provide clear, concise responses
        - Include relevant statistics when appropriate
        - Explain any data transformations performed
        - Offer recommendations for data improvement
        
        Available Dataset Columns: """ + ", ".join(columns)
        
        messages = [SystemMessage(content=system_content)] + state["messages"]

        # Check if this is a request for cleaned data
        if any(msg.content.lower().strip() == "get cleaned data" for msg in messages if isinstance(msg, HumanMessage)):
            tools_instance = Tools.get_instance()
            if tools_instance:
                cleaned_df = tools_instance.get_cleaned_data()
                if cleaned_df is not None:
                    st.session_state.cleaned_df = cleaned_df.copy()
                    return {"messages": [AIMessage(content=cleaned_df)]}
            return {"messages": [AIMessage(content="Failed to get cleaned data.")]}

        response = llm_with_tools.invoke(messages)
        
        # Handle different types of responses
        if isinstance(response, pd.DataFrame):
            st.session_state.cleaned_df = response.copy()
            return {"messages": [AIMessage(content=response)]}
        elif isinstance(response, dict):
            if "output" in response:
                if isinstance(response["output"], pd.DataFrame):
                    st.session_state.cleaned_df = response["output"].copy()
                    return {"messages": [AIMessage(content=response["output"])]}
                return {"messages": [AIMessage(content=str(response["output"]))]}
            elif "messages" in response:
                return {"messages": response["messages"]}
        elif isinstance(response, str):
            return {"messages": [AIMessage(content=response)]}
        
        return {"messages": [response]}

    graph_builder = StateGraph(state_schema=MessagesState)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.set_entry_point("assistant")
    graph_builder.add_conditional_edges("assistant", tools_condition)
    graph_builder.add_edge("tools", "assistant")

    return graph_builder.compile(checkpointer=MemorySaver())

# react_graph = create_agent()
