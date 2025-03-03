import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

class Tools:
    def __init__(self):
        self.df = None
        self.tools = []
        self.tool_names = []

    def load_data(self, df):
        """Load data and initialize tools"""
        self.df = df
        # Initialize tools with the dataframe
        self.tools = [
            self.convert_query_to_pandas,
            self.get_summary_dataset,
            self.get_correlation
        ]
        self.tool_names = [tool.__name__ for tool in self.tools]
        return self.tools

    def get_summary_dataset(self) -> str:
        """Get a summary of the dataset"""
        if self.df is None:
            return "❌ No data loaded. Please upload a dataset first."
        
        try:
            summary = f"""
            📊 Dataset Summary:
            
            Basic Information:
            • Total Rows: {len(self.df):,}
            • Total Columns: {len(self.df.columns):,}
            
            Statistical Summary:
            {self.df.describe().to_string()}
            """
            return summary
        except Exception as e:
            return f"❌ Error getting summary: {str(e)}"

    def convert_query_to_pandas(self, query: str) -> str:
        """
        Converts a natural language query into a Pandas operation using a Pandas agent.

        Args:
            query (str): The natural language query to execute.

        Returns:
            str: The output of the executed Pandas operation, or an error message.
        """
        try:
            if self.df is None:
                return "❌ No data loaded. Please upload a dataset first."
            
            agent = create_pandas_dataframe_agent(
                ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash"),
                self.df,
                verbose=True,
                allow_dangerous_code=True
            )
            result = agent.invoke(query)['output']
            return f"""
            📊 Analysis Result:
            {result}
            """
        except Exception as e:
            return f"""
            ❌ Error processing query:
            {str(e)}
            
            Please try rephrasing your question.
            """

    def get_correlation(self, column1: str, column2: str):
        """
        Calculates correlation analysis between two columns.
        
        Args:
            column1 (str): First column name
            column2 (str): Second column name
        
        Returns:
            dict: A dictionary containing the status and correlation analysis.
        """
        if self.df is None:
            return {"status": "error", "message": "❌ No data loaded. Please upload a dataset first."}
        try:
            corr = self.df[column1].corr(self.df[column2])
            interpretation = ""
            if abs(corr) < 0.3:
                interpretation = "weak"
            elif abs(corr) < 0.7:
                interpretation = "moderate"
            else:
                interpretation = "strong"
            
            direction = "positive" if corr > 0 else "negative"
            
            return {
                "status": "success",
                "message": f"""
                🔄 Correlation Analysis:
                
                Between '{column1}' and '{column2}':
                • Correlation Coefficient: {corr:.4f}
                • Interpretation: {interpretation} {direction} correlation
                """
            }
        except Exception as e:
            return {"status": "error", "message": f"❌ Error calculating correlation: {str(e)}"} 