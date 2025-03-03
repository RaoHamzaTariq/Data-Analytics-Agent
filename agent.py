import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

# agent.invoke("how many rows are there?")


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)

def create_agent(df: pd.DataFrame):
    """
    Creates an agent with tools for data analysis.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        
    Returns:
        StateGraph: A compiled graph for processing messages
    """
    from tools import Tools
    
    # Initialize tools with the dataframe
    tools_instance = Tools()
    tools = tools_instance.load_data(df)
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools_instance.tools)
    columns = df.columns.tolist()

    def assistant(state: MessagesState) -> MessagesState:
        """Processes user messages with LLM and Pandas tools."""
        system_content = f"""
        You are BI Data Agent, an advanced AI Data Assistant powered by BI Structure and developed by Rao Hamza Tariq.
        You have three key capabilities:

        1. Data Analysis & Queries (convert_query_to_pandas):
        • Execute natural language queries on the dataset
        • Perform calculations and aggregations
        • Filter and analyze data
        • Answer questions about the data

        2. Dataset Summary (get_summary_dataset):
        • Get overview of the dataset
        • View basic dataset information
        • See statistical summaries of numeric columns

        3. Correlation Analysis (get_correlation):
        • Calculate correlation between two numeric columns
        • Interpret correlation strength and direction
        • Understand relationships between variables

        Available Dataset Columns: {", ".join(columns)}

        Guidelines:
        - For data queries: Use convert_query_to_pandas
        - For dataset overview: Use get_summary_dataset
        - For relationships: Use get_correlation
        - Be conversational and friendly while maintaining professionalism
        - Keep track of the conversation context
        - If unsure, ask for clarification

        Remember to:
        - Give concise but complete answers
        - Explain technical terms when needed
        - Maintain conversation continuity
        - Be helpful and accurate
        """
        # Get existing messages from state
        messages = state["messages"]
        
        # If no messages exist or first message is not system message, add system message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_content)] + messages
        
        # Get response from LLM with full message history
        response = llm_with_tools.invoke(messages)
        
        # Update state with all messages including the new response
        return {"messages": messages + [response]}

    # Create the graph
    graph_builder = StateGraph(state_schema=MessagesState)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.set_entry_point("assistant")
    graph_builder.add_conditional_edges("assistant", tools_condition)
    graph_builder.add_edge("tools", "assistant")

    return graph_builder.compile(checkpointer=MemorySaver())

# react_graph = create_agent()
