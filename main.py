import pandas as pd
import streamlit as st
from agent import create_agent, Tools
from langchain_core.messages import HumanMessage, AIMessage
import os
from datetime import datetime
import io
from openpyxl import Workbook

# Page configuration
st.set_page_config(
    page_title="AI Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
try:
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Failed to load CSS file: {str(e)}")

# Add Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
""", unsafe_allow_html=True)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Header
st.markdown("""
<div class="header-container">
    <div class="main-title">🤖 AI Data Assistant</div>
    <div class="powered-by">Powered by BI Structure</div>
    <div class="developed-by">Developed by Rao Hamza Tariq</div>
    <div class="social-links">
        <a href="https://www.linkedin.com/in/rao-hamza-tariq" target="_blank" title="LinkedIn">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/raohamzatariq" target="_blank" title="GitHub">
            <i class="fab fa-github"></i>
        </a>
        <a href="https://www.upwork.com/freelancers/~01c73bcca9bafbd6b9" target="_blank" title="Upwork">
            <i class="fas fa-briefcase"></i>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for data management
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: var(--primary-color); font-size: 1.5rem; font-weight: 500;'>
                📊 Data Management
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"],
        help="Upload Excel (.xlsx) or CSV (.csv) file"
    )

    # Add Download Section in sidebar
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 💾 Download Options")
        
        if st.button("📥 Download Cleaned Dataset", use_container_width=True, key="sidebar_download"):
            try:
                with st.spinner("Preparing your download..."):
                    # First try to get the cleaned DataFrame from the agent
                    try:
                        get_df_result = st.session_state.agent.invoke(
                            {"messages": [HumanMessage(content="get cleaned data")]},
                            config={"configurable": {"thread_id": "download_thread"}}
                        )
                        
                        # Check if we got a DataFrame in the response
                        response_content = get_df_result["messages"][-1].content
                        if isinstance(response_content, pd.DataFrame):
                            cleaned_df = response_content
                            st.session_state.cleaned_df = cleaned_df.copy()
                        else:
                            # If not a DataFrame, try getting from session state
                            if st.session_state.cleaned_df is not None:
                                cleaned_df = st.session_state.cleaned_df.copy()
                            else:
                                # Last resort: try getting from Tools instance
                                tools_instance = Tools.get_instance()
                                if tools_instance is not None:
                                    cleaned_df = tools_instance.get_cleaned_data()
                                    if isinstance(cleaned_df, pd.DataFrame):
                                        st.session_state.cleaned_df = cleaned_df.copy()
                                    else:
                                        st.warning("No cleaning operations have been performed. Using original data.")
                                        cleaned_df = st.session_state.df
                                else:
                                    st.warning("No cleaning operations have been performed. Using original data.")
                                    cleaned_df = st.session_state.df
                    except Exception as e:
                        st.error(f"Error getting cleaned data: {str(e)}")
                        if st.session_state.cleaned_df is not None:
                            cleaned_df = st.session_state.cleaned_df.copy()
                        else:
                            cleaned_df = st.session_state.df
                    
                    # Get timestamp for filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Prepare download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📊 Download as CSV",
                            data=cleaned_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"cleaned_dataset_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="csv_download"
                        )
                    
                    with col2:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            label="📑 Download as Excel",
                            data=excel_buffer,
                            file_name=f"cleaned_dataset_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="excel_download"
                        )
                    
                    st.success("✅ Your cleaned dataset is ready for download!")
            except Exception as e:
                st.error(f"❌ Error preparing download: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")

    if uploaded_file:
        try:
            # Load dataset
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Update session state
            st.session_state.df = df
            
            # Create agent and update state
            st.session_state.agent = create_agent(df)
            st.session_state.data_loaded = True
            
            # Show success message
            st.success("✅ Dataset loaded successfully!")
            
            # Display dataset info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", f"{len(df.columns):,}")
            
            # Dataset preview
            with st.expander("Preview Dataset"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Add welcome message only if it's not already shown
            if not st.session_state.messages:
                welcome_msg = (
                    f"👋 Welcome! I've loaded your dataset successfully.\n\n"
                    f"📊 Dataset Overview:\n"
                    f"• Rows: {len(df):,}\n"
                    f"• Columns: {len(df.columns):,}\n\n"
                    "How can I help you analyze this data?"
                )
                st.session_state.messages.append(AIMessage(content=welcome_msg))
                st.session_state.conversation_history.append(AIMessage(content=welcome_msg))
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
        
        # Data Cleaning Options
        with st.expander("🧹 Data Cleaning"):
            cleaning_option = st.selectbox(
                "Select Cleaning Operation",
                ["All", "Missing Values", "Duplicates", "String Cleaning", "Outliers"]
            )
            
            if st.button("Clean Data"):
                with st.spinner("Cleaning data..."):
                    operation = cleaning_option.lower().replace(" ", "_")
                    clean_msg = f"clean the data with operation {operation}"
                    st.session_state.conversation_history.append(HumanMessage(content=clean_msg))
                    result = st.session_state.agent.invoke(
                        {"messages": st.session_state.conversation_history},
                        config={"configurable": {"thread_id": "clean_thread", "checkpoint_ns": "clean_session", "checkpoint_id": "clean_1"}}
                    )
                    # Get the cleaned DataFrame from the agent
                    get_df_result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content="get cleaned data")]},
                        config={"configurable": {"thread_id": "get_data_thread", "checkpoint_ns": "get_data_session", "checkpoint_id": "get_data_1"}}
                    )
                    # Update the cleaned DataFrame in session state - convert the content to DataFrame
                    df_content = get_df_result["messages"][-1].content
                    if isinstance(df_content, pd.DataFrame):
                        st.session_state.cleaned_df = df_content
                    else:
                        st.error("Failed to get cleaned DataFrame from agent")
                    
                    response = result["messages"][-1].content
                    st.session_state.conversation_history.append(AIMessage(content=response))
                    st.write(response)
            
            if st.button("Reset Data"):
                with st.spinner("Resetting data..."):
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content="reset data")]},
                        config={"configurable": {"thread_id": "1"}}
                    )
                    # Reset the cleaned DataFrame to None
                    st.session_state.cleaned_df = None
                    st.write(result["messages"][-1].content)
                    st.rerun()

        # Clear chat button
        if st.session_state.messages and st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()

        with st.expander("🧹 Advanced Data Cleaning"):
            if not st.session_state.data_loaded:
                st.info("Please upload a dataset first to access advanced cleaning options.")
            else:
                cleaning_tab1, cleaning_tab2 = st.tabs(["Quick Clean", "Advanced Clean"])
                
                with cleaning_tab1:
                    if st.button("Quick Clean (Remove missing, duplicates, outliers)"):
                        with st.spinner("Cleaning data..."):
                            result = st.session_state.agent.invoke(
                                {"messages": [HumanMessage(content="clean the data")]},
                                config={"configurable": {"thread_id": "quick_clean_thread", "checkpoint_ns": "quick_clean_session", "checkpoint_id": "quick_clean_1"}}
                            )
                            # Get the cleaned DataFrame after quick cleaning
                            get_df_result = st.session_state.agent.invoke(
                                {"messages": [HumanMessage(content="get cleaned data")]},
                                config={"configurable": {"thread_id": "get_quick_data_thread", "checkpoint_ns": "get_quick_data_session", "checkpoint_id": "get_quick_data_1"}}
                            )
                            # Update the cleaned DataFrame in session state
                            df_content = get_df_result["messages"][-1].content
                            if isinstance(df_content, pd.DataFrame):
                                st.session_state.cleaned_df = df_content
                            else:
                                st.error("Failed to get cleaned DataFrame from agent")
                            
                            st.write(result["messages"][-1].content)
                
                with cleaning_tab2:
                    # Missing Values
                    st.subheader("Missing Values")
                    missing_method = st.selectbox(
                        "Handle Missing Values",
                        ["drop", "mean", "median", "mode", "forward", "backward", "value"]
                    )
                    if missing_method == "value":
                        fill_value = st.text_input("Fill Value")
                    
                    # Outliers
                    st.subheader("Outliers")
                    handle_outliers = st.checkbox("Remove Outliers")
                    if handle_outliers:
                        outlier_method = st.selectbox(
                            "Outlier Method",
                            ["iqr", "zscore", "percentile"]
                        )
                        outlier_columns = st.multiselect(
                            "Select Columns for Outlier Removal",
                            st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
                        )
                    
                    # Format Standardization
                    st.subheader("Format Standardization")
                    format_cols = {}
                    for col in st.session_state.df.columns:
                        if st.checkbox(f"Standardize {col}"):
                            format_type = st.selectbox(
                                f"Format for {col}",
                                ["datetime", "float", "integer", "string"],
                                key=f"format_{col}"
                            )
                            format_cols[col] = format_type
                    
                    # Execute Cleaning
                    if st.button("Execute Advanced Cleaning"):
                        with st.spinner("Performing advanced cleaning..."):
                            operations = {
                                'missing_values': {
                                    'method': missing_method,
                                    'fill_value': fill_value if missing_method == "value" else None
                                }
                            }
                            
                            if handle_outliers:
                                operations['outliers'] = {
                                    'method': outlier_method,
                                    'columns': outlier_columns
                                }
                            
                            if format_cols:
                                operations['formats'] = format_cols
                            
                            result = st.session_state.agent.invoke(
                                {"messages": [HumanMessage(content=f"clean the data with operations {operations}")]},
                                config={"configurable": {"thread_id": "advanced_clean_thread", "checkpoint_ns": "advanced_clean_session", "checkpoint_id": "advanced_clean_1"}}
                            )
                            st.write(result["messages"][-1].content)
                            
                            # Get the cleaned DataFrame after advanced cleaning
                            get_df_result = st.session_state.agent.invoke(
                                {"messages": [HumanMessage(content="get cleaned data")]},
                                config={"configurable": {"thread_id": "get_advanced_data_thread", "checkpoint_ns": "get_advanced_data_session", "checkpoint_id": "get_advanced_data_1"}}
                            )
                            # Update the cleaned DataFrame in session state
                            df_content = get_df_result["messages"][-1].content
                            if isinstance(df_content, pd.DataFrame):
                                st.session_state.cleaned_df = df_content
                            else:
                                st.error("Failed to get cleaned DataFrame from agent")
                            
                            # Show cleaning stats
                            stats = st.session_state.agent.invoke(
                                {"messages": [HumanMessage(content="get cleaning stats")]},
                                config={"configurable": {"thread_id": "1"}}
                            )
                            st.json(stats["messages"][-1].content)

            # Reset Data Option
            if st.button("Reset to Original Data"):
                result = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content="reset data")]},
                    config={"configurable": {"thread_id": "1"}}
                )
                st.write(result["messages"][-1].content)
                st.rerun()

# Main chat interface
chat_container = st.container()

# Display messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input(
    "Ask about your data..." if st.session_state.data_loaded 
    else "Upload a dataset to start..."
):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.conversation_history.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    if not st.session_state.data_loaded:
        response = "📤 Please upload a dataset first to start the analysis."
        st.session_state.messages.append(AIMessage(content=response))
        st.session_state.conversation_history.append(AIMessage(content=response))
        with st.chat_message("assistant"):
            st.write(response)
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Analyzing..."):
                    # Check if this is a cleaning request
                    cleaning_keywords = ['clean', 'remove', 'fix', 'handle', 'process']
                    is_cleaning_request = any(keyword in prompt.lower() for keyword in cleaning_keywords)
                    
                    # Pass the entire conversation history to the agent
                    result = st.session_state.agent.invoke(
                        {"messages": st.session_state.conversation_history},
                        config={"configurable": {"thread_id": "chat_thread", "checkpoint_ns": "chat_session", "checkpoint_id": "chat_1"}}
                    )
                    response = result["messages"][-1].content
                    
                    # If this was a cleaning request, get the cleaned DataFrame
                    if is_cleaning_request:
                        try:
                            # Get the cleaned DataFrame from the Tools instance
                            tools_instance = Tools.get_instance()
                            if tools_instance:
                                cleaned_df = tools_instance.get_cleaned_data()
                                if isinstance(cleaned_df, pd.DataFrame):
                                    st.session_state.cleaned_df = cleaned_df.copy()
                                    st.success("✨ Data cleaning completed successfully!")
                                    
                                    # Show a preview of the cleaned data
                                    with st.expander("Preview Cleaned Data"):
                                        st.dataframe(cleaned_df.head(), use_container_width=True)
                                        
                                        # Show cleaning stats
                                        if tools_instance.cleaning_stats:
                                            st.markdown("### Cleaning Statistics")
                                            stats = tools_instance.cleaning_stats
                                            if 'operations_performed' in stats:
                                                for op in stats['operations_performed']:
                                                    st.write(f"- {op['operation']}: {op.get('rows_removed', op.get('nulls_filled', op.get('columns_processed', 'completed')))}")
                        except Exception as e:
                            st.error(f"Note: Cleaning operation completed but there was an error updating the preview: {str(e)}")
                    
                    st.session_state.messages.append(AIMessage(content=response))
                    st.session_state.conversation_history.append(AIMessage(content=response))
                    st.write(response)
            except Exception as e:
                error_msg = f"❌ Failed to process your request. Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                st.session_state.conversation_history.append(AIMessage(content=error_msg))

# Adjust the chat container to account for the footer
st.markdown("""
    <style>
        .chat-container {
            margin-bottom: 100px !important;
        }
        
        .stChatInputContainer {
            margin-bottom: 60px;
        }
    </style>
""", unsafe_allow_html=True)

