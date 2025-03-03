import pandas as pd
import streamlit as st
from agent import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

# Page configuration
st.set_page_config(
    page_title="AI Data Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load external CSS
try:
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Failed to load CSS file: {str(e)}")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
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
</div>
""", unsafe_allow_html=True)

# Sidebar for data upload
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: var(--primary-color); font-size: 1.5rem; font-weight: 500;'>
                📊 Data Upload
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"],
        help="Upload Excel (.xlsx) or CSV (.csv) file"
    )

    if uploaded_file:
        try:
            # Load dataset
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Update session state
            st.session_state.df = df
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
            
            # Add welcome message
            if not st.session_state.messages:
                welcome_msg = (
                    f"👋 Welcome! I've loaded your dataset successfully.\n\n"
                    f"📊 Dataset Overview:\n"
                    f"• Rows: {len(df):,}\n"
                    f"• Columns: {len(df.columns):,}\n\n"
                    "How can I help you analyze this data?"
                )
                welcome_message = AIMessage(content=welcome_msg)
                st.session_state.messages.append(welcome_message)
                st.session_state.conversation_history = [welcome_message]

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

        # Clear chat button
        if st.session_state.messages and st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
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
    # Create user message
    user_message = HumanMessage(content=prompt)
    
    # Add to display messages
    st.session_state.messages.append(user_message)
    
    # Add to conversation history
    st.session_state.conversation_history.append(user_message)
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    if not st.session_state.data_loaded:
        response = "📤 Please upload a dataset first to start the analysis."
        ai_message = AIMessage(content=response)
        st.session_state.messages.append(ai_message)
        st.session_state.conversation_history.append(ai_message)
        with st.chat_message("assistant"):
            st.write(response)
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Analyzing..."):
                    # Pass the entire conversation history to maintain context
                    result = st.session_state.agent.invoke(
                        {"messages": st.session_state.conversation_history},
                        config={"configurable": {"thread_id": "chat_thread"}}
                    )
                    
                    # Get the response and create AI message
                    response = result["messages"][-1].content
                    ai_message = AIMessage(content=response)
                    
                    # Add to both display messages and conversation history
                    st.session_state.messages.append(ai_message)
                    st.session_state.conversation_history.append(ai_message)
                    
                    st.write(response)
            except Exception as e:
                error_msg = f"❌ Failed to process your request. Error: {str(e)}"
                error_message = AIMessage(content=error_msg)
                st.session_state.messages.append(error_message)
                st.session_state.conversation_history.append(error_message)
                st.error(error_msg)

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

