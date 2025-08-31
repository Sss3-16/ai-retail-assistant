from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableBranch
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel,Field
import streamlit as st
from typing import Literal
import time
import uuid

from dotenv import load_dotenv
import os
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium ChatGPT-like CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    header[data-testid="stHeader"] {display:none;}
    .stToolbar {display:none;}
    
    /* Global font and base styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main app container */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    /* Main content area */
    .main {
        padding: 0;
        max-width: none !important;
        background: transparent;
    }
    
    /* Hide default chat styling */
    [data-testid="stChatMessage"] {
        display: none !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
        height: calc(100vh - 140px);
        overflow-y: auto;
        scroll-behavior: smooth;
    }
    
    /* Message styling */
    .message {
        margin: 20px 0;
        display: flex;
        align-items: flex-start;
        animation: fadeInUp 0.3s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* User message - right aligned */
    .user-message {
        justify-content: flex-end;
        margin-left: 20%;
    }
    
    .user-message .message-content {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        box-shadow: 0 4px 20px rgba(0, 123, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        word-wrap: break-word;
        line-height: 1.5;
        font-weight: 400;
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 12px;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Assistant message - left aligned */
    .assistant-message {
        justify-content: flex-start;
        margin-right: 20%;
    }
    
    .assistant-message .message-content {
        background: rgba(40, 44, 52, 0.95);
        color: #e8eaed;
        padding: 16px 20px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        word-wrap: break-word;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .assistant-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .premium-header {
        background: rgba(15, 15, 35, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px 0;
        text-align: center;
        margin-bottom: 20px;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .premium-header h1 {
        background: linear-gradient(135deg, #10a37f 0%, #007bff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .premium-header p {
        color: rgba(232, 234, 237, 0.7);
        font-size: 1.1rem;
        margin: 8px 0 0 0;
        font-weight: 400;
    }
    
    /* Compact header */
    .compact-header {
        background: rgba(15, 15, 35, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 0;
        text-align: center;
        margin-bottom: 10px;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .compact-header h2 {
        background: linear-gradient(135deg, #10a37f 0%, #007bff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        background: transparent;
        border: none;
        padding: 20px;
        position: sticky;
        bottom: 0;
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stChatInput > div {
        background: rgba(40, 44, 52, 0.95) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 25px !important;
        padding: 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stChatInput > div:hover {
        border-color: rgba(16, 163, 127, 0.5) !important;
        box-shadow: 0 8px 40px rgba(16, 163, 127, 0.2) !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.2), 0 8px 40px rgba(16, 163, 127, 0.3) !important;
    }
    
    .stChatInput input {
        background: transparent !important;
        border: none !important;
        color: #e8eaed !important;
        font-size: 16px !important;
        padding: 16px 24px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400;
    }
    
    .stChatInput input::placeholder {
        color: rgba(232, 234, 237, 0.5) !important;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-content {
        background: rgba(40, 44, 52, 0.8);
        backdrop-filter: blur(15px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .sidebar-content h2, .sidebar-content h3 {
        color: #e8eaed;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 14px !important;
        box-shadow: 0 4px 16px rgba(16, 163, 127, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d8f6b 0%, #157055 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(16, 163, 127, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Metric styling */
    .metric-card {
        background: rgba(40, 44, 52, 0.8);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10a37f;
        margin-bottom: 4px;
    }
    
    .metric-label {
        color: rgba(232, 234, 237, 0.7);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Info styling */
    .premium-info {
        background: rgba(0, 123, 255, 0.1);
        border: 1px solid rgba(0, 123, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        color: #e8eaed;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 123, 255, 0.1);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(40, 44, 52, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(16, 163, 127, 0.6);
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(16, 163, 127, 0.8);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        color: rgba(232, 234, 237, 0.7);
        font-style: italic;
        margin: 10px 0;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10a37f;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% {
            opacity: 0.3;
            transform: scale(0.8);
        }
        40% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Welcome screen */
    .welcome-screen {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        padding: 40px;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 24px;
        background: linear-gradient(135deg, #10a37f 0%, #007bff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e8eaed;
        margin-bottom: 16px;
        letter-spacing: -0.02em;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        color: rgba(232, 234, 237, 0.7);
        margin-bottom: 32px;
        max-width: 600px;
        line-height: 1.6;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        max-width: 800px;
        margin-top: 32px;
    }
    
    .feature-card {
        background: rgba(40, 44, 52, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(16, 163, 127, 0.2);
        border-color: rgba(16, 163, 127, 0.3);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e8eaed;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: rgba(232, 234, 237, 0.7);
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize your existing components
@st.cache_resource
def initialize_components():
    loader = JSONLoader(
        file_path="products_100.json",
        jq_schema=".[]",
        text_content=False
    )
    
    docs = loader.load()
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    multiquery_retriever=MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(
            search_kwargs={"k":5}
        ),
        llm=model
    )

    return vector_store, model, multiquery_retriever
    

# Initialize components
vector_store, model,multiquery_retriever = initialize_components()

# Your existing classes and functions
class Classifiy(BaseModel):
    intent : Literal['location','general']=Field(description="classify into the location or general")

parser_classifier = PydanticOutputParser(pydantic_object=Classifiy)

def dynamic_k_retrieval(query):
    recommendation_keywords = ['recommend', 'suggest', 'best', 'top', 'all', 'any', 'mobile', 'phone', 'smartphone']
    
    if any(keyword in query.lower() for keyword in recommendation_keywords):
        k_value = 10
    else:
        k_value = 2
    
    result = multiquery_retriever.invoke(query)
    
    matching_store = []
    for res in result:
        matching_store.append(res.page_content)
    return "\n".join(matching_store)

# Your existing prompts
intent_prompt = PromptTemplate(
    template=(
        "Classify the user query:\n\n"
        "Query: {query}\n\n"
        "Output exactly one word: 'location' if the user is asking where a product is, "
        "otherwise 'general' format as {format_type}."
    ),
    input_variables=["query"],
    partial_variables={"format_type":parser_classifier.get_format_instructions()}
)

location_prompt = PromptTemplate(
    template=(
        "User query: {query}\n"
        "Relevant product info: {document}\n\n"
        "If product found, ONLY output in this format:\n"
        "Location: <tray>, Row: <row>, Col: <col>\n\n"
        "If not found, say exactly: sorry product not found. then recommend similar product with name or type"
    ),
    input_variables=["query","document"]
)

general_prompt = PromptTemplate(
    template=(
        "User query: {query}\n"
        "Relevant product info: {document}\n\n"
        "Answer naturally and suggest products. Do not output tray/row/col.\n"
        "if the user asks for some recommendation about some product then show some product about that category or similar products\n"
        "if the question is not about the retail query specific then return sorry i don't have information about this topic\n"
        ""
    ),
    input_variables=["query","document"]
)

parser = StrOutputParser()
runnable_tok_k = RunnableLambda(dynamic_k_retrieval)
classifier_chain = intent_prompt | model | parser_classifier

def format_for_prompt(x):
    return {
        "query": x["query"],
        "document": x["document"]
    }

branch_chain = RunnableBranch(
    (
        lambda x: x["intent"].intent == "location",
        RunnableLambda(format_for_prompt) | location_prompt 
    ),
    RunnableLambda(format_for_prompt) | general_prompt 
)

parallel_chain = RunnableParallel({
    'query': RunnablePassthrough(),
    'document': runnable_tok_k,
    "intent": classifier_chain
})

final_chain = parallel_chain | branch_chain | model | parser

def history_to_string(chat_history):
    """Convert list of messages into a single formatted string."""
    formatted = []
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

# Custom message display function
def display_message(role, content, message_id=None):
    """Display a message with premium styling"""
    if message_id is None:
        message_id = str(uuid.uuid4())
    
    if role == "user":
        st.markdown(f"""
        <div class="message user-message" id="msg-{message_id}">
            <div class="message-content">{content}</div>
            <div class="user-avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message assistant-message" id="msg-{message_id}">
            <div class="assistant-avatar">🤖</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="you are a helpful AI assistant")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
def main():
    # Header
    if not st.session_state.messages:
        # Welcome screen
        st.markdown("""
        <div class="premium-header">
            <h1>🤖 AI Retail Assistant Pro</h1>
            <p>Your intelligent companion for product discovery and assistance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Welcome content
        st.markdown("""
        <div class="welcome-screen">
            <div class="welcome-icon">🚀</div>
            <div class="welcome-title">Welcome to AI Retail Assistant Pro</div>
            <div class="welcome-subtitle">
                Experience the future of conversational AI with our premium assistant. 
                Ask about product locations, get recommendations, or have general conversations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Compact header for active chat
        st.markdown("""
        <div class="compact-header">
            <h2>🤖 AI Retail Assistant Pro</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            # Removed chat_history clearing since we're not using it
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Analytics")
        
        # Custom metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.messages)}</div>
                <div class="metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            conversations = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{conversations}</div>
                <div class="metric-label">Queries</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        <div class="premium-info">
            <strong>Try these examples:</strong><br>
            • "Where is Pepsi located?"<br>
            • "Recommend me some smartphones"<br>
            • "What's the best laptop under $1000?"<br>
            • "Show me all available headphones"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat messages
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            display_message(message["role"], message["content"], str(i))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here...", key="premium_chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Removed: st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Display user message immediately
        display_message("user", prompt, f"user-{len(st.session_state.messages)}")
        
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="message assistant-message">
            <div class="assistant-avatar">🤖</div>
            <div class="typing-indicator">
                AI is typing
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Get response - sending only the current prompt (query only)
            response = final_chain.invoke(prompt)
            
            # Clear typing indicator and show response
            typing_placeholder.empty()
            display_message("assistant", response, f"assistant-{len(st.session_state.messages) + 1}")
            
        except Exception as e:
            typing_placeholder.empty()
            error_message = f"⚠️ I encountered an error: {str(e)}"
            display_message("assistant", error_message, f"error-{len(st.session_state.messages) + 1}")
            response = error_message
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Removed: st.session_state.chat_history.append(AIMessage(content=response))
        
        # Auto-scroll to bottom
        st.markdown("""
        <script>
            setTimeout(function() {
                var chatContainer = document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }, 100);
        </script>
        """, unsafe_allow_html=True)
        
        st.rerun()

if __name__ == "__main__":
    main()