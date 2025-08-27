import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import random
from pathlib import Path
from datetime import datetime

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HCIL IT Helpdesk Chat-Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Configuration for Pre-loaded Knowledge Base ---
KNOWLEDGE_BASE_PATH = 'dataset.xlsx'

st.markdown("""
<style>
/* Ensure the very root HTML and body are black */
html, body {
    background-color: #000000 !important; /* Strict background color */
    color: white !important;
}

/* Target Streamlit's main content area */
.stApp {
    background-color: #000000 !important; /* Strict background color for the app container */
}

/* Your existing .main style, updated to the desired background */
.main {
    background: #000000 !important; /* Use your desired dark background here */
    border-radius: 20px !important;
    padding: 3.5rem !important;
    max-width: 640px !important;
    margin: 2.5rem auto;
}
.stSidebar > div:first-child {
    background-color: #1F1F1F !important;
    border-right: 2px solid white;
}

/* --- BUTTON STYLING FIXES --- */

/* Style only the Start Chat button */
.start-chat-btn {
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%) !important;
    color: white !important;
    border-radius: 25px !important;
    padding: 1.5rem 3rem !important;
    font-size: 1.4rem !important;
    border: 3px solid #fff !important;
    font-weight: bold !important;
    transition: transform 0.2s !important;
    min-width: 200px !important;
    margin: auto auto !important;
    display: block !important;
}
#stBaseButton-secondary{
margin : auto !important;
}


.start-chat-btn:hover {
    transform: scale(1.08) !important;
    color: white !important;
}


/* 2. QUICK REPLY & FEEDBACK BUTTONS: New rule to apply your desired style */
.quick-reply-buttons .stButton > button,
.feedback-buttons .stButton > button {
    display: inline-block !important;
    background: #fff !important;
    color: #e53935 !important;
    border-radius: 18px !important;
    padding: 0.5rem 1.1rem !important;
    margin: 0.15rem !important;
    cursor: pointer !important;
    font-size: 0.98rem !important;
    border: 1.5px solid #e53935 !important;
    font-weight: 500 !important;
    transition: background 0.2s, color 0.2s !important;
    width: auto !important;
}

.quick-reply-buttons .stButton > button:hover,
.feedback-buttons .stButton > button:hover {
    background: #e53935 !important;
    color: #fff !important;
}

/* --- END OF BUTTON STYLING FIXES --- */


.chat-bubble {
    padding: 1rem 1.5rem;
    border-radius: 20px;
    margin-bottom: 14px;
    max-width: 75%;
    animation: fadeInUp 0.3s;
    position: relative;
    word-break: break-word;
    font-size: 1.08rem;
    display: flex;
    align-items: center;
}
.user-bubble {
    background: #fff;
    color: #111;
    align-self: flex-end;
    margin-left: auto;
    margin-right: 0;
    border: 1.5px solid #e53935;
}
.bot-bubble {
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    align-self: flex-start;
    margin-right: auto;
    margin-left: 0;
    border: 1.5px solid #fff;
}
.avatar {
    width: 38px; height: 38px; border-radius: 75%; margin: 0 10px;
    background: #3d3d3d;
    box-shadow: 0 2px 8px rgba(229,57,53,0.12);
    font-size: 1.7rem;
    text-align: center;
    line-height: 38px;
    border: 2px solid #ff0000;
    display: flex; align-items: center; justify-content: center;
}
.user-row {
    display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-end;
}
.bot-row {
    display: flex; flex-direction: row; align-items: flex-end; justify-content: flex-start;
}
.input-bar {
    background: transparent !important;
    border-radius: 20px;
    box-shadow: 0 2px 8px rgba(229,57,53,0.12);
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    padding: 0.3rem 0.8rem;
}
.input-bar input {
    background: transparent !important;
    border: 3px solid #ffffff !important;
    color: #fff;
    width: 100%;
    padding: 0.7rem 0.8rem;
    outline: none;
    font-size: 1rem;
}
.send-btn {
    background: linear-gradient(90deg, #e53935 0%, #b71c1c 100%);
    color: #fff;
    border: none;
    border-radius: 50%;
    width: 46px !important;
    height: 38px;
    font-size: 1.2rem;
    cursor: pointer;
    margin-left: 8px;
    transition: background 0.2s;
    display: flex; align-items: center; justify-content: center;
}
.send-btn:hover {
    background: #fff;
    color: #e53935;
    border: 1.5px solid #ff00;
}

/* Enhanced Sidebar Title */
.sidebar-title {
    font-size: 5.5rem;
    color: #EE4B2B;
    font-weight: 900;
    text-align: center;
    margin: 0.5rem 0 1.5rem 0;
    letter-spacing: 0.05em;
    width: 100%;
    line-height: 1.2;
    animation: rotate3D 5s infinite linear;
    transform-style: preserve-3d;
    perspective: 800px;
    text-shadow:
        0 0 5px rgba(238, 75, 43, 0.5),
        0 0 10px rgba(238, 75, 43, 0.4),
        0 0 15px rgba(238, 75, 43, 0.3),
        1px 1px 2px rgba(0,0,0,0.8);
}

@keyframes rotate3D {
    0% { transform: rotateY(0deg) scale(1); }
    25% { transform: rotateY(90deg) scale(1.05); }
    50% { transform: rotateY(180deg) scale(1); }
    75% { transform: rotateY(270deg) scale(1.05); }
    100% { transform: rotateY(360deg) scale(1); }
}

/* Main Chatbot Title Enhancement */
.elegant-heading {
    font-size: 4.5rem !important;
    font-weight: 900;
    text-align: center;
    margin-top: -40px !important;
    color: #ffffff;
    animation: fadeInUp 2.0s ease-out;
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
.transparent-spacer1 {
height: 150px;           /* Adjust the vertical space */
background: transparent;    /* Ensures it's see-through */
}
.transparent-spacer2 {
height: 70px;           /* Adjust the vertical space */
background: transparent;    /* Ensures it's see-through */
}
.typing-indicator {
    display: flex; align-items: center; margin-bottom: 1.1rem;
}
.typing-dots span {
    height: 10px; width: 10px; margin: 0 2px;
    background: #e53935; border-radius: 25%; display: inline-block;
    animation: blink 1.2s infinite both;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; }
    40% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Model Loading (Cached)
# -------------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_sentence_transformer()

# --- Pre-load Knowledge Base ---
@st.cache_resource
def load_knowledge_base(path):
    try:
        df = pd.read_excel(path)
        required_columns = {'questions', 'answers', 'categories', 'tags'}
        if not required_columns.issubset(df.columns):
            st.error(f"⚠️ **Error:** Missing required columns: {required_columns}")
            st.stop()
        embeddings = model.encode(df['questions'].tolist())
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(np.array(embeddings))
        return df, nn_model
    except FileNotFoundError:
        st.error(f"⚠️ File not found at '{path}'")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading KB: {e}")
        st.stop()

if 'df' not in st.session_state or 'nn_model' not in st.session_state:
    st.session_state.df, st.session_state.nn_model = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    st.session_state.knowledge_base_loaded = True

# -------------------------------
# Helper Functions
# -------------------------------
def is_gibberish(text):
    text = text.strip()
    if len(text) < 2 or re.fullmatch(r'[^\w\s]+', text) or len(set(text)) < 3:
        return True
    words = text.split()
    if len(words) > 0 and sum(1 for w in words if not w.isalpha()) / len(words) > 0.5:
        return True
    return False

def is_greeting(text):
    greetings = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "sup", "thank you", "thanks", "bye", "goodbye"
    ]
    text = text.lower()
    for greet in greetings:
        if fuzz.partial_ratio(greet, text) > 80:
            return greet
    return None

def get_greeting_response(greet):
    responses = {
        "hello": "Hello! 👋 Welcome to HCIL IT Support. How may I assist you today?",
        "hi": "Hi there! 🌟 Ready to help with your IT needs!",
        "hey": "Hey! 💫 What can I help you with today?",
        "greetings": "Greetings! 🎯 I'm here to assist with any IT issues.",
        "good morning": "Good morning! ☀️ How can I brighten your day with IT solutions?",
        "good afternoon": "Good afternoon! 🌤️ Ready to tackle any IT challenges!",
        "good evening": "Good evening! 🌙 How may I assist you?",
        "how are you": "I'm functioning optimally and ready to help! 🤖✨ What brings you here?",
        "what's up": "Ready to solve IT problems! 💪 What's on your mind?",
        "sup": "All systems operational! 🚀 How can I help?",
        "thank you": "You're very welcome! 🙏 Happy to help anytime!",
        "thanks": "My pleasure! ✨",
        "bye": "Thank you for using HCIL IT Support! **Sayonara!** 👋✨",
        "goodbye": "Until next time! **Mata ne!** 🌟 Have a great day!"
    }
    return responses.get(greet, "Hello! How can I help you?")

def get_bot_response(user_query, df, nn_model, model):
    if is_gibberish(user_query):
        return "🤔 I couldn't quite understand that. Could you please rephrase your question?"

    greet = is_greeting(user_query)
    if greet:
        return get_greeting_response(greet)

    questions = df['questions'].tolist()
    best_match, score = process.extractOne(user_query, questions, scorer=fuzz.token_sort_ratio)

    if score > 70:
        idx = questions.index(best_match)
        return df.iloc[idx]['answers']

    query_embed = model.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embed)
    best_idx = indices[0][0]

    if distances[0][0] > 0.45:
        return "🤔 I couldn't find a specific answer. Could you provide more details or try rephrasing?"

    return df.iloc[best_idx]['answers']

def render_chat(messages):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-row" style="display: flex; justify-content: flex-end; align-items: flex-end;">
                <div class="chat-bubble user-bubble">{msg['content']}</div>
                <div class="avatar user-avatar">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-row" style="display: flex; justify-content: flex-start; align-items: flex-end;">
                <div class="avatar bot-avatar">🤖</div>
                <div class="chat-bubble bot-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

def show_typing():
    st.markdown("""
    <div class="typing-indicator">
        <div class="avatar bot-avatar">🤖</div>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">HCIL<span class="status-indicator"></span></div>', unsafe_allow_html=True)
    
    # System Status
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.03); border-radius: 15px; padding: 1rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
        <h4 style="color: #4ade80; margin: 0;">🟢 System Online</h4>
        <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0.5rem 0 0 0;">AI Assistant Ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    if 'messages' in st.session_state and len(st.session_state.messages) > 0:
        msg_count = len([m for m in st.session_state.messages if m['role'] == 'user'])
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 0.8rem; margin: 1rem 0; border: 1px solid rgba(102, 126, 234, 0.3);">
            <p style="margin: 0; font-size: 0.9rem;">💬 Messages: {msg_count}</p>
            <p style="margin: 0; font-size: 0.9rem;">⚡ Response Time: ~1.2s</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡**Pro Tip:** Type 'bye' to end the conversation")
    
    # Current Time
    current_time = datetime.now().strftime("%I:%M %p")
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; color: rgba(255, 255, 255, 0.5); font-size: 0.8rem;">
        {current_time} | Powered by AI
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Session State Initialization
# -------------------------------
defaults = {
    'knowledge_base_loaded': False,
    'messages': [],
    'chat_ended': False,
    'feedback_request': False,
    'quick_replies': ["Reset Password", "VPN Issues", "Software Install", "Hardware Problems"],
    'show_typing': False,
    'chat_started': False,
    'show_quick_replies': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Main Title with Animation
st.markdown("""
<h1 class='elegant-heading'>
    <span class="loading-wave">🤖</span>HCIL IT Helpdesk AI Assistant
</h1>
""", unsafe_allow_html=True)

st.markdown('<div class="transparent-spacer1"></div>', unsafe_allow_html=True)

# -------------------------------
# Chat App Flow
# -------------------------------
if not st.session_state.chat_started:
    # Welcome Message
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;">
            Experience next-generation IT support powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Start Chat Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Chat", key="start_chat_button", use_container_width=True):
            st.session_state.chat_started = True
            st.session_state.show_quick_replies = True
            welcome_messages = [
                "<b>Konnichiwa!</b> Welcome to HCIL's AI Helpdesk Support System. How may I assist you today?",
                "<b>Welcome!</b> I'm your AI assistant, ready to solve any IT challenge!",
                "<b>Hello!</b> Let's get your IT issues resolved quickly and efficiently!"
            ]
            st.session_state.messages = [{
                "role": "bot",
                "content": random.choice(welcome_messages)
            }]
            st.rerun()

else:
    if st.session_state.knowledge_base_loaded:
        # Chat History
        render_chat(st.session_state.messages)

        if st.session_state.chat_ended:
            time.sleep(2)
            for key in ['messages', 'feedback_request', 'show_typing', 'chat_started', 'show_quick_replies']:
                st.session_state[key] = defaults[key]
            st.session_state.chat_ended = False
            st.rerun()

        if st.session_state.show_typing:
            show_typing()

        # Quick Replies with Enhanced Design
        if st.session_state.show_quick_replies:
            st.markdown("""
            <div style="margin: 1.5rem 0;">
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem; margin-bottom: 0.8rem;">
                    💡 Quick Actions:
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="quick-reply-buttons">', unsafe_allow_html=True)
            cols = st.columns(len(st.session_state.quick_replies))
            for idx, (col, reply) in enumerate(zip(cols, st.session_state.quick_replies)):
                with col:
                    if st.button(reply, key=f"quick_{idx}", use_container_width=True):
                        clean_reply = reply.split(' ', 1)[1] if ' ' in reply else reply
                        st.session_state.messages.append({"role": "user", "content": clean_reply})
                        st.session_state.show_typing = True
                        st.session_state.show_quick_replies = False
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Process typing animation
        if st.session_state.show_typing:
            time.sleep(1.2)
            last_user_msg = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
            if last_user_msg:
                response = get_bot_response(last_user_msg, st.session_state.df, st.session_state.nn_model, model)
                st.session_state.messages.append({"role": "bot", "content": response})
                st.session_state.feedback_request = True
                st.session_state.show_typing = False
                st.rerun()
    else:
        st.info("🔄 Loading AI Knowledge Base...")

# -------------------------------
# Input Bar + Feedback Section
# -------------------------------
if st.session_state.chat_started and not st.session_state.chat_ended:
    # Feedback Section with Premium Design
    if st.session_state.feedback_request:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                    border-radius: 20px; padding: 1.2rem; margin: 1.5rem 0; 
                    border: 1px solid rgba(255, 255, 255, 0.1);">
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 1rem; margin-bottom: 1rem; text-align: center;">
                ✨ Was this response helpful?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="feedback-buttons">', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        feedback_options = [
            ("😊 Perfect", "Excellent! I'm here if you need anything else! ✨"),
            ("👍 Good", "Great! Feel free to ask more questions!🚀"),
            ("🤔 Unclear", "Let me try to explain differently. Could you provide more details?💭"),
            ("👎 Not Helpful", "I apologise. Let me connect you with better resources.🔄"),
            ("💬 More Help", "Sure! What else would you like to know?💡")
        ]
        
        for col, (btn_text, response) in zip([col1, col2, col3, col4, col5], feedback_options):
            with col:
                if st.button(btn_text, use_container_width=True):
                    st.session_state.messages.append({"role": "bot", "content": response})
                    st.session_state.feedback_request = False
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Spacer for better layout
    if not st.session_state.feedback_request and not st.session_state.show_quick_replies:
        st.markdown('<div class="transparent-spacer2"></div>', unsafe_allow_html=True)
    
    # Modern Input Form
    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "user_input", 
                placeholder="💭 Type your message here...", 
                key="input_bar", 
                label_visibility="collapsed"
            )
        with col2:
            send_clicked = st.form_submit_button("Send 📤", use_container_width=True)
        
        if send_clicked and user_input.strip():
            user_input_clean = user_input.lower().strip()
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            if user_input_clean in ["bye", "quit", "exit", "end"]:
                farewell_messages = [
                    "Thank you for using HCIL IT Support! <b>Mata ne!</b> 🌟 Have an amazing day!",
                    "It was great helping you! <b>Mata ne!</b> ✨ See you next time!",
                    "Thanks for choosing HCIL Chatbot! <b>Goodbye!</b> 🚀 Stay awesome!"
                ]
                st.session_state.messages.append({
                    "role": "bot", 
                    "content": random.choice(farewell_messages)
                })
                st.session_state.chat_ended = True
                st.session_state.feedback_request = False
                st.session_state.show_typing = False
            else:
                st.session_state.show_typing = True
                st.session_state.show_quick_replies = False
            st.rerun()

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.05); text-align: center;">
    <p style="color: rgba(255, 255, 255, 0.4); font-size: 0.85rem;">
        Powered by AI | HCIL IT Support © 2025 ~ by Kunal Bhardwaj 
    </p>
</div>
""", unsafe_allow_html=True)
