import os
import streamlit as st
import re
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from PIL import Image
import time

# Google API Key
GOOGLE_API_KEY = "AIzaSyDOrv3RayLX8j0B9C_cWwncoDjVfVHwZds"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Updated CSS with Enhanced Chat-like Suggestion Buttons
def load_css():
    st.markdown("""
    <style>
    :root {
        --primary-color: #6a11cb;
        --secondary-color: #2575fc;
        --accent-color: #ff7e5f;
        --background-color: #f8fafc;
        --text-color: #1e293b;
        --light-text: #64748b;
        --suggestion-bg: #e6e9f0;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    h1, h2, h3 {
        color: var(--text-color);
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    h1 {
        font-size: 2.5rem !important;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #e2e8f0;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
        transform: translateY(-2px);
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #4338ca 0%, #3730a3 100%);
    }
    .css-1d391kg .css-1v3fvcr {
        color: var(--text-color);
    }
    .sidebar-title {
        background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-title h2 {
        color: white;
        margin: 0;
    }
    .sidebar-title p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        padding: 1rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
    }
    .stSelectbox>div>div {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
    }
    .info-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
    }
    .user-bubble {
        background-color: #818cf8;
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .assistant-bubble {
        background-color: white;
        color: #1e293b;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid var(--primary-color);
    }
    .suggestion-button button {
        background-color: var(--suggestion-bg);
        border: none;
        border-radius: 15px;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        width: auto;
    }
    .suggestion-button button:hover {
        background-color: #d1d5db;
        transform: translateY(-2px);
    }
    .quick-questions {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-radius: 50%;
        border-top: 5px solid var(--primary-color);
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 {
            color: #ffffff;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        }
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        .css-1v3fvcr {
            color: #ffffff;
        }
        .suggestion-button button {
            background-color: #4b5563;
            color: white;
        }
        .suggestion-button button:hover {
            background-color: #6b7280;
        }
        .quick-questions {
            color: #ffffff;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "Answer is not available in the context." Context:\n{context}\n Question:\n{question}\n Answer: """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    normalized_question = user_question.lower()
    keyword_mapping = {
        "hod": "head", "cse": "computer science and engineering", "ece": "electrical and communication engineering",
        "eee": "electrical and electronics engineering", "ai": "artificial intelligence", "ds": "data science",
    }
    for key, value in keyword_mapping.items():
        normalized_question = normalized_question.replace(key, value)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(normalized_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": normalized_question}, return_only_outputs=True)
    return response["output_text"]

def extract_csv(pathname: str) -> list[str]:
    parts = []
    try:
        with open(pathname, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            parts.append(",".join(header))
            for row in csv_reader:
                parts.append(",".join(row))
        return parts
    except FileNotFoundError:
        st.error(f"File not found: {pathname}")
        return []
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

def is_valid_roll_number(roll_number: str, combined_data: list) -> bool:
    pattern = r'^(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}$'
    if re.match(pattern, roll_number):
        return any(roll_number in entry for entry in combined_data)
    return False

def load_backlog_data(path: str) -> list[dict]:
    backlog_data = []
    with open(path, "r", newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            backlog_data.append(row)
    return backlog_data

# Team Members Data
team_members = {
    "Swathi Priya": {
        "Role": "Team Lead", "Department": "AI&DS", "College": "Ramachandra College of Engineering",
        "Email": "dswathipriya22@gmail.com", "Image_Path": r"Swathi.jpg"
    },
    "K. Kasyap": {
        "Roll Number": "21ME1A5421", "Branch": "Artificial Intelligence and Data Science",
        "Email": "saiumakasyap@gmail.com", "Image_Path": r"Kasyap.jpg"
    },
    "K. Srihitha": {
        "Roll Number": "21ME1A5433", "Branch": "Artificial Intelligence and Data Science",
        "Email": "srihithakudaravalli87@gmail.com", "Image_Path": r"Srihitha.jpg"
    },
    "SK. Asma": {
        "Roll Number": "21ME1A5457", "Branch": "Artificial Intelligence and Data Science",
        "Email": "asmashaik6281@gmail.com", "Image_Path": r"asma.jpg"
    }
}

# UI Helper Functions
def display_chat_message(message, is_user=True):
    if is_user:
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{message}</div>', unsafe_allow_html=True)

def create_info_card(title, content, icon="ℹ️"):
    st.markdown(f"""
    <div class="info-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def display_loading_animation():
    cols = st.columns(3)
    with cols[1]:
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
    time.sleep(1.5)

def show_welcome_animation():
    st.markdown("""
    <div class="animated" style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">👋</div>
        <h2>🎓 RCEE Interactive Assistant</h2>
        <p style="color: #64748b;">Your intelligent guide to college information and student analysis</p>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.5)

# Main Application
def main():
    st.set_page_config(page_title="🎓 RCEE Interactive Assistant", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
    load_css()

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">
            <h2>🎓 RCEE Assistant</h2>
            <p>Your campus companion</p>
        </div>
        """, unsafe_allow_html=True)
        app_mode = st.radio("", ["🏛️ College Info", "📊 Student Marks", "📋 Backlogs Comparison", "👥 Team & Project"], 
                           label_visibility="collapsed")

    # Initialize session state for modes that need conversation history
    if "college_messages" not in st.session_state:
        st.session_state.college_messages = []
    if "backlogs_messages" not in st.session_state:
        st.session_state.backlogs_messages = []

    # College Info Mode
    if app_mode == "🏛️ College Info":
        if "welcome_shown" not in st.session_state:
            show_welcome_animation()
            st.session_state["welcome_shown"] = True
        
        st.markdown('<h1 class="fade-in">🏛️ RCEE College Interactive Assistant</h1>', unsafe_allow_html=True)
        
        pdf_file_paths = [r"RCEE.pdf"]
        raw_text = get_pdf_text(pdf_file_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Display conversation history
        if not st.session_state.college_messages:
            display_chat_message("Hello! I'm your RCEE Assistant. Ask me anything about the college!", is_user=False)
        else:
            for message in st.session_state.college_messages:
                display_chat_message(message["content"], is_user=(message["role"] == "user"))

        # Quick Questions Section
        suggestions = [
            {"text": "What programs are offered?", "emoji": "🎓"},
            {"text": "Who is the principal?", "emoji": "👨‍🏫"},
            {"text": "What is the vision of RCEE?", "emoji": "🌟"},
            {"text": "Tell me about the college?", "emoji": "🏛"},
            {"text": "What academic accreditations does RCEE hold?", "emoji": "📚"}
        ]
        
        st.markdown('<div class="quick-questions">Quick Questions:</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(f"{suggestion['emoji']} {suggestion['text']}", key=f"suggestion_{i}"):
                    st.session_state.college_messages.append({"role": "user", "content": suggestion['text']})
                    with st.status("Processing your question..."):
                        response = user_input(suggestion['text'])
                    st.session_state.college_messages.append({"role": "assistant", "content": response})
                    st.rerun()

        # Chat input
        user_query = st.chat_input("Ask me about RCEE College...")
        if user_query:
            st.session_state.college_messages.append({"role": "user", "content": user_query})
            with st.status("Processing your question..."):
                response = user_input(user_query)
            st.session_state.college_messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Student Marks Mode
    elif app_mode == "📊 Student Marks":
        st.markdown('<h1 class="fade-in">📊 Student Result Analysis System</h1>', unsafe_allow_html=True)
        
        # Display initial message (no conversation history)
        display_chat_message("Hello! I can help analyze student marks. Please provide a roll number and question.", is_user=False)
        
        batch_selection = st.sidebar.selectbox('Batch:', ['Select Batch', 'AI & DS 2021-2025', 'AI & DS 2020-2024'])
        
        if batch_selection == 'AI & DS 2021-2025':
            st.subheader(f"{batch_selection} Student Marks Portal")
            csv_paths_ai_ds_2021_2025 = {
                "1-1": r"1-1sem.csv", "1-2": r"1-2sem.csv",
                "2-1": r"2-1sem.csv", "2-2": r"2-2sem.csv",
                "3-1": r"3-1sem.csv", "3-2": r"3-2sem.csv",
            }
            semester = st.sidebar.select_slider('Semester:', options=["1-1", "1-2", "2-1", "2-2", "3-1", "3-2"])
            combined_data_2021_2025 = extract_csv(csv_paths_ai_ds_2021_2025[semester])

            create_info_card("🔍 Ask about your results", 
                          "Enter your roll number and question. Example: 'Show my results for 21A91A6630'")
            
            user_question = st.text_input("", placeholder="Example: What are my marks? Roll number: 21A91A6630", key="marks_input")
            if st.button("🚀 Analyze My Results", key="marks_analyze"):
                display_loading_animation()
                if user_question:
                    if roll_number_match := re.search(r'\b(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}\b', user_question):
                        roll_number = roll_number_match.group(0)
                        if is_valid_roll_number(roll_number, combined_data_2021_2025):
                            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                            prompt = f"Analyze this CSV data for semester {semester}:\n{''.join(combined_data_2021_2025)}\nQuestion: {user_question}"
                            response = model.generate_content(prompt)
                            display_chat_message(user_question, is_user=True)
                            display_chat_message(response.text, is_user=False)
                        else:
                            display_chat_message(f"Roll number {roll_number} not found.", is_user=False)
                    else:
                        display_chat_message("Please include a valid roll number in your query.", is_user=False)

        elif batch_selection == 'AI & DS 2020-2024':
            st.subheader(f"{batch_selection} Student Marks Portal")
            csv_paths_ai_ds_2020_2024 = {
                "1-1": r"1-1sems.csv", "1-2": r"1-2sems.csv",
                "2-1": r"2-1sems.csv", "2-2": r"2-2sems.csv",
                "3-1": r"3-1sems.csv", "3-2": r"3-2sems.csv"
            }
            combined_data_2020_24 = []
            for path in csv_paths_ai_ds_2020_2024.values():
                combined_data_2020_24.extend(extract_csv(path))
            
            create_info_card("🔍 Ask about your results", 
                          "Enter your roll number and question. Example: 'Show my results for 20A91A6630'")
            
            user_query = st.text_input("", placeholder="Example: What are my marks? Roll number: 20A91A6630", key="marks_input_2020")
            if st.button("🚀 Analyze My Results", key="marks_analyze_2020"):
                display_loading_animation()
                if user_query:
                    if roll_number_match := re.search(r'\b(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}\b', user_query):
                        roll_number = roll_number_match.group(0)
                        if is_valid_roll_number(roll_number, combined_data_2020_24):
                            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                            prompt = f"Analyze this CSV data:\n{''.join(combined_data_2020_24)}\nQuestion: {user_query}"
                            response = model.generate_content(prompt)
                            display_chat_message(user_query, is_user=True)
                            display_chat_message(response.text, is_user=False)
                        else:
                            display_chat_message(f"Roll number {roll_number} not found.", is_user=False)
                    else:
                        display_chat_message("Please include a valid roll number in your query.", is_user=False)

    # Backlogs Comparison Mode
    elif app_mode == "📋 Backlogs Comparison":
        st.markdown('<h1 class="fade-in">📋 Backlogs Comparison</h1>', unsafe_allow_html=True)
        
        # Display conversation history
        if not st.session_state.backlogs_messages:
            display_chat_message("Hello! I can help with backlog information. Please ask your question.", is_user=False)
        else:
            for message in st.session_state.backlogs_messages:
                display_chat_message(message["content"], is_user=(message["role"] == "user"))
        
        csv_path = r"C:\Users\rishi\Desktop\Vijaya\Backlog.csv"
        csv_data = extract_csv(csv_path)
        
        if csv_data:
            create_info_card("Backlog Data Overview", "Ask questions about student backlogs below.")
            
            user_question = st.text_input("", placeholder="Example: How many backlogs does 21A91A6630 have?", key="backlogs_input")
            if st.button("🚀 Submit", key="backlogs_submit"):
                display_loading_animation()
                if user_question:
                    st.session_state.backlogs_messages.append({"role": "user", "content": user_question})
                    model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")
                    prompt = f"Analyze this CSV data:\n{''.join(csv_data)}\nQuestion: {user_question}"
                    response = model.generate_content(prompt)
                    st.session_state.backlogs_messages.append({"role": "assistant", "content": response.text})
                    st.rerun()
                else:
                    st.session_state.backlogs_messages.append({"role": "assistant", "content": "Please enter a question."})
                    st.rerun()

    # Team Members & Project Details Mode
    elif app_mode == "👥 Team & Project":
        st.markdown('<h1 class="fade-in">👥 Team & Project Details</h1>', unsafe_allow_html=True)
        st.snow()
        selected_option = st.sidebar.selectbox("Choose an Option", ["Team Members", "Project Details"])

        if selected_option == "Team Members":
            st.subheader("👥 Team Members")
            for name, details in team_members.items():
                cols = st.columns([1, 3])
                with cols[0]:
                    try:
                        image = Image.open(details["Image_Path"]).resize((150, 150))
                        st.image(image, caption=name)
                    except Exception as e:
                        st.error(f"Error loading image for {name}: {e}")
                with cols[1]:
                    create_info_card(name, "<br>".join([f"{k}: {v}" for k, v in details.items() if k != "Image_Path"]))

        elif selected_option == "Project Details":
            st.subheader("🚀 Project Details")
            create_info_card("Project Title", "AI-Powered College Information and Student Performance Chatbot Using LLMs")
            create_info_card("Batch No", "54A03")
            create_info_card("Branch", "Artificial Intelligence & Data Science")
            create_info_card("Mentor", "Mr. K. Kiran sir")
            create_info_card("Abstract", """This application leverages large language models (LLMs) to create an interactive chatbot that offers college information and student performance analysis. The College Info function allows users to inquire about topics such as academic programs, faculty, and accreditation. Using a custom PDF parsing function, the chatbot extracts relevant information from college documents, divides it into manageable sections, and stores these in a FAISS vector store to enable efficient semantic search. An LLM then generates contextually accurate answers based on user queries. In addition, the Student Marks function provides an interface for analyzing student results by batch. It retrieves data from CSV files, validates user-input roll numbers, and, when valid, presents relevant academic details. The LLM further processes inquiries related to student data, ensuring clear and accurate responses. Overall, this chatbot combines cutting-edge AI techniques to deliver precise information retrieval and real-time data analysis, offering valuable support for students and administrators alike through an intuitive, accessible platform.""")

if __name__ == "__main__":
    main()
