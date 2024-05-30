import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import re

# Initialize environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

# Function to extract text from uploaded PDF files
def extract_text_from_uploaded_pdfs(pdf_files):
    extracted_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted_text += page.extract_text()
    return extracted_text

# Function to divide text into smaller segments
def divide_text_into_segments(text):
    text_segmenter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    segments = text_segmenter.split_text(text)
    return segments

# Function to create and store a FAISS vector index
def create_and_store_vector_index(text_segments):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.from_texts(text_segments, embedding=embeddings_model)
    vector_index.save_local("faiss_index")

# Function to set up the QA chain
def initialize_qa_chain():
    template = """
    Provide a comprehensive answer based on the provided context. If the answer is not available in the context, state "answer is not available in the context". Avoid giving incorrect information.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return qa_chain

# Function to process user questions
def process_user_question(session_state, question):
    if "asked_questions" not in session_state:
        session_state.asked_questions = []

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    matching_docs = vector_index.similarity_search(question)
    qa_chain = initialize_qa_chain()
    response = qa_chain({"input_documents": matching_docs, "question": question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

    # Save asked question to session state
    if session_state.logged_in:
        session_state.asked_questions.append({"Question": question, "Reply": response["output_text"]})
        # Save history to a JSON file
        save_history(session_state)
        
    return response["output_text"]

# Function to save history to a JSON file
def save_history(session_state):
    if "Email" in session_state.user_info:
        email_safe = session_state.user_info.get("Email", "").replace('@', '_').replace('.', '_')
        if email_safe:
            filename = f"history_{email_safe}.json"
            # Save history data to the file
            history_data = {"user_info": session_state.user_info, "asked_questions": session_state.asked_questions}
            with open(filename, "w") as json_file:
                json.dump(history_data, json_file, indent=4)

# Function to load history from a JSON file based on the user's credentials
def load_history(session_state):
    if "Email" in session_state.user_info:
        email_safe = session_state.user_info.get("Email", "").replace('@', '_').replace('.', '_')
        if email_safe:
            filename = f"history_{email_safe}.json"
            if os.path.exists(filename):  # Check if history file exists
                try:
                    with open(filename, "r") as json_file:
                        history_data = json.load(json_file)
                        session_state.user_info = history_data.get("user_info", {})
                        session_state.asked_questions = history_data.get("asked_questions", [])
                except FileNotFoundError:
                    session_state.user_info = {}
                    session_state.asked_questions = []
            else:
                session_state.asked_questions = []  # If history file doesn't exist, initialize a new list

# Function to check if a user with the same credentials already exists
def check_existing_user(name, phone, email):
    user_files = [f for f in os.listdir() if f.startswith("history_") and f.endswith(".json")]
    for user_file in user_files:
        with open(user_file, "r") as json_file:
            user_data = json.load(json_file)
            if (user_data.get("user_info", {}).get("Name") == name or 
                user_data.get("user_info", {}).get("Phone") == phone or 
                user_data.get("user_info", {}).get("Email") == email):
                return True
    return False

# Function to validate user credentials during login
def validate_user_credentials(name, phone, email):
    user_files = [f for f in os.listdir() if f.startswith("history_") and f.endswith(".json")]
    for user_file in user_files:
        with open(user_file, "r") as json_file:
            user_data = json.load(json_file)
            if (user_data.get("user_info", {}).get("Name") == name and 
                user_data.get("user_info", {}).get("Phone") == phone and 
                user_data.get("user_info", {}).get("Email") == email):
                return True
    return False

# Function to clear specific session state keys on logout
def clear_session_state():
    keys_to_clear = ["logged_in", "show_login_form", "user_info", "asked_questions", "show_form"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def main():
    session_state = st.session_state
    if "logged_in" not in session_state:
        session_state.logged_in = False
    if "user_info" not in session_state:
        session_state.user_info = {}
    if "asked_questions" not in session_state:
        session_state.asked_questions = []
    if "show_login_form" not in session_state:
        session_state.show_login_form = False
    if "show_form" not in session_state:
        session_state.show_form = False

    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.title("Interact with PDF using Gemini")

    if session_state.logged_in:
        load_history(session_state)

    with st.sidebar:
        st.header("Options")
        st.write("Upload your PDF files and click 'Submit & Process' to analyze them.")
        uploaded_pdfs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])

        if uploaded_pdfs:
            st.write("Uploaded Files:")
            for pdf in uploaded_pdfs:
                st.write(f"- {pdf.name}")

        if st.button("Submit & Process"):
            if uploaded_pdfs:
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    raw_text = extract_text_from_uploaded_pdfs(uploaded_pdfs)
                    text_segments = divide_text_into_segments(raw_text)
                    create_and_store_vector_index(text_segments)
                    progress_bar.progress(100)
                    st.success("Processing Complete")

                    # Display the extracted text in an expandable section
                    with st.expander("View Extracted Text"):
                        st.write(raw_text)

                    # Add a download button for the extracted text
                    st.download_button(
                        label="Download Extracted Text",
                        data=raw_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Please upload at least one PDF file.")

    st.header("Ask Questions About Your PDF Content")
    user_question = st.text_input("Enter your question:")
    if st.button("Submit Question"):
        if user_question:
            answer = process_user_question(session_state, user_question)
        else:
            st.error("Please enter a question.")

    st.write("Upload multiple PDF files and ask questions about their content. The system will process the text and provide answers based on the context.")

    if session_state.logged_in:
        st.header("History")

        # Display history of asked questions and user information
        if session_state.user_info:
            st.subheader("User Information:")
            st.write(json.dumps(session_state.user_info, indent=4))
        if session_state.asked_questions:
            st.subheader("Asked Questions:")
            for idx, question in enumerate(session_state.asked_questions):
                st.write(f"**Question {idx + 1}**: {question['Question']}")
                st.write(f"**Reply**: {question['Reply']}")

        # Logout button
        if st.button("Logout"):
            clear_session_state()
            st.success("Logged out successfully.")
            st.experimental_rerun()

        # Download button for history
        if session_state.user_info or session_state.asked_questions:
            st.download_button(
                label="Download History",
                data=json.dumps({"user_info": session_state.user_info, "asked_questions": session_state.asked_questions}, indent=4),
                file_name="history.json",
                mime="application/json"
            )
    else:
        if st.button("Call Me"):
            session_state.show_form = True

        if session_state.show_form:
            # Collect user information for login or registration
            with st.form(key='user_info_form'):
                st.write("Please provide your contact information:")
                name = st.text_input("Name")
                phone = st.text_input("Phone Number")
                email = st.text_input("Email")
                submit_button = st.form_submit_button(label='Submit')
                
                if submit_button:
                    email_regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    phone_regex = r'^\+?\d{10,15}$'
                    
                    if not name:
                        st.error("Please fill in all fields.")
                    elif not re.match(phone_regex, phone):
                        st.error("Please enter a valid phone number (10 to 15 digits, optionally starting with +).")
                    elif not re.match(email_regex, email):
                        st.error("Please enter a valid email address.")
                    else:
                        if validate_user_credentials(name, phone, email):
                            st.success("Logged in successfully.")
                            session_state.user_info = {"Name": name, "Phone": phone, "Email": email}
                            session_state.logged_in = True
                            session_state.show_login_form = False
                            load_history(session_state)
                            st.experimental_rerun()
                        elif check_existing_user(name, phone, email):
                            st.error("User with the same name, phone number, or email already exists.")
                        else:
                            st.success("Thank you! Your account has been created and you are logged in.")
                            session_state.user_info = {"Name": name, "Phone": phone, "Email": email}
                            session_state.logged_in = True
                            session_state.show_login_form = False
                            save_history(session_state)
                            st.experimental_rerun()

if __name__ == "__main__":
    main()
