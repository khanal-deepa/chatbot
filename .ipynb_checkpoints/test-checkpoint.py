# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import json

# # Initialize environment variables
# load_dotenv()
# genai_api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=genai_api_key)

# # Function to extract text from uploaded PDF files
# def extract_text_from_uploaded_pdfs(pdf_files):
#     extracted_text = ""
#     for pdf in pdf_files:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             extracted_text += page.extract_text()
#     return extracted_text

# # Function to divide text into smaller segments
# def divide_text_into_segments(text):
#     text_segmenter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     segments = text_segmenter.split_text(text)
#     return segments

# # Function to create and store a FAISS vector index
# def create_and_store_vector_index(text_segments):
#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_index = FAISS.from_texts(text_segments, embedding=embeddings_model)
#     vector_index.save_local("faiss_index")

# # Function to set up the QA chain
# def initialize_qa_chain():
#     template = """
#     Provide a comprehensive answer based on the provided context. If the answer is not available in the context, state "answer is not available in the context". Avoid giving incorrect information.\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return qa_chain

# # Function to collect user information
# def collect_user_info():
#     with st.form(key='user_info_form'):
#         st.write("Please provide your contact information:")
#         name = st.text_input("Name")
#         phone = st.text_input("Phone Number")
#         email = st.text_input("Email")
#         submit_button = st.form_submit_button(label='Submit')
        
#         if submit_button:
#             st.success("Thank you! We will contact you soon.")
#             st.write(f"Name: {name}")
#             st.write(f"Phone: {phone}")
#             st.write(f"Email: {email}")
#             # Save user information to session state
#             st.session_state.user_info = {"Name": name, "Phone": phone, "Email": email}
#             # Load previous history if it exists
#             load_history()
#             # Save history to a JSON file
#             save_history()
#             st.session_state.logged_in = True
#             st.experimental_rerun()  # Rerun to update UI

# # Function to process user questions
# def process_user_question(question):
#     if "asked_questions" not in st.session_state:
#         st.session_state.asked_questions = []

#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_index = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
#     matching_docs = vector_index.similarity_search(question)
#     qa_chain = initialize_qa_chain()
#     response = qa_chain({"input_documents": matching_docs, "question": question}, return_only_outputs=True)
#     st.write("Reply: ", response["output_text"])

#     # Check if the user requested a call
#     if "call me" in question.lower():
#         collect_user_info()
    
#     # Save asked question to session state
#     st.session_state.asked_questions.append({"Question": question, "Reply": response["output_text"]})
#     # Save history to a JSON file
#     save_history()
        
#     return response["output_text"]

# # Function to save history to a JSON file
# def save_history():
#     history_data = {"user_info": st.session_state.user_info, "asked_questions": st.session_state.asked_questions}
#     with open("history.json", "w") as json_file:
#         json.dump(history_data, json_file, indent=4)

# # Function to load history from a JSON file
# def load_history():
#     if os.path.exists("history.json"):
#         with open("history.json", "r") as json_file:
#             history_data = json.load(json_file)
#             st.session_state.user_info = history_data.get("user_info", {})
#             st.session_state.asked_questions = history_data.get("asked_questions", [])

# # Main function to run the Streamlit application
# def main():
#     if "user_info" not in st.session_state:
#         st.session_state.user_info = {}
#     if "asked_questions" not in st.session_state:
#         st.session_state.asked_questions = []
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False

#     st.set_page_config(page_title="Chat with PDF", layout="wide")
#     st.title("Interact with PDF using Gemini")

#     with st.sidebar:
#         st.header("Options")
#         st.write("Upload your PDF files and click 'Submit & Process' to analyze them.")
#         uploaded_pdfs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])

#         if uploaded_pdfs:
#             st.write("Uploaded Files:")
#             for pdf in uploaded_pdfs:
#                 st.write(f"- {pdf.name}")

#         if st.button("Submit & Process"):
#             if uploaded_pdfs:
#                 with st.spinner("Processing..."):
#                     progress_bar = st.progress(0)
#                     raw_text = extract_text_from_uploaded_pdfs(uploaded_pdfs)
#                     text_segments = divide_text_into_segments(raw_text)
#                     create_and_store_vector_index(text_segments)
#                     progress_bar.progress(100)
#                     st.success("Processing Complete")

#                     # Display the extracted text in an expandable section
#                     with st.expander("View Extracted Text"):
#                         st.write(raw_text)

#                     # Add a download button for the extracted text
#                     st.download_button(
#                         label="Download Extracted Text",
#                         data=raw_text,
#                         file_name="extracted_text.txt",
#                         mime="text/plain"
#                     )
#             else:
#                 st.error("Please upload at least one PDF file.")

#         # Show logout button if the user is logged in
#         if st.session_state.logged_in:
#             if st.button("Logout"):
#                 st.session_state.user_info = {}
#                 st.session_state.asked_questions = []
#                 st.session_state.logged_in = False
#                 st.success("Logged out successfully.")
#                 st.experimental_rerun()  # Rerun to update UI

#     st.header("Ask Questions About Your PDF Content")
#     user_question = st.text_input("Enter your question:")
#     if st.button("Submit Question"):
#         if user_question:
#             answer = process_user_question(user_question)
#         else:
#             st.error("Please enter a question.")

#     st.write("Upload multiple PDF files and ask questions about their content. The system will process the text and provide answers based on the context.")

#     # Show "Call Me" button if the user is not logged in
#     if not st.session_state.logged_in:
#         if st.button("Call Me"):
#             collect_user_info()

#     # Display history of uploaded files, asked questions, and user information
#     if st.session_state.user_info:
#         st.header("User Information:")
#         st.write(json.dumps(st.session_state.user_info, indent=4))
#     if st.session_state.asked_questions:
#         st.header("Asked Questions:")
#         for idx, question in enumerate(st.session_state.asked_questions):
#             st.subheader(f"Question {idx + 1}")
#             st.write(f"Question: {question['Question']}")
#             st.write(f"Reply: {question['Reply']}")

#     # Download button for history
#     if st.session_state.user_info or st.session_state.asked_questions:
#         st.download_button(
#             label="Download History",
#             data=json.dumps({"user_info": st.session_state.user_info, "asked_questions": st.session_state.asked_questions}, indent=4),
#             file_name="history.json",
#             mime="application/json"
#         )

# if __name__ == "__main__":
#     main()
