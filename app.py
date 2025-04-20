#This is Version 1.2, which includes the functionality for CSV Question and Answer (QA).
import streamlit as st
import tempfile
import pandas as pd
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template,styl
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI
import pytesseract
from pdf2image import convert_from_bytes
import io
from PIL import Image


def get_file_text(uploaded_files):
    text = ""
    csv_files = []
    for file in uploaded_files:
        if file.type == 'application/pdf':
            print("PDF is selected")
            text += get_pdf_text(file)
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            print("Word is selected")
            text += get_docx_text(file)
        elif file.type == 'text/csv':
            print("CSV is selected")
            csv_files.append(file)
        else:
            st.warning(f"Unsupported file type: {file.type}")
    
    return text, csv_files

def get_docx_text(docx_file):
    st.warning("DOCX processing not yet implemented")
    return ""


def get_pdf_text(pdf_file):
    text = ""
    # Convert the Streamlit file object to a bytes stream
    pdf_bytes = pdf_file.getvalue()
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    
    text_extracted = False
    
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text += page_text
            text_extracted = True
    
    # If no text was extracted, use OCR
    if not text_extracted:
        images = convert_from_bytes(pdf_bytes)
        
        for image in images:
            pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\rvent\\Desktop\\code\\tesseract.exe'
            page_text = pytesseract.image_to_string(image)
            text += page_text
    
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation and st.session_state.csv_agent:
        # Try both conversation and CSV agent
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
        except Exception as e:
            # If conversation fails, try CSV agent
            try:
                csv_response = st.session_state.csv_agent.run(user_question)
                st.write(bot_template.replace(
                    "{{MSG}}", csv_response), unsafe_allow_html=True)
            except Exception as csv_e:
                st.error(f"Error processing question: {str(csv_e)}")
    elif st.session_state.csv_agent:
        # Only CSV agent available
        try:
            csv_response = st.session_state.csv_agent.run(user_question)
            st.write(bot_template.replace(
                "{{MSG}}", csv_response), unsafe_allow_html=True)
        except Exception as e:
            # If there's a parsing error, try to extract the response from the error message
            error_str = str(e)
            if "Could not parse LLM output:" in error_str:
                # Extract the response after the error message
                response = error_str.split("Could not parse LLM output:")[1].strip()
                st.write(bot_template.replace(
                    "{{MSG}}", response), unsafe_allow_html=True)
            else:
                st.error(f"Error processing CSV question: {str(e)}")
    elif st.session_state.conversation:
        # Only conversation available
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
    else:
        st.error("Please upload and process documents first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs and CSVs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "csv_agent" not in st.session_state:
        st.session_state.csv_agent = None

    st.header("Chat with multiple PDFs and CSVs :books:")
    st.markdown(styl, unsafe_allow_html=True)
    
    # Display welcome message
    welcome_message = "Hi, this is a Chatbot that you can ask questions to your PDF and CSV files. RAG techniques are used in this chatbot. You can upload a document, then select process. After that you can ask a question and get answer for your question. Have fun!"
    st.write(bot_template.replace("{{MSG}}", welcome_message), unsafe_allow_html=True)
    
    # User question input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your files here. PDF, DOCX, and CSV files are accepted.", 
            accept_multiple_files=True
        )
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text, csv_files = get_file_text(uploaded_files)

                # Process PDF/text files
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                # Process CSV files
                if csv_files:
                    csv_file = csv_files[0]
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(csv_file.getvalue())
                        tmp_file_path = tmp_file.name

                    llm = ChatOpenAI(temperature=0)
                    st.session_state.csv_agent = create_csv_agent(llm, tmp_file_path, verbose=False, allow_dangerous_code=True)
                    st.success("CSV file processed successfully!")
                    
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)


if __name__ == '__main__':
    main()