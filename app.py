import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain.schema import Document
from langchain.document_loaders import Docx2txtLoader

# from docx import Document as docx

import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# File upload
allowed_file_types = ["docx", "txt", "pdf"]

def generate_response(uploaded_file, openai_api_key, query_text):

    # Load document if file is uploaded
    if uploaded_file is not None:

        # Retrieve file extension
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # Check if file extension allowed
        if file_extension in allowed_file_types:

            # Read the content of the uploaded file as bytes
            file_content = uploaded_file.read()

            # Save the content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                file = temp_file.write(file_content)
                temp_file_path = temp_file.name

            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
                #Load the document by calling loader.load()
                docs = loader.load()
                
            
            elif file_extension == 'docx':
                loader = Docx2txtLoader(temp_file_path)
                docs = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap = 150,
                separators=["\n\n", "\n", " ", ""]
            ) 

            splits = text_splitter.split_documents(docs)

            # Select embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) 

            # Create a vectorstore from documents
            db = Chroma.from_documents(
                documents=splits,
                embedding=embeddings)
            
            # Build prompt
            template = """Use the following pieces of context to answer the question at the end. \
                If you don't know the answer, just say that you don't know, don't try to make up an answer. Use between 1-25 sentences unless there is a lot of information to synthesize. \
                    Keep the answer concise but do not leave out important details. Always end with "Thanks for asking 📖 🤖 !" at the end of the answer. 
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            # Create retriever interface
            retriever = db.as_retriever()

            # Create QA chain
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                retriever=retriever,
                return_source_documents=False
                )
            
            # Run chain
            result = qa_chain({"query": query_text})

            answer = result['result']

            return answer


# Page title
st.set_page_config(page_title='TL;DR Bot 📖 🤖')
st.title('TL;DR Bot 📖 🤖')


uploaded_file = st.file_uploader('Hi, I am Jemmet, the tl;dr 🤖. To start, upload a document, ask a question and enter your OpenAI API Key',
                                #  accept_multiple_files=True,
                                  type=allowed_file_types,
                                  accept_multiple_files=False)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)