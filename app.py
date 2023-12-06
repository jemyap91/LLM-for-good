import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import FAISS

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Read the content of the uploaded file as bytes
        file_content = uploaded_file.read()

        # Save the content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        #Load the document by calling loader.load()
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
        
        # Create retriever interface
        retriever = db.as_retriever()

        # Create QA chain
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            chain_type='stuff',
            retriever=retriever
            )
        result = qa_chain({"query": query_text})

        return result['result']
        # return qa.run(query_text)


# Page title
st.set_page_config(page_title='Let\'s make Economics easy again!')
st.title('Let\'s make Economics easy again!')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
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