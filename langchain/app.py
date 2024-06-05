from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')
st.title("chat with llama3 demo")

llm=ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:
        try:
            # Initialize embeddings
            st.session_state.embeddings = OpenAIEmbeddings()
            
            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader("./data")
            st.session_state.docs = st.session_state.loader.load()
            
            # Debug: Check the loaded documents
            if not st.session_state.docs:
                st.error("No documents were loaded.")
                return
            
            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            
            # Debug: Check the final_documents list
            if not st.session_state.final_documents:
                st.error("No documents were split into chunks.")
                return
            
            # Create vector store from documents and embeddings
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector Store DB is ready.")
        except Exception as e:
            st.error(f"An error occurred during vector embedding: {e}")



prompt1=st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write("PAGE NO:",doc.metadata['page'],"Content is:",doc.page_content)
            st.write("--------------------------------")

