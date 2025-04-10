import pandas as pd
import openai
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import mysql.connector
import os

# Load environment variables
# load_dotenv()
OPENAI_API_KEY="YourAPIKey"
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# MySQL Connection Function
def get_mysql_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Rajat@123",
            database="employee",
            autocommit=True,
            connection_timeout=60
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Database connection failed: {e}")
        return None

# Fetch employee data from MySQL
def fetch_employee_data():
    conn = get_mysql_connection()
    if conn:
        query = "SELECT e.Employee_ID, e.Employee_Name, e.Contract_Type, e.Role, e.Actual_Clock_In, e.Actual_Clock_Out, e.Date FROM employee.scheduling_data  e;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    return pd.DataFrame()

# Load and Process Data
def load_and_process_data():
    df = fetch_employee_data()
    if df.empty:
        return None
    df["text"] = df.apply(lambda row: f"Employee {row['name']} (ID: {row['employee_id']}) from {row['department']} working as {row['role']} clocked in at {row['clock_in']} and clocked out at {row['clock_out']} on {row['date']}.", axis=1)
    return df

# Setup FAISS Vector Store
def setup_vector_store(df):
    documents = df["text"].tolist()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text("\n".join(documents))
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load or Create Vector Store
def get_vector_store():
    df = load_and_process_data()
    if df is None:
        return None
    return setup_vector_store(df)

# Retrieval-Augmented Generation (RAG) Setup
def setup_rag_pipeline():
    # vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain

# Initialize RAG Pipeline
qa_chain = setup_rag_pipeline()

# Streamlit UI
st.set_page_config(page_title="AttendAI", layout="wide")
st.markdown("<h1 style='text-align: center;'>AttendAI - Employee Chatbot</h1>", unsafe_allow_html=True)

user_query = st.text_input("Ask a question:")

if user_query:
    response = qa_chain.run(user_query)
    st.write(f"🟢 Chatbot: {response}")
