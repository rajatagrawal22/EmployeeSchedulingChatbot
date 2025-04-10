import pandas as pd
import openai
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load employee data
merged_df = pd.read_csv(r".\Data\Final_Adjusted_Employee_Scheduling_Data.csv")

# Merge both datasets for full context
# merged_df = pd.merge(employee_times, employee_info, on="employee_id", how="left")

# Convert to readable text for embeddings
merged_df["text"] = merged_df.apply(
    lambda row: f"Employee {row['Employee_Name']} (ID: {row['Employee_ID']}) works on {row['Contract_Type']} contract as {row['Role']}. "
                f"On {row['Date']}, they clocked in at {row['Actual_Clock_In']} and clocked out at {row['Actual_Clock_Out']} while the their clock in time was {row['Scheduled_Start']} and clock out time was{row['Scheduled_End']} .",
    axis=1
)

# Load text data
documents = merged_df["text"].tolist()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_text("\n".join(documents))

# Generate OpenAI embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)

# Save FAISS index for later use
vector_store.save_local("faiss_index")
print("✅ Employee data has been embedded and stored in FAISS!")
