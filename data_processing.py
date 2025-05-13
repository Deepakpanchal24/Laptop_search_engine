import pandas as pd
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_validate_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    data = df[["Product Name", "Specification", "Price"]].copy()

    # Validate Price column
    try:
        data["Price"] = data["Price"].astype(float)
    except ValueError as e:
        print("Warning: Non-numeric values found in Price column. Please ensure all prices are numeric (e.g., 50000.00).")
        print(f"Error details: {e}")
        exit(1)

    return data

def format_row(row):
    return f"{row['Product Name']} | {row['Specification']} | ₹{row['Price']:.2f}"

def create_documents(data):
    # Create documents for each product with price metadata
    docs = [Document(page_content=format_row(row), metadata={"price": row["Price"], "specification": row["Specification"]}) for _, row in data.iterrows()]
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    return chunks