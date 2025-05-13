from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

def setup_rag(chunks, groq_api_key):
    # Initialize embeddings and vectorstore
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Save the vectorstore locally
    vectorstore.save_local("faiss_index_laptops")
    
    # Load the LLM model (Groq)
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.0,
            max_tokens=500
        )
    except Exception as e:
        print(f"Error initializing Groq model: {str(e)}")
        print("Please ensure your Groq API key is valid.")
        exit(1)
    
    # Define the prompt template
    template = """
    You are an expert assistant for laptop recommendations, specializing in finding laptops that precisely match user queries.

    Your task is to analyze the provided context, which contains laptop details in the format: Product Name | Specification | Price. Recommend laptops that exactly match the user's query, including price range and specific features (e.g., RAM, storage).

    Guidelines:
    - If the query specifies a price range (e.g., "₹50000 to ₹70000"), only recommend laptops within that range.
    - If specific features are requested (e.g., "16 GB RAM", "1 TB harddisk"), only include laptops that match those features exactly.
    - Define "best" as laptops with the strongest specifications (e.g., faster processor, higher RAM, larger storage) within the price range and requested features.
    - List up to 3 laptops, sorted by price (cheapest first), including product name, key specifications, and price.
    - If no laptops match all criteria, respond with: "Sorry, no laptops found matching your criteria in the dataset."
    - Do not invent information, include laptops outside the context, or make assumptions about user intent (e.g., "for college students").
    - Keep the response concise and user-friendly.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    
    # Initialize prompt
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    
    # Initialize the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Set up the RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
    )
    
    return rag_chain, llm