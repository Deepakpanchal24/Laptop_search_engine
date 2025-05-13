import gradio as gr
from data_processing import load_and_validate_data, create_documents
from rag_setup import setup_rag
from agent_tools import initialize_product_search_agent

def main():
    # Initialize chat history
    chat_history = []
    
    # Load and process data
    try:
        data = load_and_validate_data("flipkart_laptop_cleaned.csv")
    except FileNotFoundError:
        return "Error: flipkart_laptop_cleaned.csv not found. Please ensure the file is in the project directory."
    chunks = create_documents(data)
    
    # Setup RAG pipeline
    groq_api_key = "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"
    try:
        rag_chain, llm = setup_rag(chunks, groq_api_key)
    except Exception as e:
        return f"Error setting up RAG pipeline: {str(e)}"
    
    # Initialize agent
    try:
        agent = initialize_product_search_agent(llm, data, rag_chain, chat_history)
    except Exception as e:
        return f"Error initializing agent: {str(e)}"
    
    # Gradio Interface
    def recommend_laptops(user_query):
        if not user_query:
            return "Please enter a query."
        try:
            # Use the product search tool directly for simplicity
            response = agent.tools[0].invoke({"query": user_query})
            return response
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    iface = gr.Interface(
        fn=recommend_laptops,
        inputs=gr.Textbox(label="Enter your query:", placeholder="Enter query about laptops (e.g., best laptop below 50000 Rs)"),
        outputs=gr.Textbox(label="Response"),
        title="Laptop Recommendation System",
        description="Find the best laptops based on price and specifications."
    )
    
    try:
        iface.launch(server_port=7860)
    except Exception as e:
        return f"Error launching Gradio interface: {str(e)}"

if __name__ == "__main__":
    result = main()
    if result:
        print(result)