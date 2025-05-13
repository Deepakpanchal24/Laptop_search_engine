import re
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

def initialize_product_search_agent(llm, data, rag_chain, chat_history):
    @tool
    def product_search_tool(query: str) -> str:
        """Search laptops based on user query, considering price range, specific features, and chat history."""
        # Extract price range from query
        price_min, price_max = None, None
        price_range_match = re.search(r'(\d+)\s*(?:rs|rupees)?\s*to\s*(\d+)\s*(?:rs|rupees)?', query.lower())
        if price_range_match:
            price_min = float(price_range_match.group(1))
            price_max = float(price_range_match.group(2))
        
        # Extract RAM and storage requirements
        ram_match = re.search(r'(\d+)\s*gb\s*ram', query.lower())
        storage_match = re.search(r'(\d+)\s*tb\s*(harddisk|hdd|ssd)', query.lower())
        required_ram = ram_match.group(1) if ram_match else None
        required_storage = storage_match.group(1) if storage_match else None
        
        # Incorporate chat history into the query
        refined_query = query + "\nPrevious chats:\n" + "\n".join([f"Q: {entry['question']}\nA: {entry['response']}" for entry in chat_history])
        
        # Run the RAG chain
        result = rag_chain.invoke(refined_query)
        
        # Fallback: Filter dataset if RAG fails
        if "Sorry" in result.content and (price_min is not None or required_ram or required_storage):
            filtered_laptops = data
            if price_min is not None and price_max is not None:
                filtered_laptops = filtered_laptops[(filtered_laptops["Price"] >= price_min) & (filtered_laptops["Price"] <= price_max)]
            if required_ram:
                filtered_laptops = filtered_laptops[filtered_laptops["Specification"].str.lower().str.contains(f"{required_ram}\s*gb\s*ram", case=False, na=False)]
            if required_storage:
                filtered_laptops = filtered_laptops[filtered_laptops["Specification"].str.lower().str.contains(f"{required_storage}\s*tb", case=False, na=False)]
            
            if not filtered_laptops.empty:
                filtered_laptops = filtered_laptops.sort_values(by="Price")
                result = f"Recommended laptops between ₹{price_min:.2f} and ₹{price_max:.2f} with {required_ram} GB RAM and {required_storage} TB storage:\n"
                result += "\n".join([f"{row['Product Name']} | {row['Specification']} | ₹{row['Price']:.2f}" for _, row in filtered_laptops.head(3).iterrows()])
            else:
                result = f"Sorry, no laptops found between ₹{price_min:.2f} and ₹{price_max:.2f} with {required_ram} GB RAM and {required_storage} TB storage in the dataset."
        
        chat_history.append({
            "question": query,
            "response": result.content if hasattr(result, 'content') else result
        })
        return result.content if hasattr(result, 'content') else result

    # Initialize agent
    agent = initialize_agent(
        tools=[product_search_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    
    return agent