import os
import json
from flask import Flask, request, jsonify
from google.cloud import secretmanager_v1
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

app = Flask(__name__)

# --- Configuration from Environment Variables ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
ALLOYDB_DATABASE_NAME = os.environ.get("ALLOYDB_DATABASE_NAME")
ALLOYDB_TABLE_NAME = os.environ.get("ALLOYDB_TABLE_NAME")
ALLOYDB_CLUSTER_NAME = os.environ.get("ALLOYDB_CLUSTER_NAME")
ALLOYDB_INSTANCE_NAME = os.environ.get("ALLOYDB_INSTANCE_NAME")
ALLOYDB_SECRET_NAME = os.environ.get("ALLOYDB_SECRET_NAME")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Initialize Secret Manager Client ---
secret_manager_client = secretmanager_v1.SecretManagerServiceClient()

def get_alloydb_password():
    try:
        secret_name = secret_manager_client.secret_version_path(
            project=PROJECT_ID, secret=ALLOYDB_SECRET_NAME, secret_version="latest"
        )
        secret_request = secretmanager_v1.AccessSecretVersionRequest(name=secret_name)
        secret_response = secret_manager_client.access_secret_version(request=secret_request)
        return secret_response.payload.data.decode("UTF-8").strip()
    except Exception as e:
        app.logger.error(f"Failed to retrieve AlloyDB password from Secret Manager: {e}")
        raise

# --- Initialize AlloyDB Engine and Vector Store ---
vectorstore = None
engine = None
try:
    PGPASSWORD = get_alloydb_password()
    engine = AlloyDBEngine.from_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=ALLOYDB_CLUSTER_NAME,
        instance=ALLOYDB_INSTANCE_NAME,
        database=ALLOYDB_DATABASE_NAME,
        user="postgres",
        password=PGPASSWORD
    )

    vectorstore = AlloyDBVectorStore.create_sync(
        engine=engine,
        table_name=ALLOYDB_TABLE_NAME,
        embedding_service=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GOOGLE_API_KEY
        ),
        id_column="id",
        content_column="description",
        embedding_column="product_embedding",
        metadata_columns=["id", "name", "categories", "price_usd_units", "price_usd_nanos", "price_usd_currency_code"]
    )
    app.logger.info("AlloyDB and Vector Store initialized successfully.")
except Exception as e:
    app.logger.error(f"Error initializing AlloyDB or Vector Store: {e}")
    vectorstore = None

# --- Define Tools for Gemini to Call ---

def search_products_tool(query: str, max_price: float = None, min_price: float = None, currency: str = "USD") -> str:
    """
    Search for products in the database using semantic similarity.
    
    Args:
        query: Search query (e.g., "sunglasses", "kitchen items")
        max_price: Maximum price filter (optional)
        min_price: Minimum price filter (optional)
        currency: Currency code for price filtering (default: USD)
    
    Returns:
        JSON string with product IDs and basic info
    """
    if not vectorstore:
        return json.dumps({"error": "Database not available"})
    
    try:
        results = vectorstore.similarity_search(query, k=10)
        
        products = []
        for doc in results:
            price_units = doc.metadata.get('price_usd_units', 0)
            price_nanos = doc.metadata.get('price_usd_nanos', 0)
            price_code = doc.metadata.get('price_usd_currency_code', 'USD')
            product_price = price_units + price_nanos / 1e9
            
            # Apply price filters
            if price_code != currency:
                continue
            if min_price is not None and product_price < min_price:
                continue
            if max_price is not None and product_price > max_price:
                continue
            
            products.append({
                "id": doc.metadata['id'],
                "name": doc.metadata.get('name', ''),
                "price": product_price,
                "currency": price_code
            })
        
        return json.dumps({"products": products, "count": len(products)})
    except Exception as e:
        app.logger.error(f"Error in search_products_tool: {e}")
        return json.dumps({"error": str(e)})

def get_product_by_id_tool(product_id: str) -> str:
    """
    Get detailed information about a specific product by ID.
    
    Args:
        product_id: The product ID to look up
    
    Returns:
        JSON string with product details
    """
    if not vectorstore:
        return json.dumps({"error": "Database not available"})
    
    try:
        # Query database for specific product
        with engine.connect() as conn:
            result = conn.execute(
                f"SELECT id, name, categories, price_usd_units, price_usd_nanos, price_usd_currency_code, description FROM {ALLOYDB_TABLE_NAME} WHERE id = %s",
                (product_id,)
            )
            row = result.fetchone()
            
            if row:
                product = {
                    "id": row[0],
                    "name": row[1],
                    "categories": row[2],
                    "price_units": row[3],
                    "price_nanos": row[4],
                    "currency": row[5],
                    "description": row[6]
                }
                return json.dumps(product)
            else:
                return json.dumps({"error": f"Product {product_id} not found"})
    except Exception as e:
        app.logger.error(f"Error in get_product_by_id_tool: {e}")
        return json.dumps({"error": str(e)})

def get_gift_recommendations_tool(gender: str, age: int, preferences: str) -> str:
    """
    Get gift recommendations based on recipient details.
    
    Args:
        gender: Gender of the recipient
        age: Age of the recipient
        preferences: Interests or preferences (e.g., "tech gadgets", "kitchen")
    
    Returns:
        JSON string with recommended product IDs
    """
    if not vectorstore:
        return json.dumps({"error": "Database not available"})
    
    try:
        gift_query = f"gift for {age} year old {gender} who likes {preferences}"
        results = vectorstore.similarity_search(gift_query, k=5)
        
        products = []
        for doc in results:
            products.append({
                "id": doc.metadata['id'],
                "name": doc.metadata.get('name', '')
            })
        
        return json.dumps({"products": products, "count": len(products)})
    except Exception as e:
        app.logger.error(f"Error in get_gift_recommendations_tool: {e}")
        return json.dumps({"error": str(e)})

def list_product_categories_tool() -> str:
    """
    List all available product categories in the database.
    
    Returns:
        JSON string with category list
    """
    if not engine:
        return json.dumps({"error": "Database not available"})
    
    try:
        with engine.connect() as conn:
            result = conn.execute(
                f"SELECT DISTINCT categories FROM {ALLOYDB_TABLE_NAME}"
            )
            categories = [row[0] for row in result.fetchall()]
            return json.dumps({"categories": categories})
    except Exception as e:
        app.logger.error(f"Error in list_product_categories_tool: {e}")
        return json.dumps({"error": str(e)})

# --- Create Tools List ---
tools = [
    Tool(
        name="search_products",
        func=search_products_tool,
        description="Search for products using semantic similarity. Use this when the user wants to find products matching a description, category, or price range. Returns product IDs."
    ),
    Tool(
        name="get_product_by_id",
        func=get_product_by_id_tool,
        description="Get detailed information about a specific product by ID. Use this when you need more details about a product."
    ),
    Tool(
        name="get_gift_recommendations",
        func=get_gift_recommendations_tool,
        description="Get gift recommendations based on recipient's gender, age, and preferences. Only use when you have all three pieces of information."
    ),
    Tool(
        name="list_categories",
        func=list_product_categories_tool,
        description="List all available product categories. Use this when user asks what types of products are available."
    )
]

# --- Initialize Gemini with Tools ---
llm_with_tools = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# --- Create Agent Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI shopping assistant with access to a product database.
    
Your goal is to help users find products, make recommendations, and manage their shopping experience.

When responding, ALWAYS output a JSON array of actions. Available actions:
- {{"task": "response", "message": "<text>"}}
- {{"task": "search", "product_ids": ["id1", "id2"], "message": "<text>"}}
- {{"task": "add-to-cart", "product_ids": ["id1"], "quantity": <number>}}
- {{"task": "empty-cart", "message": "<text>"}}
- {{"task": "view-cart"}}
- {{"task": "recommend"}}
- {{"task": "gift-recommendation", "person_details": {{"gender": "<text>", "age": <num>, "preferences": "<text>"}}, "product_ids": ["id1"]}}
- {{"task": "compare", "product_ids": ["id1", "id2"]}}
- {{"task": "checkout"}}

IMPORTANT RULES:
1. Use the tools to search the database - DON'T make up product IDs
2. For gift recommendations, if details are missing, ask for them ONE AT A TIME
3. Extract product IDs from tool results and include them in your actions
4. Always include helpful messages
5. For price queries, extract the amount and use search_products tool with filters

User Context: {user_context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Create Agent ---
agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

@app.route("/process_query", methods=['POST'])
def process_query():
    """Process user queries using Gemini with database tools"""
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('conversation_history', [])
    user_context = data.get('user_context', {})
    image_data = data.get('image', None)

    app.logger.info(f"Processing: '{user_message}'")
    
    try:
        # Format conversation history for agent
        chat_history = []
        for msg in conversation_history[-5:]:  # Last 5 messages
            if msg.get('type') == 'user':
                chat_history.append(HumanMessage(content=msg.get('content', '')))
        
        # Execute agent
        result = agent_executor.invoke({
            "input": user_message,
            "chat_history": chat_history,
            "user_context": json.dumps(user_context)
        })
        
        output = result.get('output', '{}')
        app.logger.info(f"Agent output: {output}")
        
        # Parse the output
        try:
            # Try to extract JSON from the output
            import re
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                actions = json.loads(json_match.group(0))
            else:
                # If no JSON found, wrap in response action
                actions = [{"task": "response", "message": output}]
        except json.JSONDecodeError:
            actions = [{"task": "response", "message": output}]
        
        return jsonify({"actions": actions})
        
    except Exception as e:
        app.logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            "actions": [{
                "task": "response",
                "message": "I encountered an error. Please try again."
            }]
        })

@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database": vectorstore is not None,
        "tools_available": len(tools)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
