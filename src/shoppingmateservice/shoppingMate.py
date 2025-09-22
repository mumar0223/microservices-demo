# src/shoppingmate_ai_service/main.py
import os
import json
from flask import Flask, request, jsonify
from google.cloud import secretmanager_v1
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_core.messages import HumanMessage
import re

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

app = Flask(__name__)

# --- Configuration from Environment Variables ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
ALLOYDB_DATABASE_NAME = os.environ.get("ALLOYDB_DATABASE_NAME")
ALLOYDB_TABLE_NAME = os.environ.get("ALLOYDB_TABLE_NAME")
ALLOYDB_CLUSTER_NAME = os.environ.get("ALLOYDB_CLUSTER_NAME")
ALLOYDB_INSTANCE_NAME = os.environ.get("ALLOYDB_INSTANCE_NAME")
ALLOYDB_SECRET_NAME = os.environ.get("ALLOYDB_SECRET_NAME")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Ensure this is set in your deployment

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

    # Ensure price_usd is included in metadata_columns for filtering
    vectorstore = AlloyDBVectorStore.create_sync(
        engine=engine,
        table_name=ALLOYDB_TABLE_NAME,
        embedding_service=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY),
        id_column="id",
        content_column="description",
        embedding_column="product_embedding",
        metadata_columns=["id", "name", "categories", "price_usd_units", "price_usd_nanos", "price_usd_currency_code"] # Include price components
    )
    app.logger.info("AlloyDB and Vector Store initialized successfully.")
except Exception as e:
    app.logger.error(f"Error initializing AlloyDB or Vector Store: {e}")
    vectorstore = None # Ensure vectorstore is None if initialization fails

# --- Gemini Models ---
llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
llm_text = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

@app.route("/process_query", methods=['POST'])
def process_query():
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('conversation_history', [])
    user_context = data.get('user_context', {})
    image_data = data.get('image', None) # Base64 image data

    app.logger.info(f"Received query: '{user_message}'")
    app.logger.info(f"Conversation History: {conversation_history}")
    app.logger.info(f"User Context: {user_context}")

    actions = []
    room_description = ""
    try:
        # --- Step 1: Get room description from image if provided ---
        if image_data:
            app.logger.info("Processing image for room description...")
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the interior design style of the room in this image."},
                    {"type": "image_url", "image_url": image_data},
                ]
            )
            vision_response = llm_vision.invoke([vision_message])
            room_description = vision_response.content
            app.logger.info(f"Room description from image: {room_description}")
            actions.append({"task": "response", "message": f"I see a room with a {room_description} style."})

        # --- Step 2: Determine user intent and generate actions ---
        # IMPORTANT: The prompt is crucial for guiding the AI's behavior.
        # It instructs the AI to return specific JSON actions and product IDs.
        prompt_parts = [
            "You are an AI shopping assistant for an e-commerce platform. Your goal is to understand user requests and provide structured JSON actions.",
            "Always include a 'response' action with a helpful message.",
            "For product-related actions (search, gift-recommendation, add-to-cart, compare), identify relevant product IDs from the catalog and include them in the 'product_ids' array.",
            "Do NOT return full product details; only return product IDs (alphanumeric strings). The frontend will fetch details.",
            "If the user asks for products by name, description, or ID, search the catalog and return matching product IDs.",
            "If the user asks for products within a price range (e.g., 'under $100', 'over 50 EUR'), identify the currency and range, then search the catalog for matching product IDs.",
            "If the user asks for products by category (e.g., 'electronics', 'clothing'), search the catalog and return matching product IDs.",
            "For 'gift-recommendation', if any 'person_details' (gender, age, preferences) are missing, generate a 'response' action asking specifically for the next missing detail. Do NOT attempt to recommend without sufficient details. Once all details are present, find the best suited products from the database using similarity search and return their IDs.",
            "If the user asks to 'add all' or 'add selected' to cart, use the 'latest_selected_product_ids' from the context.",
            "If the user asks to 'empty cart', use the 'empty-cart' action.",
            "Use conversation history and user context to provide better responses and maintain flow.",
            "Available actions (return ONLY these JSON structures):",
            "- { \"task\": \"response\", \"message\": \"<text>\" }",
            "- { \"task\": \"add-to-cart\", \"product_ids\": [\"<id1>\", \"<id2>\"], \"quantity\": <number> }",
            "- { \"task\": \"empty-cart\", \"message\": \"<text>\" }",
            "- { \"task\": \"view-cart\" }",
            "- { \"task\": \"search\", \"query\": \"<text>\", \"product_ids\": [\"<id1>\", \"<id2>\"] }",
            "- { \"task\": \"recommend\" }", # AI will not provide product_ids for this, Go backend will call gRPC
            "- { \"task\": \"gift-recommendation\", \"person_details\": { \"gender\": \"<text>\", \"age\": <num>, \"preferences\": \"<text>\" }, \"product_ids\": [\"<id1>\", \"<id2>\"] }",
            "- { \"task\": \"compare\", \"product_ids\": [\"<id1>\", \"<id2>\"] }",
            "- { \"task\": \"checkout\" }",
            "- { \"task\": \"update-context\", \"context\": { \"key\": \"value\" } }",
            f"\nUser Message: \"{user_message}\"",
            f"\nConversation History: {json.dumps(conversation_history)}",
            f"\nUser Context: {json.dumps(user_context)}",
            f"\nRoom Description (if applicable): {room_description}",
            "\nOutput ONLY a JSON array of actions. Do NOT include any other text."
        ]

        full_prompt = "\n".join(prompt_parts)
        app.logger.info(f"Sending prompt to Gemini: {full_prompt}")

        gemini_response = llm_text.invoke([HumanMessage(content=full_prompt)])
        ai_output = gemini_response.content.strip()
        app.logger.info(f"Raw Gemini response: {ai_output}")

        # Attempt to parse the AI's response
        parsed_actions = []
        try:
            parsed_actions = json.loads(ai_output)
            if not isinstance(parsed_actions, list):
                raise ValueError("AI response is not a JSON array.")
        except json.JSONDecodeError as e:
            app.logger.error(f"Failed to parse AI response JSON: {e}. Raw response: {ai_output}")
            actions.append({"task": "response", "message": "I apologize, I had trouble understanding that. Could you please rephrase?"})
        except ValueError as e:
            app.logger.error(f"Invalid AI response format: {e}. Raw response: {ai_output}")
            actions.append({"task": "response", "message": "I received an unexpected response format. Please try again."})

        # --- Process each action to perform database search if needed ---
        for action in parsed_actions:
            task = action.get("task")
            
            if vectorstore: # Ensure vectorstore is initialized
                if task == "search":
                    query = action.get("query", "")
                    app.logger.info(f"Performing similarity search for query: {query}")
                    
                    price_min = None
                    price_max = None
                    currency_code = "USD" # Default currency for price filtering

                    # Extract price range from query
                    price_match = re.search(r'(under|over)\s*(\d+(\.\d+)?)\s*(USD|EUR|JPY|GBP|TRY|CAD)?', query, re.IGNORECASE)
                    if price_match:
                        amount = float(price_match.group(2))
                        if price_match.group(4): # Check if currency code was captured
                            currency_code = price_match.group(4).upper()
                        if price_match.group(1).lower() == 'under':
                            price_max = amount
                        elif price_match.group(1).lower() == 'over':
                            price_min = amount

                    search_results = vectorstore.similarity_search(query, k=10) # Get top 10 similar products
                    
                    filtered_product_ids = []
                    for doc in search_results:
                        # Access price from metadata
                        product_price_units = doc.metadata.get('price_usd_units', 0)
                        product_price_nanos = doc.metadata.get('price_usd_nanos', 0)
                        product_price_code = doc.metadata.get('price_usd_currency_code', 'USD')
                        
                        current_product_price = product_price_units + product_price_nanos / 1e9

                        # Apply price filtering if specified and currency matches (or is default USD)
                        if (price_min is None or current_product_price >= price_min) and \
                           (price_max is None or current_product_price <= price_max) and \
                           (product_price_code == currency_code): # Only filter if currency matches
                            filtered_product_ids.append(doc.metadata['id'])
                        elif price_min is None and price_max is None: # If no price filter, include all
                            filtered_product_ids.append(doc.metadata['id'])

                    action["product_ids"] = filtered_product_ids
                    if not action.get("message"):
                        if filtered_product_ids:
                            action["message"] = f"I found some products related to '{query}'. Here are their IDs: {', '.join(filtered_product_ids)}"
                        else:
                            action["message"] = f"I couldn't find any products related to '{query}'."

                elif task == "gift-recommendation":
                    person_details = action.get("person_details", {})
                    gender = person_details.get("gender")
                    age = person_details.get("age")
                    preferences = person_details.get("preferences")

                    if gender and age and preferences:
                        app.logger.info(f"Generating gift recommendations for: {person_details}")
                        gift_query = f"gift for {age} year old {gender} with preferences for {preferences}"
                        gift_results = vectorstore.similarity_search(gift_query, k=5)
                        gift_product_ids = [doc.metadata['id'] for doc in gift_results]
                        action["product_ids"] = gift_product_ids
                        if not action.get("message"):
                            if gift_product_ids:
                                action["message"] = f"Here are some gift recommendations for a {age} year old {gender} with preferences for {preferences}: {', '.join(gift_product_ids)}"
                            else:
                                action["message"] = "I couldn't find specific gift recommendations based on the details provided."
                    else:
                        # If details are missing, remove product_ids and set a response message
                        action["product_ids"] = [] # Ensure no product_ids are sent if details are incomplete
                        if not gender:
                            action["message"] = "To recommend a gift, I need to know the recipient's gender. What is their gender?"
                        elif not age:
                            action["message"] = "What is their age?"
                        elif not preferences:
                            action["message"] = "What are their preferences (e.g., hobbies, interests)?"
                        action["task"] = "response" # Change task to response to just display message

                elif task == "recommend":
                    # AI will not provide product_ids for this. Go backend will call gRPC.
                    # Ensure no product_ids are set by AI for this task.
                    action["product_ids"] = []
                    if not action.get("message"):
                        action["message"] = "I can recommend some popular products for you."

                elif task == "compare" and action.get("product_ids"):
                    # AI should provide product_ids for comparison.
                    # No additional search needed here, just ensure the IDs are passed through.
                    app.logger.info(f"Comparing products with IDs: {action['product_ids']}")
                    if not action.get("message"):
                        action["message"] = f"Here are the products you asked to compare: {', '.join(action['product_ids'])}"
                
                elif task == "empty-cart": # Handle empty-cart action
                    action["message"] = action.get("message", "Your cart has been emptied.")
                    action["product_ids"] = [] # No product IDs for empty cart

            actions.append(action) # Add the processed action to the list

    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")
        actions.append({"task": "response", "message": "An internal error occurred. Please try again later."})

    return jsonify({"actions": actions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
