# src/shoppingmate_ai_service/main.py
import os
import json
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import re
import logging

app = Flask(__name__)

# Configure logging for Flask app
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# --- Mock Database ---
MOCK_DATABASE = [
  {
    "id": "OLJCESPC7Z",
    "name": "Sunglasses",
    "price_usd_units": 19,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "Add a modern touch to your outfits with these sleek aviator sunglasses."
  },
  {
    "id": "66VCHSJNUP",
    "name": "Tank Top",
    "price_usd_units": 18,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "Perfectly cropped cotton tank, with a scooped neckline."
  },
  {
    "id": "1YMWWN1N4O",
    "name": "Watch",
    "price_usd_units": 109,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "This gold-tone stainless steel watch will work with most of your outfits."
  },
  {
    "id": "L9ECAV7KIM",
    "name": "Loafers",
    "price_usd_units": 89,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "A neat addition to your summer wardrobe."
  },
  {
    "id": "2ZYFJ3GM2N",
    "name": "Hairdryer",
    "price_usd_units": 24,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "This lightweight hairdryer has 3 heat and speed settings. It's perfect for travel."
  },
  {
    "id": "0PUK6V6EV0",
    "name": "Candle Holder",
    "price_usd_units": 18,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "This small but intricate candle holder is an excellent gift."
  },
  {
    "id": "LS4PSXUNUM",
    "name": "Salt & Pepper Shakers",
    "price_usd_units": 18,
    "price_usd_nanos": 490000000,
    "price_usd_currency_code": "USD",
    "description": "Add some flavor to your kitchen."
  },
  {
    "id": "9SIQT8TOJO",
    "name": "Bamboo Glass Jar",
    "price_usd_units": 5,
    "price_usd_nanos": 490000000,
    "price_usd_currency_code": "USD",
    "description": "This bamboo glass jar can hold 57 oz (1.7 l) and is perfect for any kitchen."
  },
  {
    "id": "6E92ZMYYFZ",
    "name": "Mug",
    "price_usd_units": 8,
    "price_usd_nanos": 990000000,
    "price_usd_currency_code": "USD",
    "description": "A simple mug with a mustard interior."
  }
]

# --- Gemini Models ---
# Assuming GOOGLE_API_KEY is set in the environment for these to work
GOOGLE_API_KEY = "AIzaSyCEMrsxVtii0n4eDKVhdx0j4VrZE1fRggY"
llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
llm_text = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# --- In-Memory Product Search Function ---
def search_products_in_mock_db(query_text=None, price_min=None, price_max=None, currency_code="USD", gender=None, age=None, preferences=None):
    found_product_ids = []
    query_text_lower = query_text.lower() if query_text else ""

    for product in MOCK_DATABASE:
        match_text = False
        if query_text:
            # Search by ID, name, description
            if query_text_lower in product.get("id", "").lower() or \
               query_text_lower in product.get("name", "").lower() or \
               query_text_lower in product.get("description", "").lower():
                match_text = True
        else:
            match_text = True # If no query text, all products initially match text criteria

        if not match_text:
            continue

        # Convert product price to float for comparison
        product_price_float = product["price_usd_units"] + product["price_usd_nanos"] / 1_000_000_000
        product_currency = product["price_usd_currency_code"]

        # Apply price filters
        match_price = True
        if price_min is not None:
            if product_price_float < price_min or product_currency != currency_code:
                match_price = False
        if price_max is not None:
            if product_price_float > price_max or product_currency != currency_code:
                match_price = False
        
        if not match_price:
            continue

        # Apply gift recommendation filters (simple keyword matching for mock)
        match_gift = True
        if gender:
            if gender.lower() not in product.get("description", "").lower() and \
               gender.lower() not in product.get("name", "").lower():
                match_gift = False
        if age: # Very basic age filtering, could be improved
            if (age < 18 and "adult" in product.get("description", "").lower()) or \
               (age >= 18 and "kid" in product.get("description", "").lower()):
                match_gift = False
        if preferences:
            if preferences.lower() not in product.get("description", "").lower() and \
               preferences.lower() not in product.get("name", "").lower():
                match_gift = False
        
        if match_text and match_price and match_gift:
            found_product_ids.append(product["id"])
            
    return found_product_ids

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
            try:
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
            except Exception as e:
                app.logger.error(f"Error processing image with Gemini Vision: {e}")
                actions.append({"task": "response", "message": "I had trouble processing the image. Please try again or describe the room."})
                # Continue without room_description if vision fails

        # --- Step 2: Determine user intent and generate actions ---
        prompt_parts = [
            "You are an AI shopping assistant for an e-commerce platform. Your goal is to understand user requests and provide structured JSON actions.",
            "You have access to a product catalog. For product-related queries, you will search this catalog directly and return product IDs.",
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

        # --- Process each action to perform in-memory search if needed ---
        for action in parsed_actions:
            task = action.get("task")
            
            if task == "search":
                query = action.get("query", "")
                app.logger.info(f"Performing in-memory search for query: {query}")
                
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

                # Use the in-memory search function
                found_product_ids = search_products_in_mock_db(
                    query_text=query, 
                    price_min=price_min, 
                    price_max=price_max, 
                    currency_code=currency_code
                )
                
                action["product_ids"] = found_product_ids
                if not action.get("message"):
                    if found_product_ids:
                        action["message"] = f"I found some products related to '{query}'. Here are their IDs: {', '.join(found_product_ids)}"
                    else:
                        action["message"] = f"I couldn't find any products related to '{query}'."

            elif task == "gift-recommendation":
                person_details = action.get("person_details", {})
                gender = person_details.get("gender")
                age = person_details.get("age")
                preferences = person_details.get("preferences")

                if gender and age and preferences:
                    app.logger.info(f"Generating gift recommendations for: {person_details}")
                    # Use the in-memory search function for gift recommendations
                    gift_product_ids = search_products_in_mock_db(
                        gender=gender, 
                        age=age, 
                        preferences=preferences
                    )
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
        app.logger.error(f"An unexpected error occurred during request processing: {e}")
        app.logger.error(f"Traceback: {e.__traceback__}") # Print full traceback for debugging
        actions.append({"task": "response", "message": "An internal error occurred. Please try again later."})

    return jsonify({"actions": actions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8070)
