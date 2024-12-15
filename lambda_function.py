import json
import boto3
import os
import requests
import re
from botocore.exceptions import ClientError
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOpenAI

# Load environment variables
openAi_api_key = os.environ.get('OPENAI_API_KEY')
google_api_key = os.environ.get('GOOGLE_API_KEY')

# Initialize services
llm = ChatOpenAI(model='gpt-4o-2024-08-06', temperature=0, openai_api_key=openAi_api_key)
lambda_client = boto3.client('lambda')
dynamodb = boto3.client('dynamodb')

# Amazon Knowledge Bases Retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="9UVBDVWHHZ",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
)

# Contextualize query prompt
contextualize_query_system_prompt = """ 
Rephrase the user’s query to make it clear and standalone, without chat history context. Do not answer, just rewrite.
History: {chat_history}
Recent: {recent_interactions}
"""
contextualize_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_query_system_prompt),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_query_prompt)

# System prompt adjustments:
# - Explicitly tell the model not to start every response with a greeting or the user's name.
# - Encourage natural, human-like conversation.
system_prompt = """
You are Beirut AI, a knowledgeable and friendly restaurant recommender in Lebanon.

Be personable, conversational, and natural. You may occasionally use the user's name ({username}), but:
- Do NOT start every response with a greeting or the user's name.
- Avoid "Hey {username}!" or similar repetitive salutations at the beginning of responses.
- If you greet, do so sparingly and naturally.

When the user asks about dining in an area, suggest a restaurant. Do not mention the plates unless explicitly asked. Avoid repeating the area or restaurant name unnecessarily.

If you cannot find a suitable restaurant, be honest, ask for clarification, or encourage the user to consider different options. Keep responses warm and human. Use minimal, relevant emojis if you like.

At the end of every answer, always include a line in the format:
"Recommended Restaurant: <name>"

Context: {context}
Preferences: {user_preferences}
History: {chat_history}
Recent: {recent_interactions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def update_user_preferences(session_id, preference_key, preference_value, append=False):
    try:
        if append:
            dynamodb.update_item(
                TableName='Beirut_chat_history',
                Key={'SessionId': {'S': session_id}},
                UpdateExpression="ADD Preferences.#key :value",
                ExpressionAttributeNames={"#key": preference_key},
                ExpressionAttributeValues={":value": {"SS": [preference_value]}},
            )
        else:
            dynamodb.update_item(
                TableName='Beirut_chat_history',
                Key={'SessionId': {'S': session_id}},
                UpdateExpression="SET Preferences.#key = :value",
                ExpressionAttributeNames={"#key": preference_key},
                ExpressionAttributeValues={":value": {"S": preference_value}},
            )
    except ClientError as e:
        print(f"Failed to update user preferences: {e}")


def detect_preferences(message):
    preferences = {}
    if "I love" in message:
        cuisine = message.split("I love")[1].strip()
        preferences['Cuisine'] = cuisine
    if "budget" in message:
        price_range = "low" if "cheap" in message else "high" if "expensive" in message else "medium"
        preferences['PriceRange'] = price_range
    return preferences


def get_user_preferences(session_id):
    key = {'SessionId': {'S': session_id}}
    response = dynamodb.get_item(TableName='Beirut_chat_history', Key=key)
    preferences = {}
    if 'Item' in response:
        prefs = response['Item'].get('Preferences', {}).get('M', {})
        for k, v in prefs.items():
            if 'S' in v:
                preferences[k] = v['S']
            elif 'SS' in v:
                preferences[k] = v['SS']
    return preferences


def get_place_details_by_name(place_name):
    if not google_api_key:
        print("No GOOGLE_API_KEY found in environment. Skipping place details.")
        return None

    print(f"Fetching place details for: {place_name}")

    base_url = "https://maps.googleapis.com/maps/api/place"
    text_search_params = {
        "query": f"{place_name} Lebanon",
        "key": google_api_key,
    }
    text_search_url = f"{base_url}/textsearch/json"
    try:
        r = requests.get(text_search_url, params=text_search_params)
        data = r.json()
        print("Text Search Response:", json.dumps(data))

        if data.get("status") != "OK":
            print(f"Text search request failed. Status: {data.get('status')}, Error: {data.get('error_message')}")
            return None

        results = data.get("results", [])
        if not results:
            print("No results found for that place name in text search.")
            return None

        place_id = results[0].get("place_id")
        if not place_id:
            print("No place_id found in text search results.")
            return None

        details_params = {
            "place_id": place_id,
            "fields": "name,formatted_address,rating",
            "key": google_api_key,
        }

        details_url = f"{base_url}/details/json"
        r_details = requests.get(details_url, params=details_params)
        details_data = r_details.json()
        print("Place Details Response:", json.dumps(details_data))

        if details_data.get("status") != "OK":
            print(f"Place details request failed. Status: {details_data.get('status')}, Error: {details_data.get('error_message')}")
            return None

        result = details_data.get("result", {})
        place_info = {
            "name": result.get("name"),
            "address": result.get("formatted_address"),
            "rating": result.get("rating")
        }

        return place_info

    except requests.RequestException as re:
        print(f"HTTP request exception occurred: {re}")
        return None
    except Exception as e:
        print(f"Unexpected exception while fetching place details: {e}")
        return None


def lambda_handler(event, context):
    print("Event received:", json.dumps(event))

    try:
        connection_id = event['requestContext']['connectionId']
        body = json.loads(event['body'])
        query = body.get('query', '')
        session_id = body.get('sessionId', '')
        username = body.get('username', 'Adnan')

        key = {'SessionId': {'S': session_id}}
        chatSession = dynamodb.get_item(TableName='Beirut_chat_history', Key=key)

        if 'Item' in chatSession:
            item = chatSession['Item']
            count = int(item['Count']['N'])
            thisMessageNumber = count + 1
            history = item.get('History', {}).get('S', "")
            messages = item.get('Messages', {}).get('M', {})
        else:
            thisMessageNumber = 1
            history = ""
            messages = {}

        recentInteractions = ""
        if thisMessageNumber > 1:
            for i in range(max(1, thisMessageNumber - 2), thisMessageNumber):
                if str(i) in messages:
                    msg = messages[str(i)]['M']
                    recentInteractions += f"Human: {msg['Human']['S']}\nAI: {msg['AI']['S']}\n"

        # Initialize session if first message
        if thisMessageNumber == 1:
            # Neutral, natural opener without naming the user
            initial_greeting = "Hello! I can help you find a great place to eat in Lebanon. What are you looking for?"
            dynamodb.put_item(
                TableName='Beirut_chat_history',
                Item={
                    'SessionId': {'S': session_id},
                    'Count': {'N': str(1)},
                    'Messages': {
                        'M': {
                            str(1): {
                                'M': {
                                    'AI': {'S': initial_greeting},
                                    'Human': {'S': query},
                                }
                            }
                        }
                    },
                    'History': {'S': ""},
                    'Preferences': {'M': {}},
                }
            )
        else:
            recentInteractions += f"Human: {query}\n"

        # Detect preferences and update
        prefs_found = detect_preferences(query)
        if prefs_found:
            print(f"Detected user preferences: {prefs_found}")
            for key_pref, value_pref in prefs_found.items():
                update_user_preferences(session_id, key_pref, value_pref)

        user_prefs = get_user_preferences(session_id)
        user_prefs_text = ""
        if user_prefs:
            pref_list = [f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in user_prefs.items()]
            user_prefs_text = "Preferences: " + ", ".join(pref_list) + "."

        # Generate response from RAG chain
        print("Generating response from RAG chain...")
        response = ""
        for chunk in rag_chain.stream({
            "chat_history": history,
            "input": query,
            "recent_interactions": recentInteractions,
            "user_preferences": user_prefs_text,
            "username": username,
        }):
            if 'answer' in chunk:
                answer_chunk = chunk['answer']
                response += answer_chunk

        # Extract recommended restaurant
        recommended_restaurant = None
        match = re.search(r"Recommended Restaurant:\s*(.*)", response)
        if match:
            recommended_restaurant = match.group(1).strip()
            response = re.sub(r"Recommended Restaurant:\s*.*", "", response).strip()

        # Decide whether to return restaurant details
        # Only return details if user asked about exact location, rating, or reviews.
        details_requested = any(
            keyword in query.lower() 
            for keyword in ["location", "address", "review", "reviews", "rating", "exact place"]
        )

        final_lines = []
        if response:
            final_lines.append(response.strip())

        if recommended_restaurant and recommended_restaurant.lower() not in ["none", ""]:
            if details_requested:
                place_details = get_place_details_by_name(recommended_restaurant)
                if place_details:
                    details_text = f"{place_details['name']} is located at {place_details['address']}."
                    if place_details.get("rating"):
                        details_text += f" It has a rating of {place_details['rating']}."
                    final_lines.append(details_text.strip())
                else:
                    final_lines.append(f"I couldn't find details on {recommended_restaurant}. Maybe consider another option or share more preferences?")
        else:
            if not recommended_restaurant or recommended_restaurant.lower() in ["none", ""]:
                if not response:
                    final_lines.append("I'm not sure what fits best at the moment. Could you tell me a bit more about what you prefer?")

        # Join all lines into a final structured answer
        answer = "\n\n".join(line for line in final_lines if line).strip()

        history += f"Human: {query}\nAI: {answer}\n"

        # Update DynamoDB
        print("Updating DynamoDB with the new message and history.")
        dynamodb.update_item(
            TableName='Beirut_chat_history',
            Key=key,
            UpdateExpression="SET #count = :new_count, Messages.#thisMsgNum = :new_message, History = :history",
            ExpressionAttributeNames={
                '#count': 'Count',
                '#thisMsgNum': str(thisMessageNumber),
            },
            ExpressionAttributeValues={
                ':new_count': {'N': str(thisMessageNumber)},
                ':new_message': {
                    'M': {
                        'AI': {'S': answer},
                        'Human': {'S': query},
                    }
                },
                ':history': {'S': history},
            },
        )

        payload = {
            "sessionId": session_id,
            "messageNumber": thisMessageNumber,
            "history": history,
            "recentInteractions": recentInteractions,
            "query": query,
        }

        print("Invoking beirut_history Lambda function asynchronously.")
        lambda_client.invoke(
            FunctionName='beirut_history',
            InvocationType='Event',
            Payload=json.dumps(payload),
        )

        print("Returning successful response.")
        return {
            'statusCode': 200,
            'body': json.dumps({"response": answer}),
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)}),
        }
