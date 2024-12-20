# Beirut AI Restaurant Recommender Lambda Function

## Overview
This AWS Lambda function serves as the backend for a conversational restaurant recommendation service tailored for users in Lebanon. It integrates OpenAI's GPT-based language model and external services like Amazon Knowledge Bases and Google Places API to provide context-aware, personalized responses and restaurant suggestions. The function maintains a history of user interactions and preferences in Amazon DynamoDB for a contextual, personalized experience.

---

## Features
1. **Chat-based Recommendations**:
   - Offers restaurant recommendations in Lebanon based on user queries.
   - Context-aware responses generated using a Retrieval-Augmented Generation (RAG) chain.
   - Can provide additional details like location and ratings if requested.

2. **Preference Detection and Tracking**:
   - Dynamically detects user preferences (e.g., cuisine type, budget) from the conversation.
   - Stores and updates preferences in a DynamoDB table.

3. **Integration with APIs**:
   - **OpenAI API**: For natural language understanding and response generation.
   - **Amazon Knowledge Bases Retriever**: For fetching relevant information from predefined knowledge bases.
   - **Google Places API**: For retrieving restaurant details such as location and ratings.

4. **Session Management**:
   - Maintains session history and interactions in DynamoDB.
   - Generates responses considering the full chat history and recent interactions.

---

## Architecture
- **AWS Lambda**: Hosts the function and manages requests from connected users.
- **Amazon DynamoDB**: Stores chat history, session data, and user preferences.
- **APIs Used**:
  - OpenAI API for GPT-4-based conversational AI.
  - Google Places API for restaurant details.
  - Amazon Knowledge Bases Retriever for information retrieval.

---

## Environment Variables
The function relies on the following environment variables:
- `OPENAI_API_KEY`: API key for accessing OpenAI's GPT models.
- `GOOGLE_API_KEY`: API key for accessing the Google Places API.

---

## Sample Payload
### Input:
```json
{
  "query": "Can you recommend a Lebanese restaurant in Beirut?",
  "sessionId": "12345",
  "username": "Adnan"
}
```
### Output:
```json
{
  "response": "How about trying Pick a Poke in Chiyah? It's a fantastic place for poke bowls."
}
```

## Functions and Modules

### 1. `detect_preferences(message)`
- Extracts preferences like cuisine type and budget from the user's query.

### 2. `update_user_preferences(session_id, preference_key, preference_value, append)`
- Updates user preferences in the DynamoDB table.

### 3. `get_user_preferences(session_id)`
- Fetches stored preferences for a given session.

### 4. `get_place_details_by_name(place_name)`
- Retrieves restaurant details like location and ratings using the Google Places API.

### 5. `lambda_handler(event, context)`
- Entry point for the Lambda function, handling requests, updating chat history, and generating responses.

---

## API Integration

### Google Places API
- Used to fetch restaurant details (e.g., address, rating).
- Relevant endpoints: `/textsearch`, `/details`.

### OpenAI GPT-4
- Used for generating conversational responses and processing chat history.
- Requires a valid API key.

### Amazon Knowledge Bases
- Retrieves additional restaurant-related information.

---

## Error Handling
- Errors during API calls or DynamoDB updates are logged and gracefully handled.
- A default error message is returned to the user if the Lambda function encounters an exception.

---

## Future Enhancements
1. Add more advanced preference detection using machine learning or rule-based NLP.
2. Enhance session management to include expiration or cleanup of old data.
3. Integrate more APIs for broader restaurant coverage and reviews.

---

