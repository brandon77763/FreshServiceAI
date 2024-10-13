import base64
import requests
import time
import subprocess
import logging
import re  # Import for HTML tag removal
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Freshservice API Details
api_key = "dssadasdadasdadasda"
encoded_api_key = base64.b64encode(f"{api_key}:".encode()).decode()

headers = {
    'Authorization': f'Basic {encoded_api_key}',
    'Content-Type': 'application/json'
}

FRESHSERVICE_DOMAIN = 'belsolutions.freshservice.com'
API_URL = f'https://{FRESHSERVICE_DOMAIN}/api/v2/tickets'

# Cache for RAG (Retrieval Augmented Generation)
rag_cache = {}

# OpenAI Client Setup (replace with actual values as necessary)
client = OpenAI(base_url="http://192.168.1.0:1234/v1", api_key="lm-studio")

# Function to remove HTML tags from the input
def remove_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    return clean_text

# Function to fetch open tickets
def fetch_open_tickets():
    try:
        response = requests.get(API_URL, headers=headers, verify=False)
        response.raise_for_status()
        tickets = response.json().get('tickets', [])
        open_tickets = [ticket for ticket in tickets if ticket.get('status') == 2]
        logger.info(f"Fetched {len(open_tickets)} open tickets.")
        return open_tickets
    except requests.RequestException as e:
        logger.error(f"Error fetching tickets: {e}")
        return []

# Function to retrieve the conversation history for a ticket
def get_conversation_history(ticket_id, max_messages=10):
    try:
        conversations_url = f'{API_URL}/{ticket_id}/conversations'
        response = requests.get(conversations_url, headers=headers, verify=False)
        response.raise_for_status()

        conversations = response.json().get('conversations', [])
        if not isinstance(conversations, list):
            logger.error(f"Unexpected data format for ticket ID {ticket_id}")
            return []

        # Sort conversations by timestamp in ascending order
        conversations.sort(key=lambda x: x.get('created_at', ''))

        # Build message history
        message_history = []

        for convo in conversations:
            body = remove_html_tags(convo.get('body_text', '').strip() or convo.get('body', '').strip())
            if not body:
                continue

            if convo.get('incoming'):
                # User message
                message_history.append({"role": "user", "content": body})
            else:
                # Agent reply
                message_history.append({"role": "assistant", "content": body})

        # Limit the message history to the last max_messages
        message_history = message_history[-max_messages:]

        return message_history
    except (requests.RequestException, ValueError) as e:
        logger.error(f"Error retrieving ticket data for ticket ID {ticket_id}: {e}")
        return []

# Function to get GraphRAG context with a longer timeout and improved logging
def get_graphrag_context(query):
    clean_query = remove_html_tags(query)  # Clean the query by removing any HTML tags

    if clean_query in rag_cache:
        logger.info(f"Returning cached context for query: {clean_query[:30]}...")
        return rag_cache[clean_query].get('graphrag_context', '')

    try:
        # Log the exact query being sent to GraphRAG
        logger.info(f"Querying GraphRAG with: {clean_query}")

        # Increase the timeout from 10 seconds to 780 seconds
        result = subprocess.run(
            ['python3', '-m', 'graphrag.query', '--root', '/home/brandon/graphrag', '--method', 'local', clean_query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=780  # Increased timeout to 780 seconds
        )

        # Check if the subprocess call was successful
        if result.returncode != 0:
            logger.error(f"GraphRAG query failed with error: {result.stderr}")
            return ""  # Return empty context if the query failed

        # Capture the context from the GraphRAG query output
        context = result.stdout.strip()

        # Log the response from GraphRAG
        logger.info(f"GraphRAG response: {context}")

        if context:
            # Cache the result if the context is not empty
            rag_cache[clean_query] = {'graphrag_context': context}
            logger.info(f"GraphRAG context retrieved successfully for query: {clean_query[:30]}...")
        else:
            # Log a warning if the context is empty
            logger.warning(f"GraphRAG returned no context for query: {clean_query[:30]}...")

        return context

    except subprocess.TimeoutExpired:
        logger.error(f"GraphRAG query timed out for query: {clean_query[:30]}...")
        return ""  # Return empty context in case of a timeout

    except subprocess.SubprocessError as e:
        logger.error(f"Error using GraphRAG for query: {clean_query[:30]}... Error: {e}")
        return ""  # Return empty context in case of a subprocess error

# Function to check response appropriateness using AI with detailed context and logging
def check_response_appropriateness(response):
    # Provide more context for the AI to evaluate the response
    evaluation_prompt = [
        {"role": "system", "content": "You are a quality control assistant for a technical support AI. Your job is to evaluate whether the response given below is appropriate, professional, and helpful for an IT support technician assisting a user. Please consider clarity, professionalism, tone, and relevance to the user's problem."},
        {"role": "user", "content": f"Response to evaluate:\n\n{response}\n\nIs this response appropriate for a professional IT support technician?"}
    ]

    try:
        # Call the AI model to evaluate the response
        completion = client.chat.completions.create(
            model="model-identifier",  # Replace with your model identifier for evaluation
            messages=evaluation_prompt,
            temperature=0.3  # A lower temperature to ensure more deterministic responses
        )

        # Correctly access the 'content' from the message
        evaluation_result = completion.choices[0].message.content.strip().lower()

        # Log the AI's evaluation result
        logger.info(f"AI evaluation result: {evaluation_result}")

        if 'yes' in evaluation_result:
            logger.info("Response deemed appropriate by AI.")
            return True
        else:
            logger.warning(f"Response deemed inappropriate by AI. Evaluation: {evaluation_result}")
            return False

    except Exception as e:
        logger.error(f"Error checking response appropriateness: {e}")
        return False

# Function to generate AI response based on the entire conversation history
def generate_response(ticket, context):
    conversation_history = get_conversation_history(ticket['id'])
    clean_ticket_description = remove_html_tags(ticket['description'])

    # Modify system prompt
    system_prompt = {"role": "system", "content": "You are a highly skilled IT support technician with extensive technical knowledge. You provide clear, concise, and professional responses to assist users in resolving their technical issues. If there is additional context provided, ensure to incorporate that into your response as if the user is not already aware of it."}

    # Build the messages
    messages = [system_prompt]
    # Include ticket subject and description as user message
    messages.append({"role": "user", "content": f"Ticket Subject: {ticket['subject']}\nTicket Description: {clean_ticket_description}"})

    # Include the conversation history
    messages.extend(conversation_history)

    # Add additional context from GraphRAG if available
    if context:
        logger.info(f"Adding GraphRAG context to the response for ticket ID {ticket['id']}.")
        messages.append({"role": "system", "content": f"This is additional information that should be included in your response: {context}"})
    else:
        logger.warning(f"No GraphRAG context available for ticket ID {ticket['id']}.")

    try:
        # Log the message history for debugging
        logger.info(f"Message history for ticket ID {ticket['id']}: {messages}")

        # Generate AI response with added context
        completion = client.chat.completions.create(
            model="model-identifier",  # Replace with your model identifier
            messages=messages,
            temperature=0.7,
            stream=False,  # Disable streaming to get the entire response at once
        )

        # Extract the response from the completion
        response = completion.choices[0].message.content.strip()

        # Log the generated response for debugging
        logger.info(f"Generated response: {response}")

        # Check if the response is not empty before passing to appropriateness check
        if response and check_response_appropriateness(response):
            logger.info(f"Generated appropriate response for ticket ID {ticket['id']}.")
            return response
        else:
            logger.warning(f"Response for ticket ID {ticket['id']} deemed inappropriate or empty. Escalating issue.")
            escalate_issue(ticket['id'])
            return None

    except Exception as e:
        logger.error(f"Error generating response for ticket ID {ticket['id']}: {e}")
        return ""

# Function to reply to a ticket and update its status
def reply_to_ticket(ticket_id, response_content):
    try:
        # Add a public note
        note_data = {'body': response_content, 'private': False}
        requests.post(f'{API_URL}/{ticket_id}/notes', headers=headers, json=note_data, verify=False).raise_for_status()

        # Update ticket status to "Pending"
        status_update = {'status': 3}
        requests.put(f'{API_URL}/{ticket_id}', headers=headers, json=status_update, verify=False).raise_for_status()

        logger.info(f"Successfully replied to ticket ID {ticket_id} and updated status to Pending.")
    except requests.RequestException as e:
        logger.error(f"Error replying to ticket ID {ticket_id}: {e}")

# Function to escalate the issue
def escalate_issue(ticket_id):
    logger.info(f"Escalating issue for ticket ID {ticket_id}")

    # Data to update the ticket: set priority to "High" and status to "Pending"
    update_data = {
        'priority': 3,  # 3 corresponds to "High" priority in Freshservice
        'status': 3     # 3 corresponds to "Pending" status in Freshservice
    }

    # Data for the public reply informing the user of the escalation
    reply_data = {
        'body': "We are escalating your issue to a higher priority. Our team will get back to you shortly.",
        'private': False  # Set to False to make the note public
    }

    try:
        # API request to update the ticket priority and status
        response = requests.put(f'{API_URL}/{ticket_id}', headers=headers, json=update_data, verify=False)
        response.raise_for_status()  # Raise an error for any unsuccessful request

        logger.info(f"Ticket ID {ticket_id} successfully escalated to High priority and marked as Pending.")

        # API request to add a public reply to the ticket
        reply_response = requests.post(f'{API_URL}/{ticket_id}/notes', headers=headers, json=reply_data, verify=False)
        reply_response.raise_for_status()  # Raise an error for any unsuccessful request

        logger.info(f"Added escalation reply to ticket ID {ticket_id}.")

    except requests.RequestException as e:
        logger.error(f"Failed to escalate ticket ID {ticket_id}: {e}")

# Main loop
def main_loop():
    while True:
        tickets = fetch_open_tickets()
        for ticket in tickets:
            ticket_id = ticket['id']
            if ticket_id not in rag_cache:
                rag_cache[ticket_id] = {'status': 'open', 'responses': []}

            graphrag_context = get_graphrag_context(ticket['description'])
            response = generate_response(ticket, graphrag_context)

            if response:
                reply_to_ticket(ticket_id, response)
                rag_cache[ticket_id]['status'] = 'pending'
                rag_cache[ticket_id]['responses'].append(response)

        logger.info("Sleeping for 60 seconds before the next check.")
        time.sleep(60)

if __name__ == '__main__':
    main_loop()
