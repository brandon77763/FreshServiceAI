import base64
import requests
from openai import OpenAI
import time
import subprocess
import logging
import urllib3
import re
from bs4 import BeautifulSoup

# Suppress the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Freshservice API Details
api_key = "sdfsfsdfsdfsfsfsdf"
encoded_api_key = base64.b64encode(f"{api_key}:".encode()).decode()

headers = {
    'Authorization': f'Basic {encoded_api_key}',
    'Content-Type': 'application/json'
}

FRESHSERVICE_DOMAIN = 'domain.freshservice.com'
API_URL = f'https://{FRESHSERVICE_DOMAIN}/api/v2/tickets'
SOLUTIONS_API_URL = f'https://{FRESHSERVICE_DOMAIN}/api/v2/solutions/articles'

# Temporary RAG Cache
rag_cache = {}

# Setup for OpenAI Studio
client = OpenAI(base_url="http://192.168.1.1:1234/v1", api_key="lm-studio")



def clean_ticket_description(description_html):
    """
    Clean the ticket description by removing HTML tags, email signatures, and unnecessary data.
    """
    # Use BeautifulSoup to clean up HTML content
    soup = BeautifulSoup(description_html, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    
    # Remove common email signatures or system-generated lines
    cleaned_text = re.sub(r'Sent using.*', '', cleaned_text, flags=re.I)
    cleaned_text = re.sub(r'Zoho Mail', '', cleaned_text, flags=re.I)
    
    # Remove extra line breaks and trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def clean_conversation(conversation):
    """
    Clean the conversation text, focusing on the relevant issue.
    """
    # Strip out system responses or signatures using regex
    cleaned_conversation = re.sub(r'Sent using.*', '', conversation, flags=re.I)
    cleaned_conversation = re.sub(r'Zoho Mail', '', cleaned_conversation, flags=re.I)
    
    # Remove extra line breaks and trim
    cleaned_conversation = re.sub(r'\s+', ' ', cleaned_conversation).strip()
    
    return cleaned_conversation

def prepare_data_for_ai(ticket):
    """
    Prepare the cleaned ticket data and structure it for AI processing.
    """
    # Clean the ticket description
    cleaned_description = clean_ticket_description(ticket['description'])
    
    # Fetch and clean the conversation
    conversation = fetch_ticket_conversation(ticket['id'])
    cleaned_conversation = clean_conversation(conversation)
    
    # Create a structured format for AI processing
    structured_data = (
        f"Ticket Subject: {ticket['subject']}\n\n"
        f"Issue Description: {cleaned_description}\n\n"
        f"Conversation:\n{cleaned_conversation}\n\n"
        f"Resolution Needed: [Insert resolution here or pending resolution]"
    )
    
    return structured_data


# Function to fetch all tickets and filter closed ones locally
def fetch_closed_tickets(retries=3, backoff_factor=2):
    attempt = 0
    while attempt < retries:
        try:
            params = {
                'include': 'tags'  # Request ticket tags
            }
            response = requests.get(API_URL, headers=headers, params=params, verify=False)
            response.raise_for_status()
            tickets = response.json().get('tickets', [])
            logger.info(f"Fetched {len(tickets)} tickets.")
            
            closed_tickets = []
            for ticket in tickets:
                ticket_id = ticket.get('id')
                ticket_tags = ticket.get('tags', [])
                logger.info(f"Ticket ID: {ticket_id}, Tags: {ticket_tags}")
                
                if ticket.get('status') == 5 and "AI Scanned" not in ticket_tags:
                    closed_tickets.append(ticket)

            logger.info(f"Filtered {len(closed_tickets)} closed tickets without 'AI Scanned' tag.")
            return closed_tickets
        except requests.RequestException as e:
            if response.status_code == 429:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching tickets: {e}")
                return []
        attempt += 1
    return []



# Function to fetch the entire conversation from a ticket
def fetch_ticket_conversation(ticket_id):
    try:
        conversations_url = f'{API_URL}/{ticket_id}/conversations'
        response = requests.get(conversations_url, headers=headers, verify=False)
        response.raise_for_status()
        conversations = response.json().get('conversations', [])
        full_conversation = "\n".join([convo.get('body_text', '') for convo in conversations])
        logger.info(f"Fetched full conversation for ticket ID {ticket_id}.")
        return full_conversation
    except requests.RequestException as e:
        logger.error(f"Error fetching conversation for ticket ID {ticket_id}: {e}")
        return ""

# Function to check if the ticket contains enough data for a resolution
def check_data_sufficiency(conversation, retries=3, backoff_factor=2):
    attempt = 0
    while attempt < retries:
        try:
            prompt = (
                f"Does the following conversation contain enough information to form a proper resolution?\n"
                f"Conversation: {conversation}\n"
                f"Respond with 'Sufficient data' or 'Insufficient data'."
            )

            completion = client.chat.completions.create(
                model="model-identifier",  # Replace with your model identifier
                messages=[{"role": "system", "content": prompt}],
                temperature=0.5,
                max_tokens=50,
            )

            response = completion.choices[0].message.content.strip().lower()
            logger.info(f"Data sufficiency check response: {response}")

            if "sufficient" in response:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code == 429:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Rate limit hit on OpenAI, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error checking data sufficiency with AI: {e}")
                return False
        attempt += 1
    return False


def check_resolution_with_rag(conversation):
    """
    Check if the resolution exists in the GraphRAG knowledge base.
    This is a placeholder for the actual GraphRAG check logic.
    """
    # Add your logic to interact with GraphRAG
    # For now, we will simulate a GraphRAG lookup
    logger.info(f"Checking resolution in GraphRAG for conversation: {conversation}")
    
    # Simulate a RAG resolution check (replace with real logic)
    rag_resolved = False  # Simulated result

    if rag_resolved:
        logger.info("Resolution found in GraphRAG.")
        return True
    else:
        logger.info("Resolution not found in GraphRAG.")
        return False



# Function to check if the resolution exists using AI
def check_resolution_with_ai(conversation):
    """
    Check if the resolution exists in the AI model.
    """
    prompt = (
        f"Check if the following issue has a resolution in our knowledge base or AI model:\n"
        f"Conversation: {conversation}\n"
        f"Only respond with 'Resolution exists' if this exact resolution is known. Otherwise, respond with 'Resolution not found'."
    )

    try:
        completion = client.chat.completions.create(
            model="model-identifier",  # Replace with your model identifier
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=50,
        )

        # Correctly access the generated response text
        response = completion.choices[0].message.content.strip().lower()  # Convert to lowercase for matching
        logger.info(f"AI response for conversation: {response}")

        if "resolution exists" in response:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error checking resolution with AI: {e}")
        return False



def check_resolution_with_ai_and_rag(conversation):
    """
    First check if the resolution exists in GraphRAG, then check with AI if not found.
    """
    # Step 1: Check in GraphRAG first
    if check_resolution_with_rag(conversation):
        logger.info("Resolution found in GraphRAG.")
        return True  # Resolution found in GraphRAG

    # Step 2: Fall back to checking with AI
    if check_resolution_with_ai(conversation):
        logger.info("Resolution found with AI.")
        return True  # Resolution found with AI

    logger.info("No resolution found in GraphRAG or AI.")
    return False  # No resolution found



def create_draft_article_with_solution(ticket, conversation):
    """
    Create a draft article with a solution generated from the conversation.
    """
    # Generate the solution using AI
    generated_solution = generate_solution_from_conversation(ticket, conversation)
    
    # Create the article template with the generated solution
    article_template = (
        f"**Ticket Subject:** {ticket['subject']}\n\n"
        f"**Ticket ID:** {ticket['id']}\n\n"
        f"**Issue Description:**\n\n"
        f"{ticket.get('description', 'No description provided')}\n\n"
        f"**Conversation History:**\n\n"
        f"{conversation}\n\n"
        f"**Resolution:**\n\n"
        f"{generated_solution}"
    )
    
    article_data = {
        "title": f"Resolution for Ticket #{ticket['id']} - {ticket['subject']}",
        "description": article_template,
        "status": 1,  # 1 indicates draft status
        "folder_id": 32000001549  # Replace with your actual folder ID
    }

    try:
        response = requests.post(SOLUTIONS_API_URL, headers=headers, json=article_data, verify=False)
        response.raise_for_status()
        logger.info(f"Draft article created for ticket ID {ticket['id']}.")
        return response.json().get('article', {}).get('id')
    except requests.RequestException as e:
        logger.error(f"Error creating draft article for ticket ID {ticket['id']}: {e}")
        return None


def generate_solution_from_conversation(ticket, conversation):
    """
    Use AI to generate a solution based on the conversation.
    """
    prompt = (
        f"Based on the following conversation, generate a solution to the issue described:\n\n"
        f"Ticket Subject: {ticket['subject']}\n"
        f"Conversation: {conversation}\n\n"
        f"Provide the solution in detail."
    )
    
    try:
        completion = client.chat.completions.create(
            model="model-identifier",  # Replace with your model identifier
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,  # Adjust temperature if necessary for more detailed responses
            max_tokens=200,
        )
        
        # Extract the AI's response
        solution = completion.choices[0].message.content.strip()
        logger.info(f"Generated solution: {solution}")
        return solution
        
    except Exception as e:
        logger.error(f"Error generating solution from conversation: {e}")
        return "Pending resolution or no resolution provided."


def run_quality_check_before_creation(ticket, conversation):
    """
    Run the quality check before creating the draft article.
    """
    # Defining the content that would be used in the draft article
    article_template = (
        f"**Ticket Subject:** {ticket['subject']}\n\n"
        f"**Ticket ID:** {ticket['id']}\n\n"
        f"**Issue Description:**\n\n"
        f"{ticket.get('description', 'No description provided')}\n\n"
        f"**Conversation History:**\n\n"
        f"{conversation}\n\n"
        f"**Resolution:**\n\n"
        "Pending resolution or no resolution provided."
    )
    
    prompt = (
        f"Perform a quality check on the following article content:\n"
        f"Title: Resolution for Ticket #{ticket['id']} - {ticket['subject']}\n"
        f"Content: {article_template}\n"
        f"Respond with any improvements needed or 'No issues found'."
    )
    
    try:
        completion = client.chat.completions.create(
            model="model-identifier",  # Replace with your model identifier
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        # Access the response content properly
        response_text = completion.choices[0].message.content.strip()
        logger.info(f"Quality check response: {response_text}")
        
        # Check if the phrase "No issues found" is present in the response
        if "no issues found" in response_text.lower():
            logger.info("No quality issues found.")
            return True  # Quality check passed, proceed with article creation
        else:
            logger.warning(f"Quality issues found: {response_text}")
            return False  # Quality check failed, do not proceed with article creation
        
    except Exception as e:
        logger.error(f"Error during quality check: {e}")
        return False



# Function to update the ticket with the "AI Scanned" tag
def tag_ticket_as_scanned(ticket_id):
    try:
        update_data = {"tags": ["AI Scanned"]}
        response = requests.put(f'{API_URL}/{ticket_id}', headers=headers, json=update_data, verify=False)
        response.raise_for_status()
        logger.info(f"Ticket ID {ticket_id} tagged as 'AI Scanned'.")
    except requests.RequestException as e:
        logger.error(f"Error tagging ticket ID {ticket_id} as 'AI Scanned': {e}")

# Main function to process closed tickets
def process_closed_tickets():
    closed_tickets = fetch_closed_tickets()

    for ticket in closed_tickets:
        ticket_id = ticket['id']

        # Double-check if the ticket has the "AI Scanned" tag before proceeding
        if "AI Scanned" in ticket.get('tags', []):
            logger.info(f"Skipping ticket ID {ticket_id} as it is already marked as 'AI Scanned'.")
            continue

        # Fetch the entire conversation
        conversation = fetch_ticket_conversation(ticket_id)

        # Clean the ticket description and conversation
        cleaned_description = clean_ticket_description(ticket['description'])
        cleaned_conversation = clean_conversation(conversation)

        # Step 1: Check if the ticket contains sufficient data (with cleaned conversation)
        if not check_data_sufficiency(cleaned_conversation):
            logger.info(f"Insufficient data for ticket ID {ticket_id}. Marking as 'AI Scanned'.")
            tag_ticket_as_scanned(ticket_id)
            continue

        # Step 2: Check if the resolution exists in GraphRAG or AI (with cleaned conversation)
        if check_resolution_with_ai_and_rag(cleaned_conversation):
            logger.info(f"Resolution already exists for ticket ID {ticket_id}. Marking as 'AI Scanned'.")
            tag_ticket_as_scanned(ticket_id)
            continue

        # Step 3: Create a draft article with the AI-generated solution
        article_id = create_draft_article_with_solution(ticket, cleaned_conversation)
        
        if article_id:
            logger.info(f"Draft article created successfully for ticket ID {ticket_id}.")
        
        # Tag the ticket as "AI Scanned"
        tag_ticket_as_scanned(ticket_id)

        # Small delay to avoid hitting API rate limits
        time.sleep(2)




if __name__ == '__main__':
    process_closed_tickets()
