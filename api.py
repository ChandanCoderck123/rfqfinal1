import requests

def use_grok_api(query):
    """
    Function to send a query to the Grok API and retrieve a response.
    
    Parameters:
    - query (str): The input text to send to the Grok API.
    
    Returns:
    - dict: The response JSON from the API.
    """
    api_url = "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"  # Replace with actual Grok API URL
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",  # Replace with actual API Key
        "Content-Type": "application/json"
    }
    payload = {
        "query": query
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Example usage:
response = use_grok_api("What is the latest news in AI?")
print(response)
