import requests
from typing import Dict, Any

class N8nWorkflow:
    def __init__(self):
        self.webhook_url = "https://naru-07.app.n8n.cloud/webhook/8eb201fc-42bf-4a38-b259-60c661666c19"

    def execute_workflow(self, query: str) -> Dict[Any, Any]:
        """
        Execute the n8n workflow by sending the query to the webhook
        
        Args:
            query (str): The user's prompt/query
            
        Returns:
            Dict[Any, Any]: The response from the n8n workflow
        """
        try:
            # Construct the URL with the query parameter
            url = f"{self.webhook_url}?query={query}"
            
            # Send GET request to the webhook
            response = requests.get(url)
            response.raise_for_status()
            
            # Return the JSON response
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error executing n8n workflow: {str(e)}")