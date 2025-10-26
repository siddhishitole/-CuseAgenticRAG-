from typing import Dict, Any
from ..tools.n8n_workflow import N8nWorkflow

class WorkflowAgent:
    def __init__(self):
        self.n8n_workflow = N8nWorkflow()

    def process(self, query: str) -> Dict[Any, Any]:
        """
        Process the user query through the n8n workflow
        
        Args:
            query (str): The user's prompt/query
            
        Returns:
            Dict[Any, Any]: The response from the n8n workflow
        """
        try:
            # Execute the workflow with the given query
            response = self.n8n_workflow.execute_workflow(query)
            return response
        except Exception as e:
            raise Exception(f"Error in workflow agent: {str(e)}")

    def __call__(self, query: str) -> Dict[Any, Any]:
        """
        Make the agent callable
        """
        return self.process(query)