import logging
import json
from typing import Optional, Dict, Any

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import Field
import httpx

logger = logging.getLogger(__name__)

class OpenAIRAGConfig(FunctionBaseConfig, name="openai_rag"):
    """Configuration for OpenAI RAG client."""

    api_key: str = Field(description="OpenAI API key")
    base_url: str = Field(description="Base URL for the OpenAI API endpoint")
    model: str = Field(default="gpt-3.5-turbo", description="Model to use for completion")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    timeout: int = Field(default=60, description="Timeout for API requests in seconds")

@register_function(config_type=OpenAIRAGConfig)
async def openai_rag_function(tool_config: OpenAIRAGConfig, builder: Builder):
    """Function to query OpenAI API with RAG capabilities."""
    
    async with httpx.AsyncClient(
        base_url=tool_config.base_url,
        headers={
            "Authorization": f"Bearer {tool_config.api_key}",
            "Content-Type": "application/json"
        },
        timeout=tool_config.timeout
    ) as client:
        
        async def runnable(query: str) -> str:
            try:
                payload = {
                    "model": tool_config.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    "temperature": tool_config.temperature,
                    "max_tokens": tool_config.max_tokens
                }

                logger.debug("Sending request to OpenAI endpoint")
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                
                # Log the raw response for debugging
                logger.debug("Raw response: %s", response.text)
                
                try:
                    output = response.json()
                    if not isinstance(output, dict):
                        logger.error("Response is not a dictionary: %s", output)
                        return "Error: Invalid response format from server"
                    
                    choices = output.get("choices", [])
                    if not choices:
                        logger.error("No choices in response: %s", output)
                        return "Error: No choices in server response"
                    
                    message = choices[0].get("message", {})
                    if not message:
                        logger.error("No message in first choice: %s", choices[0])
                        return "Error: No message in server response"
                    
                    content = message.get("content")
                    if not content:
                        logger.error("No content in message: %s", message)
                        return "Error: No content in server response"
                    
                    return content
                    
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON response: %s", e)
                    logger.error("Raw response content: %s", response.text)
                    return f"Error: Invalid JSON response from server: {str(e)}"
                    
            except httpx.HTTPStatusError as e:
                logger.error("HTTP error occurred: %s", e)
                return f"Error: HTTP {e.response.status_code} - {e.response.text}"
            except httpx.RequestError as e:
                logger.error("Request error occurred: %s", e)
                return f"Error: Failed to connect to server - {str(e)}"
            except Exception as e:
                logger.exception("Unexpected error while running the tool", exc_info=True)
                return f"Error: Unexpected error occurred - {str(e)}"

        yield FunctionInfo.from_fn(
            runnable,
            description="Query OpenAI API with RAG capabilities to get responses based on the provided query"
        ) 