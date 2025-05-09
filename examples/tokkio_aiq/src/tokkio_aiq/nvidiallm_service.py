import logging
from pydantic import Field
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from openai import OpenAI
import asyncio
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class NVIDIALLMConfig(FunctionBaseConfig, name="nvidiallm_service"):
    """Configuration for NVIDIA LLM service."""

    api_key: str = Field(description="NVIDIA API key")
    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Base URL for the NVIDIA API endpoint"
    )
    model: str = Field(
        default="meta/llama-3.3-70b-instruct",
        description="Model to use for completion"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for response generation"
    )
    top_p: float = Field(
        default=0.7,
        description="Top-p sampling parameter"
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in response"
    )
    timeout: int = Field(
        default=60,
        description="Timeout for API requests in seconds"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to enable verbose logging"
    )


@register_function(config_type=NVIDIALLMConfig)
async def nvidiallm_function(tool_config: NVIDIALLMConfig, builder: Builder):
    """Function to query NVIDIA LLM service using OpenAI client."""
    
    if not tool_config.api_key:
        raise ValueError("API key is required for NVIDIA LLM service")

    client = OpenAI(
        base_url=tool_config.base_url,
        api_key=tool_config.api_key
    )

    async def runnable(query: str) -> str:
        try:
            if tool_config.verbose:
                logger.debug("Sending request to NVIDIA LLM endpoint")
                logger.debug("Query: %s", query)

            # Create the completion with streaming
            completion = client.chat.completions.create(
                model=tool_config.model,
                messages=[{"role": "user", "content": query}],
                temperature=tool_config.temperature,
                top_p=tool_config.top_p,
                max_tokens=tool_config.max_tokens,
                stream=True
            )

            # Collect the streamed response
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    if tool_config.verbose:
                        logger.debug("Received chunk: %s", content)

            if tool_config.verbose:
                logger.debug("Final response: %s", full_response)

            return full_response

        except Exception as e:
            error_msg = f"Error occurred while querying NVIDIA LLM: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    yield FunctionInfo.from_fn(
        runnable,
        description="This is a LLM tool for general queries. It uses the NVIDIA LLM service to get responses based on the provided query. Prior to using this tool over wikipedia unless the query asks for wikipedia."
    ) 