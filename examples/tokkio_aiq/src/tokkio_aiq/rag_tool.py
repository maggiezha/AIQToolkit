# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#curl -X 'POST' \
#  'https://rag-server-x8vcu248o.brevlab.com/v1/generate' \
#  -H 'accept: application/json' \
#  -H 'Content-Type: application/json' \
#  -d '{
#  "messages": [
#    {
#      "role": "user",
#      "content": "Hello! What can you help me with?"
#    }
#  ],
#  "use_knowledge_base": true,
#  "temperature": 0.2,
#  "top_p": 0.7,
#  "max_tokens": 1024,
#  "reranker_top_k": 10,
#  "vdb_top_k": 50,
#  "vdb_endpoint": "http://milvus:19530",
#  "collection_name": "multimodal_data",
#  "enable_query_rewriting": false,
#  "enable_reranker": true,
#  "enable_guardrails": false,
#  "enable_citations": true,
#  "model": "meta/llama-3.1-8b-instruct",
#  "llm_endpoint": "nim-llm:8000",
#  "embedding_model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
#  "embedding_endpoint": "nemoretriever-embedding-ms:8000",
#  "reranker_model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
#  "reranker_endpoint": "nemoretriever-ranking-ms:8000",
#  "stop": []
#}'



import logging
import json
from typing import Optional, Dict, Any, List

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import Field 
import httpx

logger = logging.getLogger(__name__)



import json
import re


def parse_sse_data(sse_text: str) -> List[Dict[str, Any]]:
    """
    Parse Server-Sent Events (SSE) data from a string into a list of dictionaries.
    Also extracts and processes citations and results from the response.

    Args:
        sse_text (str): The SSE text containing 'data:' prefixed JSON strings

    Returns:
        list: A list of parsed dictionaries from the JSON data
    """
    # Find all data lines using regex
    data_lines = re.findall(r'data: (.*?)(?=\n\n|\Z)', sse_text, re.DOTALL)

    # Parse each JSON string into a dictionary
    result = []
    citations = []
    content_parts = []
    
    for data_line in data_lines:
        try:
            # Strip any leading/trailing whitespace and parse JSON
            json_data = json.loads(data_line.strip())
            result.append(json_data)
            
            # Extract content from choices if available
            if 'choices' in json_data and json_data['choices']:
                choice = json_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content_parts.append(choice['message']['content'])
            
            # Extract citations if available
            if 'citations' in json_data and 'results' in json_data['citations']:
                citations.extend(json_data['citations']['results'])
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"Problematic data line: {data_line.strip()}")
            continue

    # Combine all content parts
    full_content = ''.join(content_parts)
    
    # Create a structured response
    structured_response = {
        'content': full_content,
        'citations': citations,
        'raw_chunks': result
    }
    
    return structured_response


class RAGPostFnConfig(FunctionBaseConfig, name="rag_post"):
    """Function to make a post request to a remotely hosted RAG agent."""

    timeout: int = Field(default=60, description="Timeout for post request to remote RAG agent.")
    url: str = Field(description="The url to the hosted RAG agent.")
    collection_name: str = Field(description="The name of the collection to use for the RAG agent.")
    temperature: float = Field(default=0.2, description="Temperature for response generation.")
    top_p: float = Field(default=0.7, description="Top p for response generation.")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response.")
    reranker_top_k: int = Field(default=5, description="Number of top results from reranker.")
    vdb_top_k: int = Field(default=20, description="Number of top results from vector database.")
    enable_query_rewriting: bool = Field(default=False, description="Whether to enable query rewriting.")
    enable_reranker: bool = Field(default=True, description="Whether to enable reranking.")
    enable_guardrails: bool = Field(default=False, description="Whether to enable guardrails.")
    enable_citations: bool = Field(default=True, description="Whether to enable citations.")
    model: str = Field(default="meta/llama-3.1-8b-instruct", description="LLM model to use.")
    llm_endpoint: str = Field(default="nim-llm:8000", description="LLM endpoint.")
    embedding_model: str = Field(default="nvidia/llama-3.2-nv-embedqa-1b-v2", description="Embedding model to use.")
    embedding_endpoint: str = Field(default="nemoretriever-embedding-ms:8000", description="Embedding endpoint.")
    reranker_model: str = Field(default="nvidia/llama-3.2-nv-rerankqa-1b-v2", description="Reranker model to use.")
    reranker_endpoint: str = Field(default="nemoretriever-ranking-ms:8000", description="Reranker endpoint.")

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length while preserving sentence boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find the last sentence boundary before max_length
    last_period = text[:max_length].rfind('.')
    last_question = text[:max_length].rfind('?')
    last_exclamation = text[:max_length].rfind('!')
    
    # Find the last sentence boundary
    last_boundary = max(last_period, last_question, last_exclamation)
    
    if last_boundary == -1:
        # If no sentence boundary found, just truncate at max_length
        return text[:max_length] + "..."
    
    return text[:last_boundary + 1] + "..."

@register_function(config_type=RAGPostFnConfig)
async def rag_function(tool_config: RAGPostFnConfig, builder: Builder):
    import httpx
    import json

    async with httpx.AsyncClient(headers={
            "accept": "application/json",
            "Content-Type": "application/json"
    }, timeout=tool_config.timeout) as client:

        async def runnable(query: str) -> str:
            try:
                # Enhance query to encourage concise responses
                enhanced_query = f"Please provide a concise answer about animals and their Spanish names. Question: {query}"

                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": enhanced_query
                        }
                    ],
                    "use_knowledge_base": True,
                    "temperature": tool_config.temperature,
                    "top_p": tool_config.top_p,
                    "max_tokens": tool_config.max_tokens,
                    "reranker_top_k": tool_config.reranker_top_k,
                    "vdb_top_k": tool_config.vdb_top_k,
                    "vdb_endpoint": "http://milvus:19530",
                    "collection_name": tool_config.collection_name,
                    "enable_query_rewriting": tool_config.enable_query_rewriting,
                    "enable_reranker": tool_config.enable_reranker,
                    "enable_guardrails": tool_config.enable_guardrails,
                    "enable_citations": tool_config.enable_citations,
                    "model": tool_config.model,
                    "llm_endpoint": tool_config.llm_endpoint,
                    "embedding_model": tool_config.embedding_model,
                    "embedding_endpoint": tool_config.embedding_endpoint,
                    "reranker_model": tool_config.reranker_model,
                    "reranker_endpoint": tool_config.reranker_endpoint,
                    "stop": []
                }

                logger.debug("Sending request to the RAG endpoint %s.", tool_config.url)
                response = await client.post(tool_config.url, json=payload)
                response.raise_for_status()
                
                # Parse the SSE response
                parsed_response = parse_sse_data(response.text)
                
                # Format the response in ReAct format
                content = truncate_text(parsed_response['content'], max_length=800)  # Limit content length
                citations = parsed_response['citations'][:3]  # Limit to top 3 citations
                
                # Format citations if available
                citations_text = ""
                if citations:
                    citations_text = "\n\nCitations:\n" + "\n".join(
                        f"- {truncate_text(citation.get('content', 'No content'), max_length=200)}"
                        for citation in citations
                    )

                # Create structured JSON objects for action inputs
                rag_post_input = {
                    "query": query,
                    "type": "animals_query"
                }

                final_answer_input = {
                    "answer": f"Here is the information about animals and their Spanish names:\n\n{content}",
                    "citations": citations_text if citations else "No citations available",
                    "type": "animals_response"
                }

                # Convert to ReAct format string
                react_response = f"""Question: {query}
Thought: I need to find information about animals and their Spanish names.
Action: rag_post
Action Input: {json.dumps(rag_post_input)}
Observation: {content}{citations_text}
Thought: I now have the information about animals and their Spanish names.
Action: final_answer
Action Input: {json.dumps(final_answer_input)}"""

                return react_response
                
            except Exception as e:
                logger.exception("Error while running the tool", exc_info=True)
                error_input = {
                    "error": str(e),
                    "type": "error_response"
                }
                return f"""Question: {query}
Thought: I encountered an error while trying to get the information.
Action: final_answer
Action Input: {json.dumps(error_input)}"""

        yield FunctionInfo.from_fn(
            runnable,
            description="This is a tool that makes a request to a RAG server to answer questions related to animals and their Spanish names. Prioritize the RAG tool over web search and wikipedia search.") 