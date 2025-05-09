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
    reranker_top_k: int = Field(default=10, description="Number of top results from reranker.")
    vdb_top_k: int = Field(default=50, description="Number of top results from vector database.")
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
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
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
                content = parsed_response['content']
                citations = parsed_response['citations']
                
                # Format citations if available
                citations_text = ""
                if citations:
                    citations_text = "\n\nCitations:\n" + "\n".join(
                        f"- {citation.get('content', 'No content')}"
                        for citation in citations
                    )
                
                # Format the response in ReAct format
                react_response = f"""Question: {query}
Thought: I need to analyze the financial report to find the revenue information.
Action: rag_post
Action Input: {query}
Observation: {content}{citations_text}
Thought: I now have the information from the financial report.
Action: final_answer
Action Input: Based on the financial report, {content}"""

                return react_response
                
            except Exception as e:
                logger.exception("Error while running the tool", exc_info=True)
                return f"""Question: {query}
Thought: I encountered an error while trying to get the information.
Action: final_answer
Action Input: Error while running the tool: {e}"""

        yield FunctionInfo.from_fn(
            runnable,
            description="This is a financial report analysis tool that makes a request to a RAG server to answer questions related to the NVIDIA financial report. No not use wikipedia search.")

