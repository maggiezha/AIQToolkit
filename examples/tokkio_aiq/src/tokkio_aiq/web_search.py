import logging
from pydantic import Field
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import json

logger = logging.getLogger(__name__)

def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text using various patterns.
    
    Args:
        text: Text to extract dates from
        
    Returns:
        List of found dates in ISO format (YYYY-MM-DD)
    """
    # Common date patterns
    patterns = [
        # ISO format (YYYY-MM-DD)
        r'\d{4}-\d{2}-\d{2}',
        # Month DD, YYYY
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
        # DD Month YYYY
        r'\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}',
        # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}/\d{1,2}/\d{4}',
        # Today, Yesterday, etc.
        r'(?:today|yesterday|tomorrow)',
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            date_str = match.group()
            try:
                # Handle special cases
                if date_str.lower() == 'today':
                    dates.append(datetime.now().strftime('%Y-%m-%d'))
                elif date_str.lower() == 'yesterday':
                    yesterday = datetime.now().replace(day=datetime.now().day-1)
                    dates.append(yesterday.strftime('%Y-%m-%d'))
                elif date_str.lower() == 'tomorrow':
                    tomorrow = datetime.now().replace(day=datetime.now().day+1)
                    dates.append(tomorrow.strftime('%Y-%m-%d'))
                else:
                    # Try to parse the date
                    try:
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        try:
                            parsed_date = datetime.strptime(date_str, '%B %d, %Y')
                        except ValueError:
                            try:
                                parsed_date = datetime.strptime(date_str, '%d %B %Y')
                            except ValueError:
                                try:
                                    parsed_date = datetime.strptime(date_str, '%m/%d/%Y')
                                except ValueError:
                                    continue
                    dates.append(parsed_date.strftime('%Y-%m-%d'))
            except Exception as e:
                logger.debug(f"Failed to parse date {date_str}: {str(e)}")
                continue
    
    return list(set(dates))  # Remove duplicates

def truncate_text(text: str, max_length: int = 500) -> str:
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

def parse_search_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse and format Tavily search results into a structured format.
    
    Args:
        results: List of search results from Tavily API
        
    Returns:
        Dict containing formatted results with the following structure:
        {
            "summary": str,  # Concatenated content from all results
            "sources": List[Dict],  # List of sources with title, url, and content
            "total_results": int,  # Total number of results
            "dates": List[str],  # List of dates found in the results
            "current_date": str  # Today's date in ISO format
        }
    """
    try:
        if not results:
            return {
                "summary": "No results found",
                "sources": [],
                "total_results": 0,
                "dates": [],
                "current_date": datetime.now().strftime('%Y-%m-%d')
            }

        # Extract and format sources
        sources = []
        all_content = []
        for result in results:
            source = {
                "title": truncate_text(result.get("title", ""), max_length=100),
                "url": result.get("url", ""),
                "content": truncate_text(result.get("content", ""), max_length=300)
            }
            sources.append(source)
            all_content.append(source["content"])

        # Create a summary by concatenating content from all results
        summary = truncate_text("\n\n".join(all_content), max_length=800)

        # Extract dates from all content
        dates = extract_dates(summary)

        return {
            "summary": summary,
            "sources": sources,
            "total_results": len(results),
            "dates": dates,
            "current_date": datetime.now().strftime('%Y-%m-%d')
        }

    except Exception as e:
        logger.error(f"Error parsing search results: {str(e)}")
        return {
            "summary": f"Error parsing results: {str(e)}",
            "sources": [],
            "total_results": 0,
            "dates": [],
            "current_date": datetime.now().strftime('%Y-%m-%d')
        }

class WebSearchConfig(FunctionBaseConfig, name="web_search"):
    """Configuration for Tavily web search tool."""

    api_key: str = Field(description="Tavily API key")
    max_results: int = Field(
        default=3,
        description="Maximum number of search results to return"
    )
    search_depth: str = Field(
        default="basic",
        description="Search depth: 'basic' or 'deep'"
    )
    include_domains: list[str] = Field(
        default_factory=list,
        description="List of domains to include in search"
    )
    exclude_domains: list[str] = Field(
        default_factory=list,
        description="List of domains to exclude from search"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to enable verbose logging"
    )


@register_function(config_type=WebSearchConfig)
async def web_search_function(tool_config: WebSearchConfig, builder: Builder):
    """Function to perform web search using Tavily API."""
    
    if not tool_config.api_key:
        raise ValueError("API key is required for Tavily web search")

    # Set the API key in environment variable
    os.environ["TAVILY_API_KEY"] = tool_config.api_key

    # Initialize Tavily search tool
    tavily_tool = TavilySearchResults(
        api_key=tool_config.api_key,
        max_results=tool_config.max_results,
        search_depth=tool_config.search_depth,
        include_domains=tool_config.include_domains or [],
        exclude_domains=tool_config.exclude_domains or []
    )

    async def runnable(query: str) -> str:
        try:
            if tool_config.verbose:
                logger.debug("Performing web search with query: %s", query)

            # Create structured input
            search_input = {
                "query": query,
                "type": "web_search_query"
            }

            # Perform the search
            results = tavily_tool.invoke(query)
            
            if tool_config.verbose:
                logger.debug("Raw search results: %s", results)

            # Parse and format the results
            parsed_results = parse_search_results(results)
            
            if tool_config.verbose:
                logger.debug("Parsed search results: %s", parsed_results)

            # Create structured output
            output = {
                "action": "web_search",
                "action_input": json.dumps(search_input),
                "observation": json.dumps(parsed_results),
                "final_answer": json.dumps({
                    "answer": parsed_results["summary"],
                    "sources": parsed_results["sources"],
                    "dates": parsed_results["dates"]
                })
            }

            return json.dumps(output)

        except Exception as e:
            error_msg = f"Error occurred while performing web search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_output = {
                "action": "web_search",
                "action_input": json.dumps({"query": query, "type": "web_search_query"}),
                "observation": json.dumps({
                    "error": error_msg,
                    "summary": error_msg,
                    "sources": [],
                    "total_results": 0,
                    "dates": [],
                    "current_date": datetime.now().strftime('%Y-%m-%d')
                }),
                "final_answer": json.dumps({
                    "error": error_msg,
                    "answer": error_msg,
                    "sources": [],
                    "dates": []
                })
            }
            return json.dumps(error_output)

    yield FunctionInfo.from_fn(
        runnable,
        description="This is a tool to perform web search using Tavily API to get relevant information from the Internet"
    ) 