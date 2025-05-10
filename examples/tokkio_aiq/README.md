# A Knowledge Agent 

This is an agent that can handle a variety of questions. The agent is using a LLM with reasoning ability. You can find more information about the models at: https://build.nvidia.com/ 

When a query comes, the agent can think, make a decision to use one or more of four tools (RAG, LLM, WebSearch, Wikipedia), analyse the answers, and generate a final answer.

The RAG server can be set up using NVIDIA RAG blueprint: https://github.com/NVIDIA-AI-Blueprints/rag
You can upload PDF files to RAG for any specific domain, such as financial reports. Then the agent could call the RAG tool to answer your queries related to this domain. 

For general questions, the agent could call the LLM tool.

For news or date related queries, the agent could call the Web Search tool using Tavily.

For wikipedia related questions, the agent could call the Wikipedia search tool.


## How to run it
How to run it






