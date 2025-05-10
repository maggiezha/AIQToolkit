# A Knowledge Agent 

This is a versatile agent that utilizes a Large Language Model (LLM) with advanced reasoning capabilities to provide accurate and informative responses to a wide range of queries. The agent is equipped with four specialized tools:

RAG (Retrieval-Augmented Generator): enabled by the NVIDIA RAG blueprint (https://github.com/NVIDIA-AI-Blueprints/rag), which allows for the upload of domain-specific PDF files, such as financial reports.

LLM: handles general questions and provides insightful responses.

WebSearch: powered by Tavily, provides up-to-date information on news and date-related queries.

Wikipedia: provides authoritative answers to questions on a vast range of topics.

When a query is received, the agent uses its reasoning capabilities to select the most suitable tool(s), analyse the answers, and generate a final answer.

You can find more information about the models used here, or try other models by visiting: https://build.nvidia.com/


## How to run it

```bash
uv pip install -e .
```

```bash
aiq serve --config_file configs/config.yml --host 0.0.0.0 --port 8000
```
aiq serve --config_file configs/config.yml --host 0.0.0.0 --port 8000

You can test with swagger endpoints at http://localhost:8000/docs

Or send a query:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ "input_messages": "What is NVIDIA's revenue in the year 2025?"
}'
```




