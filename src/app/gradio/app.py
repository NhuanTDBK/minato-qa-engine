import os
import gradio as gr
from elasticsearch import Elasticsearch
from langchain_community.chat_models import ChatOpenAI

from src.agents.websearch import summarize_websearch_conversation, refine_query

# Initialize the Elasticsearch client
INDEX_NAME = "naruto_episode"
es = Elasticsearch(
    [
        {
            "host": os.environ.get("ES_HOST", "localhost"),
            "port": os.environ.get("ES_PORT", 9200),
        }
    ]
)

base_url, model, api_key = "http://localhost:1234/v1", "llama3", "as"
base_url, model, api_key = (
    "http://localhost:1234/v1",
    "QuantFactory/Phi-3-mini-128k-instruct-GGUF/Phi-3-mini-128k-instruct.Q8_0.gguf",
    "as",
)
# base_url, model, api_key = "https://api.groq.com/openai/v1", "llama3-8b-8192"
# llm_rag = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)
llm_query = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)
llm_rag = llm_query


def search_as_you_type(query):
    # Elasticsearch query for autocomplete
    body = {
        "suggest": {
            "query_suggest": {
                "prefix": query.lower(),
                "completion": {"field": "title.suggest"},
            }
        }
    }
    response = es.search(index=INDEX_NAME, body=body, _source=["title"])
    suggestions = [
        option["text"] for option in response["suggest"]["query_suggest"][0]["options"]
    ]
    return list(set(suggestions))


def search(query):
    # Elasticsearch query for final search
    body = {
        "query": {
            "multi_match": {"query": query.lower(), "fields": ["content"]}
        }
    }
    response = es.search(index=INDEX_NAME, body=body)
    results = response["hits"]["hits"]
    return [result["_source"] for result in results]


def update_suggestions(query):
    suggestions = search_as_you_type(query)
    return "\n".join(suggestions)


def search_results(query, stream_mode):
    refined_query = refine_query(llm=llm_query, query=query).content.strip()
    print("Refined query:", refined_query)
    results = search(refined_query)
    # Take first result, and show content
    if results:
        # take 3 contents
        contents = "\n".join(["{i}. " + result["content"] for i, result in enumerate(results[:3])])
        print("Get RAG: ", len(contents))
        llm_summarization_response = summarize_websearch_conversation(
            llm=llm_rag,
            query=query,
            context=contents,
            stream=stream_mode,
        )
        output = ""
        for chunk in llm_summarization_response:
            content = chunk.content
            output = output + content
            yield output
        # return llm_summarization_response

    # return "Try another query"


with gr.Blocks() as demo:
    search_box = gr.Textbox(label="Search", placeholder="Type your query here...")
    suggestions_box = gr.Textbox(label="Suggestions", interactive=False)
    search_button = gr.Button("Search")
    stream_mode = gr.Checkbox(label="Stream mode", value=False)
    result_box = gr.Textbox(label="Results", lines=10)

    # search_box.change(fn=update_suggestions, inputs=search_box, outputs=suggestions_box)
    search_button.click(
        fn=search_results,
        inputs=[search_box, stream_mode],
        outputs=result_box,
        queue=True,
    )

demo.queue()
demo.launch()
