from typing import Optional
from pydantic import BaseSettings


class AppSetting(BaseSettings):
    app_name: str = "Gradio"
    app_version: str = "1.0.0"
    app_description: str = "Gradio is a simple web-based UI for your machine learning models."
    groq_api_key: Optional[str] = None
    groq_base_url: Optional[str] = "https://api.groq.com/openai/v1"
    ollama_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = "http://localhost:11434/v1"
    query_model: Optional[str] = "llama3"
    retrieval_model: Optional[str] = "llama3-8b-8192"