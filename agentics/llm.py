"""Central LLM factory — reads OPENAI_API_KEY and OPENAI_BASE_URL from the .env file."""
import os
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the Vocareum proxy."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ.get("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
    )
