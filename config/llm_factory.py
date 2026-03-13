from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama # Use the new package as suggested by your logs
from config.settings import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL

def get_llm(provider="gemini", model_name=None, temperature=0.1):
    """Returns a model and ensures we don't crash on free-tier limits."""
    
    if provider == "ollama":
        # Local models have NO rate limits! Great for testing.
        return ChatOllama(
            model=model_name or "deepseek-coder",
            base_url="http://localhost:11434",
            temperature=temperature
        )
    
    if provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY,
            temperature=temperature
        )
    
    # Default to Gemini, but add retries for that 429 error
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_retries=5, 
        convert_system_message_to_human=True
    )