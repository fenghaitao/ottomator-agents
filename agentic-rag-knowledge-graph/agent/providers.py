"""
Flexible provider configuration for LLM and embedding models using LiteLLM.
"""

import os
from typing import Optional, Union
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.litellm import LiteLLMModel
import openai
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_model(model_choice: Optional[str] = None) -> Union[LiteLLMModel, OpenAIModel]:
    """
    Get LLM model configuration based on environment variables using LiteLLM.
    
    Args:
        model_choice: Optional override for model choice
    
    Returns:
        Configured LiteLLM model for GitHub Copilot or OpenAI model for other providers
    """
    llm_choice = model_choice or os.getenv('LLM_CHOICE', 'github_copilot/gpt-4.1')
    api_key = os.getenv('LLM_API_KEY', '')
    
    # For GitHub Copilot models, use native LiteLLM support
    if llm_choice.startswith('github_copilot/'):
        print(f"✅ Using native LiteLLM GitHub Copilot model: {llm_choice}")
        return LiteLLMModel(llm_choice)
    else:
        # For non-GitHub Copilot models, use OpenAI provider
        provider = OpenAIProvider(api_key=api_key)
        return OpenAIModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables using LiteLLM.
    
    Returns:
        Configured OpenAI-compatible client for embeddings through LiteLLM
    """
    api_key = os.getenv('EMBEDDING_API_KEY', '')
    
    # LiteLLM handles GitHub Copilot embeddings directly
    # We use standard OpenAI client but LiteLLM will route based on model name
    return openai.AsyncOpenAI(api_key=api_key)


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.
    
    Returns:
        Embedding model name
    """
    return os.getenv('EMBEDDING_MODEL', 'github_copilot/text-embedding-3-small')


def get_ingestion_model() -> Union[LiteLLMModel, OpenAIModel]:
    """
    Get ingestion-specific LLM model (can be faster/cheaper than main model).
    
    Returns:
        Configured model for ingestion tasks
    """
    ingestion_choice = os.getenv('INGESTION_LLM_CHOICE')
    
    # If no specific ingestion model, use the main model
    if not ingestion_choice:
        return get_llm_model()
    
    return get_llm_model(model_choice=ingestion_choice)


# Provider information functions
def get_llm_provider() -> str:
    """Get the LLM provider name."""
    return os.getenv('LLM_PROVIDER', 'github_copilot')


def get_embedding_provider() -> str:
    """Get the embedding provider name."""
    return os.getenv('EMBEDDING_PROVIDER', 'github_copilot')


def validate_configuration() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if configuration is valid
    """
    # For GitHub Copilot, OAuth2 is used, so API keys are not strictly required
    provider = get_llm_provider()
    
    if provider == 'github_copilot':
        # GitHub Copilot uses OAuth2, just check model configuration
        required_vars = ['LLM_CHOICE', 'EMBEDDING_MODEL']
    else:
        # Other providers need API keys
        required_vars = [
            'LLM_API_KEY',
            'LLM_CHOICE', 
            'EMBEDDING_API_KEY',
            'EMBEDDING_MODEL'
        ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def get_model_info() -> dict:
    """
    Get information about current model configuration.
    
    Returns:
        Dictionary with model configuration info
    """
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": os.getenv('LLM_CHOICE'),
        "llm_base_url": os.getenv('LLM_BASE_URL'),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "embedding_base_url": os.getenv('EMBEDDING_BASE_URL'),
        "ingestion_model": os.getenv('INGESTION_LLM_CHOICE', 'same as main'),
    }