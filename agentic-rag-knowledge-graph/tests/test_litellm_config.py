#!/usr/bin/env python3
"""
Test script to verify LiteLLM GitHub Copilot configuration.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our providers
try:
    from agent.providers import get_llm_model, get_embedding_model, get_model_info
    from ingestion.embedder import create_embedder
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Agent directory exists: {(project_root / 'agent').exists()}")
    sys.exit(1)

# Load environment variables with override to ensure fresh values
load_dotenv(override=True)


async def test_embedding():
    """Test GitHub Copilot embedding functionality."""
    print("🔤 Testing GitHub Copilot Embeddings")
    print("-" * 40)
    
    try:
        # Create embedder with GitHub Copilot model
        embedder = create_embedder(
            model="github_copilot/text-embedding-3-small",
            use_cache=False
        )
        
        # Test single embedding
        test_text = "This is a test of GitHub Copilot embeddings through LiteLLM"
        print(f"Text: {test_text}")
        
        embedding = await embedder.generate_embedding(test_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"🔢 First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_batch_embedding():
    """Test batch embedding functionality."""
    print("\n📦 Testing Batch Embeddings")
    print("-" * 40)
    
    try:
        embedder = create_embedder(
            model="github_copilot/text-embedding-3-small",
            use_cache=False
        )
        
        texts = [
            "First test text for batch embedding",
            "Second test text for verification",
            "Third text to complete the batch"
        ]
        
        print(f"Processing {len(texts)} texts...")
        embeddings = await embedder.generate_embeddings_batch(texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"  Text {i+1}: {len(emb)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_chat_model():
    """Test GitHub Copilot chat model functionality."""
    print("\n💬 Testing GitHub Copilot Chat Model")
    print("-" * 40)
    
    try:
        import litellm
        
        # Test chat completion with GitHub Copilot
        model = "github_copilot/gpt-4.1"
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
        
        print(f"Model: {model}")
        print(f"Message: {messages[0]['content']}")
        print("💡 Using OAuth2 authentication (no API key needed)")
        
        # GitHub Copilot uses OAuth2, no API key needed
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=10,
            extra_headers={
                "Editor-Version": "vscode/1.85.0",
                "Copilot-Integration-Id": "vscode-chat"
            }
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ Response: {answer}")
        print(f"💰 Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion tokens")
        
        # Verify it's a reasonable response
        if "paris" in answer.lower():
            print("🎯 Response appears correct!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Note: GitHub Copilot requires OAuth2 authentication and may need IDE context")
        return False


async def test_ingestion_model():
    """Test ingestion model functionality."""
    print("\n📄 Testing Ingestion Model")
    print("-" * 40)
    
    try:
        from agent.providers import get_ingestion_model
        import litellm
        
        # Get ingestion model
        ingestion_model = get_ingestion_model()
        print(f"Ingestion model: {ingestion_model}")
        
        # Test with a simple ingestion task
        messages = [
            {
                "role": "user", 
                "content": "Summarize this text in one sentence: 'LiteLLM is a library that provides a unified interface to multiple LLM providers including OpenAI, Anthropic, and GitHub Copilot. It simplifies the process of switching between different models and providers.'"
            }
        ]
        
        print("Testing ingestion task: text summarization")
        
        # Use LiteLLM directly for ingestion model
        model_name = os.getenv('INGESTION_LLM_CHOICE', 'github_copilot/gpt-4.1')
        
        print(f"Model: {model_name}")
        if model_name.startswith('github_copilot/'):
            print("💡 Using OAuth2 authentication (no API key needed)")
        
        # GitHub Copilot uses OAuth2, other models may need API key
        if model_name.startswith('github_copilot/'):
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=50,
                extra_headers={
                    "Editor-Version": "vscode/1.85.0",
                    "Copilot-Integration-Id": "vscode-chat"
                }
            )
        else:
            api_key = os.getenv('LLM_API_KEY')
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=50,
                api_key=api_key
            )
        
        summary = response.choices[0].message.content.strip()
        print(f"✅ Summary: {summary}")
        print(f"💰 Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration settings."""
    print("\n⚙️  Testing Configuration")
    print("-" * 40)
    
    try:
        # Test model info
        info = get_model_info()
        print("Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test LLM model
        llm_model = get_llm_model()
        print(f"\n✅ LLM Model: {llm_model}")
        
        # Test embedding model
        embedding_model = get_embedding_model()
        print(f"✅ Embedding Model: {embedding_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 LiteLLM GitHub Copilot Configuration Test")
    print("=" * 50)
    
    # Check if we're using GitHub Copilot models (which use OAuth2)
    llm_model = os.getenv('LLM_CHOICE', 'github_copilot/gpt-4.1')
    embedding_model = os.getenv('EMBEDDING_MODEL', 'github_copilot/text-embedding-3-small')
    
    using_github_copilot = (
        llm_model.startswith('github_copilot/') or 
        embedding_model.startswith('github_copilot/')
    )
    
    if using_github_copilot:
        print("💡 Using GitHub Copilot models with OAuth2 authentication")
    else:
        # Check API key for non-GitHub Copilot models
        api_key = os.getenv('LLM_API_KEY')
        if not api_key or api_key == 'your-github-copilot-api-key-here':
            print("⚠️  Warning: LLM_API_KEY not set for non-GitHub Copilot models")
    
    # Run tests
    results = []
    
    # Test configuration
    results.append(test_configuration())
    
    # Test API functionality
    results.append(await test_embedding())
    results.append(await test_batch_embedding())
    results.append(await test_chat_model())
    results.append(await test_ingestion_model())
    
    # Summary
    print("\n📊 Test Summary")
    print("-" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check configuration and API key.")
    
    print("\n💡 Next steps:")
    print("- Set up your .env file with GitHub Copilot API key")
    print("- Test with your actual data")
    print("- Integrate with your RAG pipeline")


if __name__ == "__main__":
    asyncio.run(main())