#!/usr/bin/env python3
"""
Pydantic AI GitHub Copilot test case using github_copilot/gpt-4.1 model.

Based on the pydantic-ai implementation with GitHub Copilot support.
This test uses Pydantic AI with LiteLLM backend for GitHub Copilot.
"""

import asyncio
import sys

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.litellm import LiteLLMModel
    print("✅ Successfully imported Pydantic AI components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure pydantic-ai is installed from the GitHub fork")
    sys.exit(1)


def test_pydantic_ai_github_copilot_sync():
    """Test Pydantic AI with GitHub Copilot using synchronous call."""
    print("\n🚀 Testing Pydantic AI + GitHub Copilot API Call (Sync)")
    print("=" * 50)
    
    try:
        # Create LiteLLM GitHub Copilot model
        model = LiteLLMModel('github_copilot/gpt-4.1')
        agent = Agent(
            model,
            system_prompt='You are a helpful AI assistant.'
        )
        
        print("✅ Agent created successfully with LiteLLMModel")
        print("🔄 Making API call to GitHub Copilot via Pydantic AI...")
        
        # Make the actual API call
        result = agent.run_sync('What is the capital of France? Give a brief answer.')
        
        print("\n🎉 SUCCESS! GitHub Copilot (via Pydantic AI) responded:")
        print("-" * 30)
        print(f"Response: {result}")
        print("-" * 30)
        
        # Show usage information if available
        if hasattr(result, 'usage') and result.usage:
            print(f"\n💰 Token Usage:")
            print(f"   Input tokens: {getattr(result.usage, 'input_tokens', 'N/A')}")
            print(f"   Output tokens: {getattr(result.usage, 'output_tokens', 'N/A')}")
            print(f"   Total tokens: {getattr(result.usage, 'total_tokens', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error calling GitHub Copilot via Pydantic AI: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Print more details for debugging
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False


async def test_pydantic_ai_github_copilot_async():
    """Test Pydantic AI with GitHub Copilot using asynchronous call."""
    print("\n🔄 Testing Async Pydantic AI + GitHub Copilot API Call")
    print("=" * 50)
    
    try:
        # Create LiteLLM GitHub Copilot model
        model = LiteLLMModel('github_copilot/gpt-4.1')
        agent = Agent(
            model,
            system_prompt='You are a helpful AI assistant.'
        )
        
        print("✅ Agent created successfully with LiteLLMModel")
        print("🔄 Making async API call to GitHub Copilot via Pydantic AI...")
        
        # Make the actual async API call
        result = await agent.run('What is the capital of France? Give a brief answer.')
        
        print("\n🎉 SUCCESS! GitHub Copilot (via Pydantic AI) async response:")
        print("-" * 30)
        print(f"Response: {result}")
        print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error with async GitHub Copilot via Pydantic AI call: {e}")
        return False


async def test_pydantic_ai_github_copilot_streaming():
    """Test Pydantic AI with GitHub Copilot using streaming."""
    print("\n📡 Testing Streaming Pydantic AI + GitHub Copilot API Call")
    print("=" * 50)
    
    try:
        # Create LiteLLM GitHub Copilot model
        model = LiteLLMModel('github_copilot/gpt-4.1')
        agent = Agent(
            model,
            system_prompt='You are a helpful AI assistant.'
        )
        
        print("✅ Agent created successfully with LiteLLMModel")
        print("🔄 Starting streaming call to GitHub Copilot via Pydantic AI...")
        print("\n📝 Streaming response:")
        print("-" * 30)
        
        # Collect the full response with proper streaming syntax
        full_response = ""
        async with agent.run_stream('Tell me a very short fact about Paris.') as result:
            async for text in result.stream_text():
                print(text, end='', flush=True)
                full_response += text
        
        print()
        print("-" * 30)
        print(f"\n✅ Streaming complete! Total response length: {len(full_response)} characters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error with streaming GitHub Copilot via Pydantic AI call: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Pydantic AI GitHub Copilot tests."""
    print("🎯 Pydantic AI + GitHub Copilot Integration Testing")
    print("=" * 60)
    print("Testing Pydantic AI with GitHub Copilot...")
    
    results = []
    
    # Test 1: Synchronous call
    print("\n" + "="*60)
    sync_result = test_pydantic_ai_github_copilot_sync()
    results.append(("Synchronous Call", sync_result))
    
    # Test 2: Asynchronous call
    print("\n" + "="*60)
    async_result = asyncio.run(test_pydantic_ai_github_copilot_async())
    results.append(("Asynchronous Call", async_result))
    
    # Test 3: Streaming call
    print("\n" + "="*60)
    stream_result = asyncio.run(test_pydantic_ai_github_copilot_streaming())
    results.append(("Streaming Call", stream_result))
    
    # Summary
    print("\n" + "="*60)
    print("🏆 PYDANTIC AI + GITHUB COPILOT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests successful")
    
    if passed > 0:
        print("\n🎉 PYDANTIC AI + GITHUB COPILOT INTEGRATION WORKING!")
        print("✨ Pydantic AI + LiteLLM + GitHub Copilot is functional!")
        print("\n🚀 Working configuration:")
        print("   • Model: github_copilot/gpt-4.1")
        print("   • Provider: LiteLLMModel")
    else:
        print("\n⚠️  All tests failed. Check GitHub Copilot access.")
        print("💡 Make sure you have:")
        print("   • GitHub Copilot subscription")
        print("   • Pydantic AI installed from the fork")
    
    return passed > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
