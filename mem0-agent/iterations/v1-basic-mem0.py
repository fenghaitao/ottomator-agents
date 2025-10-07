from dotenv import load_dotenv
from mem0 import Memory

# Load environment variables
load_dotenv()

config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "github_copilot/gpt-4.1",
            "temperature": 0.7,
            "max_tokens": 1000,
            # OAuth2 authentication handled automatically
        }
    },
    "embedder": {
        "provider": "github_copilot", 
        "config": {
            "model": "github_copilot/text-embedding-3-small",
            "embedding_dims": 1536,
            # OAuth2 authentication handled automatically
        }
    }
}

memory = Memory.from_config(config)

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
    
    # Generate Assistant response using GitHub Copilot through mem0's LLM
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    
    # Use mem0's configured LLM (GitHub Copilot) to generate response
    assistant_response = memory.llm.generate_response(messages)

    # Create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()
