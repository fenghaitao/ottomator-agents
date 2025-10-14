#!/usr/bin/env python3
"""
Full RAG Pipeline Test with GitHub Copilot Models

This test demonstrates the complete workflow:
1. Document ingestion and chunking
2. Embedding generation with GitHub Copilot
3. Vector storage and indexing
4. Semantic search functionality
5. RAG response generation
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our components
try:
    from ingestion.chunker import create_chunker, ChunkingConfig
    from ingestion.embedder import create_embedder
    from agent.providers import get_llm_model, get_embedding_model
    import litellm
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load environment variables with override
load_dotenv(override=True)


class RAGPipelineTest:
    """Test class for the complete RAG pipeline."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        # Chunking configuration
        self.chunking_config = ChunkingConfig(
            chunk_size=800,
            chunk_overlap=150,
            use_semantic_splitting=True
        )
        
        # Create components
        self.chunker = create_chunker(self.chunking_config)
        self.embedder = create_embedder(
            model="github_copilot/text-embedding-3-small",
            use_cache=True
        )
        
        # Storage for processed documents
        self.document_chunks = []
        self.chunk_embeddings = []
        
    async def ingest_documents(self, doc_directory: str = "big_tech_docs") -> Dict[str, Any]:
        """
        Ingest and process documents from the specified directory.
        
        Args:
            doc_directory: Directory containing documents to process
            
        Returns:
            Dictionary with ingestion statistics
        """
        print("📁 Document Ingestion Phase")
        print("=" * 50)
        
        doc_path = project_root / doc_directory
        if not doc_path.exists():
            print(f"❌ Document directory not found: {doc_path}")
            return {"error": f"Directory not found: {doc_path}"}
        
        # Find all markdown files
        md_files = list(doc_path.glob("*.md"))
        print(f"Found {len(md_files)} documents to process")
        
        total_chunks = 0
        processed_docs = 0
        
        for md_file in md_files[:5]:  # Process first 5 files for testing
            print(f"\n📄 Processing: {md_file.name}")
            
            try:
                # Read document content
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    print(f"⚠️  Skipping empty file: {md_file.name}")
                    continue
                
                # Chunk the document
                chunks = await self.chunker.chunk_document(
                    content=content,
                    title=md_file.stem.replace('_', ' ').title(),
                    source=str(md_file.name)
                )
                
                print(f"  Created {len(chunks)} chunks")
                
                # Generate embeddings for chunks
                def progress_callback(current, total):
                    print(f"  Embedding batch {current}/{total}")
                
                embedded_chunks = await self.embedder.embed_chunks(
                    chunks, 
                    progress_callback=progress_callback
                )
                
                # Store chunks and embeddings
                self.document_chunks.extend(embedded_chunks)
                
                total_chunks += len(embedded_chunks)
                processed_docs += 1
                
                print(f"  ✅ Processed {len(embedded_chunks)} embedded chunks")
                
            except Exception as e:
                print(f"  ❌ Error processing {md_file.name}: {e}")
                continue
        
        stats = {
            "processed_documents": processed_docs,
            "total_chunks": total_chunks,
            "embedding_model": "github_copilot/text-embedding-3-small",
            "chunk_config": {
                "size": self.chunking_config.chunk_size,
                "overlap": self.chunking_config.chunk_overlap,
                "semantic_splitting": self.chunking_config.use_semantic_splitting
            }
        }
        
        print(f"\n📊 Ingestion Complete:")
        print(f"  Documents processed: {processed_docs}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Embedding model: github_copilot/text-embedding-3-small")
        
        return stats
    
    async def test_semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Test semantic search functionality.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        print(f"\n🔍 Semantic Search Test")
        print("=" * 50)
        print(f"Query: '{query}'")
        print(f"Searching through {len(self.document_chunks)} chunks...")
        
        if not self.document_chunks:
            print("❌ No documents ingested yet")
            return []
        
        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)
        print(f"✅ Generated query embedding ({len(query_embedding)} dimensions)")
        
        # Calculate similarity scores
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0
        
        # Score all chunks
        scored_chunks = []
        for chunk in self.document_chunks:
            if hasattr(chunk, 'embedding') and chunk.embedding:
                similarity = cosine_similarity(query_embedding, chunk.embedding)
                scored_chunks.append({
                    "chunk": chunk,
                    "score": similarity,
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "source": chunk.metadata.get('source', 'Unknown'),
                    "title": chunk.metadata.get('title', 'Unknown')
                })
        
        # Sort by similarity score
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_results = scored_chunks[:top_k]
        
        print(f"\n🏆 Top {len(top_results)} Results:")
        for i, result in enumerate(top_results, 1):
            print(f"\n{i}. [{result['score']:.4f}] {result['title']}")
            print(f"   Source: {result['source']}")
            print(f"   Content: {result['content']}")
        
        return top_results
    
    async def test_rag_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Test RAG response generation using GitHub Copilot chat model.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated response
        """
        print(f"\n💬 RAG Response Generation")
        print("=" * 50)
        
        if not context_chunks:
            print("❌ No context chunks provided")
            return "No relevant context found."
        
        # Prepare context from top chunks
        context_texts = []
        for chunk_data in context_chunks[:3]:  # Use top 3 chunks
            chunk = chunk_data['chunk']
            context_texts.append(f"Source: {chunk.metadata.get('source', 'Unknown')}\n{chunk.content}")
        
        context = "\n\n---\n\n".join(context_texts)
        
        # Create RAG prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
Use only the information from the context to answer questions. If the answer is not in the context, say so.
Be concise but comprehensive in your responses."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"Generating response using github_copilot/gpt-4.1...")
        print(f"Context length: {len(context)} characters")
        print(f"Using {len(context_chunks)} context chunks")
        
        try:
            # Generate response using GitHub Copilot
            response = await litellm.acompletion(
                model="github_copilot/gpt-4.1",
                messages=messages,
                max_tokens=500,
                extra_headers={
                    "Editor-Version": "vscode/1.85.0",
                    "Copilot-Integration-Id": "vscode-chat"
                }
            )
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            print(f"✅ Response generated!")
            print(f"💰 Token usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
            print(f"\n📝 Response:\n{answer}")
            
            return answer
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {e}"
            print(error_msg)
            return error_msg
    
    async def run_full_pipeline_test(self):
        """Run the complete RAG pipeline test."""
        print("🚀 Full RAG Pipeline Test with GitHub Copilot")
        print("=" * 70)
        
        try:
            # Step 1: Document Ingestion
            stats = await self.ingest_documents()
            if "error" in stats:
                print(f"❌ Ingestion failed: {stats['error']}")
                return
            
            # Step 2: Semantic Search Tests
            test_queries = [
                "What is OpenAI's funding situation?",
                "Tell me about Microsoft and OpenAI partnership",
                "What are the latest developments in AI regulation?",
                "How is NVIDIA performing in the AI market?"
            ]
            
            search_results = {}
            for query in test_queries:
                results = await self.test_semantic_search(query, top_k=5)
                search_results[query] = results
                
                # Step 3: RAG Response for each query
                if results:
                    await self.test_rag_response(query, results)
                
                print("\n" + "="*70)
            
            # Final Summary
            print(f"\n🎉 Pipeline Test Complete!")
            print(f"📊 Summary:")
            print(f"  - Documents processed: {stats['processed_documents']}")
            print(f"  - Chunks created: {stats['total_chunks']}")
            print(f"  - Queries tested: {len(test_queries)}")
            print(f"  - Embedding model: github_copilot/text-embedding-3-small")
            print(f"  - Chat model: github_copilot/gpt-4.1")
            print(f"  - Authentication: OAuth2")
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Run the full RAG pipeline test."""
    pipeline = RAGPipelineTest()
    await pipeline.run_full_pipeline_test()


if __name__ == "__main__":
    asyncio.run(main())