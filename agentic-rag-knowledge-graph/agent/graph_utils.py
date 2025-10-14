"""
Graph utilities for Neo4j/Graphiti integration.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import asyncio

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
import litellm
from dotenv import load_dotenv

# Load environment variables - force override any existing env vars
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Custom LiteLLM client for Graphiti
class LiteLLMGraphitiClient:
    """Custom LiteLLM client that implements OpenAI AsyncClient interface for Graphiti."""
    
    def __init__(self, model: str):
        self.model = model
        
        # Create a chat object that has the completions.create method
        self.chat = self
        self.completions = self
    
    async def create(self, messages, **kwargs):
        """Chat completion method that Graphiti expects via client.chat.completions.create()."""
        try:
            # Convert Graphiti messages to LiteLLM format
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and hasattr(msg, 'role'):
                    formatted_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                elif isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:
                    logger.warning(f"Unexpected message format: {type(msg)}")
                    continue
            
            logger.info(f"🚀 Calling LiteLLM with GitHub Copilot model: {self.model}")
            
            # Call LiteLLM with modified system prompt for simpler JSON
            # Check if this is a Graphiti prompt and modify accordingly
            is_graphiti_prompt = any(
                'entities' in msg.get('content', '').lower() or 
                'relationships' in msg.get('content', '').lower() or
                'facts' in msg.get('content', '').lower()
                for msg in formatted_messages
            )
            
            if is_graphiti_prompt:
                # Add instruction for simple JSON format compatible with Neo4j
                system_msg = None
                for msg in formatted_messages:
                    if msg.get('role') == 'system':
                        system_msg = msg
                        break
                
                if system_msg:
                    original_content = system_msg['content']
                    system_msg['content'] = (
                        f"{original_content}\n\n"
                        "IMPORTANT: Return only flat JSON objects with simple string/number values. "
                        "DO NOT use nested objects, arrays of objects, or complex structures. "
                        "Use simple key-value pairs only."
                    )
            
            # Handle GPT-5 model constraints
            temperature = kwargs.get('temperature', 0.0)
            if 'gpt-5' in self.model.lower():
                # GPT-5 models only support temperature=1
                temperature = 1.0
                logger.info("🔧 Adjusted temperature to 1.0 for GPT-5 compatibility")
            
            # Call LiteLLM (no max_tokens constraint - let model use natural limits)
            response = await litellm.acompletion(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                response_format=kwargs.get('response_format', {'type': 'json_object'})
            )
            
            logger.info(f"✅ LiteLLM response received successfully")
            
            # Selectively flatten nested objects for Neo4j while preserving Graphiti structure
            # Apply to ALL responses when using GitHub Copilot to ensure Neo4j compatibility
            if hasattr(response, 'choices') and response.choices:
                try:
                    content = response.choices[0].message.content
                    if content:
                        import json
                        parsed_json = json.loads(content)
                        
                        # Only flatten if it's a dictionary
                        if isinstance(parsed_json, dict):
                            flattened_json = self._flatten_neo4j_incompatible_fields(parsed_json)
                            response.choices[0].message.content = json.dumps(flattened_json)
                            logger.info("✅ Flattened Neo4j-incompatible nested objects while preserving Graphiti structure")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Could not flatten response: {e}")
            else:
                logger.info("✅ Keeping natural JSON response format")
            
            # Return in the format Graphiti expects
            return response
            
        except Exception as e:
            logger.error(f"❌ LiteLLM completion failed: {e}")
            raise
    
    def _flatten_json_for_neo4j(self, obj, max_depth=2):
        """
        Selectively flatten nested JSON structures to make them compatible with Neo4j.
        Preserves top-level structure but flattens deeply nested objects.
        
        Args:
            obj: The JSON object to flatten
            max_depth: Maximum nesting depth to allow before flattening
        
        Returns:
            Flattened dictionary with simple values
        """
        def flatten_value(value, depth=0):
            if isinstance(value, dict):
                if depth >= max_depth:
                    # Convert deeply nested dicts to strings
                    return str(value)
                else:
                    # Recursively process dict, but flatten its nested objects
                    result = {}
                    for k, v in value.items():
                        flattened_v = flatten_value(v, depth + 1)
                        if isinstance(flattened_v, dict) and depth + 1 >= max_depth:
                            # Flatten this nested dict
                            for nested_k, nested_v in flattened_v.items():
                                result[f"{k}_{nested_k}"] = nested_v
                        else:
                            result[k] = flattened_v
                    return result
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    return ', '.join(str(item) for item in value)
                else:
                    # For complex lists, stringify each item
                    simple_items = [str(item) for item in value[:5]]  # Limit to first 5 items
                    return ', '.join(simple_items)
            elif isinstance(value, (str, int, float, bool, type(None))):
                # Keep simple types as-is
                return value
            else:
                # Convert other types to strings
                return str(value)
        
        if isinstance(obj, dict):
            return flatten_value(obj)
        else:
            return obj
    
    def _flatten_neo4j_incompatible_fields(self, obj):
        """
        Selectively flatten only the nested objects that Neo4j can't handle,
        while preserving the overall structure that Graphiti expects.
        
        Args:
            obj: The JSON object to process
        
        Returns:
            Object with Neo4j-incompatible nested objects flattened to strings
        """
        def process_value(value):
            if isinstance(value, dict):
                # Check if this is a schema-like object that should be flattened
                if all(key in value for key in ['description', 'title', 'type']):
                    # This looks like a schema definition - flatten to description only
                    return value.get('description', str(value))
                elif len(value) > 0 and all(isinstance(v, dict) for v in value.values()):
                    # Multiple nested dicts - likely incompatible with Neo4j
                    return str(value)
                else:
                    # Regular dict - recurse but preserve structure
                    return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                # Process list items
                processed_list = []
                for item in value:
                    if isinstance(item, dict):
                        processed_list.append(process_value(item))
                    else:
                        processed_list.append(item)
                return processed_list
            else:
                # Primitive value - keep as-is
                return value
        
        if isinstance(obj, dict):
            return process_value(obj)
        else:
            return obj


# Custom LiteLLM embedder for Graphiti
class LiteLLMGraphitiEmbedder(EmbedderClient):
    """Custom LiteLLM embedder that inherits from Graphiti's EmbedderClient."""
    
    def __init__(self, model: str, dimensions: int = 1536):
        self.model = model
        self.dimensions = dimensions
    
    async def create(self, input_data):
        """Main embedder method that Graphiti expects (EmbedderClient interface)."""
        if isinstance(input_data, str):
            # Single text input
            return await self._embed_single(input_data)
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # List of text inputs - return first embedding for compatibility
            embeddings = await self._embed_batch(input_data)
            return embeddings[0] if embeddings else []
        else:
            # Handle other input types as string
            return await self._embed_single(str(input_data))
    
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Batch embedder method that Graphiti may use."""
        return await self._embed_batch(input_data_list)
    
    async def _embed_single(self, text: str) -> list[float]:
        """Embed single text using LiteLLM."""
        try:
            logger.info(f"🚀 Calling LiteLLM embeddings with GitHub Copilot model: {self.model}")
            
            response = await litellm.aembedding(
                model=self.model,
                input=[text]
            )
            
            # Extract the embedding vector
            embedding = response.data[0]['embedding']
            logger.info(f"✅ LiteLLM embedding received successfully (dimension: {len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ LiteLLM embedding failed: {e}")
            raise
    
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using LiteLLM."""
        try:
            logger.info(f"🚀 Calling LiteLLM batch embeddings with GitHub Copilot model: {self.model}")
            
            response = await litellm.aembedding(
                model=self.model,
                input=texts
            )
            
            # Extract all embedding vectors
            embeddings = [item['embedding'] for item in response.data]
            logger.info(f"✅ LiteLLM batch embeddings received successfully ({len(embeddings)} embeddings)")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ LiteLLM batch embedding failed: {e}")
            raise

# Help from this PR for setting up the custom clients: https://github.com/getzep/graphiti/pull/601/files
class GraphitiClient:
    """Manages Graphiti knowledge graph operations."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize Graphiti client.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        # LLM configuration
        self.llm_choice = os.getenv("LLM_CHOICE", "gpt-4.1-mini")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        
        # Configure for GitHub Copilot via LiteLLM
        if self.llm_choice.startswith("github_copilot/"):
            logger.info(f"🚀 Configuring Graphiti for GitHub Copilot model via LiteLLM: {self.llm_choice}")
            
            # Use LiteLLM directly - no proxy server needed
            self.use_litellm_direct = True
            
            # Use the original GitHub Copilot model name - LiteLLM will handle authentication
            self.graphiti_model = self.llm_choice
            
            # GitHub Copilot authentication removed - using direct API key approach
            logger.info("ℹ️  Using direct API key authentication (no GitHub token required)")
            
            logger.info(f"✅ Will use LiteLLM directly for GitHub Copilot model {self.llm_choice}")
        else:
            self.use_litellm_direct = False
            self.graphiti_model = self.llm_choice
        
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        
        # Embedding configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("VECTOR_DIMENSION", "1536"))
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        
        # Configure embedding for GitHub Copilot (no OAuth2 required)
        if self.llm_choice.startswith("github_copilot/"):
            logger.info(f"🚀 Configuring Graphiti embeddings for GitHub Copilot model")
        
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable not set")
        
        self.graphiti: Optional[Graphiti] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Graphiti client."""
        if self._initialized:
            return
        
        try:
            # Configure LLM client based on whether we're using LiteLLM direct
            if self.use_litellm_direct:
                # Use our custom LiteLLM client for GitHub Copilot
                logger.info(f"✅ Using custom LiteLLM client for Graphiti: {self.graphiti_model}")
                
                # We'll create a mock config for Graphiti but use our custom client
                llm_config = LLMConfig(
                    api_key="litellm-direct",  # Placeholder
                    model=self.graphiti_model,
                    small_model=self.graphiti_model,
                    base_url="https://api.openai.com/v1"  # Placeholder
                )
                
                # Create our custom LiteLLM client
                custom_client = LiteLLMGraphitiClient(model=self.graphiti_model)
                
                # Use OpenAIGenericClient but with our custom async client
                llm_client = OpenAIGenericClient(config=llm_config, client=custom_client)
                
            else:
                # Use regular OpenAI client for non-GitHub Copilot models
                llm_config = LLMConfig(
                    api_key=self.llm_api_key,
                    model=self.graphiti_model,
                    small_model=self.graphiti_model,
                    base_url=self.llm_base_url
                )
                llm_client = OpenAIClient(config=llm_config)
                logger.info(f"✅ Using OpenAI client for Graphiti: {self.llm_base_url}")
            
            # Configure embedder - use custom LiteLLM embedder for GitHub Copilot
            if self.use_litellm_direct and self.embedding_model.startswith("github_copilot/"):
                # Use our custom LiteLLM embedder for GitHub Copilot embeddings
                logger.info(f"✅ Using custom LiteLLM embedder for GitHub Copilot: {self.embedding_model}")
                embedder = LiteLLMGraphitiEmbedder(
                    model=self.embedding_model,
                    dimensions=self.embedding_dimensions
                )
            elif self.use_litellm_direct:
                # Use regular OpenAI embedder for non-GitHub Copilot embeddings
                embedder = OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        api_key=self.embedding_api_key,
                        embedding_model=self.embedding_model,
                        embedding_dim=self.embedding_dimensions,
                        base_url=self.embedding_base_url
                    )
                )
                logger.info(f"✅ Using OpenAI embedder: {self.embedding_model}")
            else:
                # Use regular OpenAI embedder for non-LiteLLM configurations
                embedder = OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        api_key=self.embedding_api_key,
                        embedding_model=self.embedding_model,
                        embedding_dim=self.embedding_dimensions,
                        base_url=self.embedding_base_url
                    )
                )
                logger.info(f"✅ Using OpenAI embedder: {self.embedding_model}")
            
            # Initialize Graphiti with custom clients
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )
            
            # Build indices and constraints
            await self.graphiti.build_indices_and_constraints()
            
            self._initialized = True
            logger.info(f"Graphiti client initialized successfully with LLM: {self.llm_choice} and embedder: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise
    
    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")
    
    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an episode to the knowledge graph.
        
        Args:
            episode_id: Unique episode identifier
            content: Episode content
            source: Source of the content
            timestamp: Episode timestamp
            metadata: Additional metadata
        """
        if not self._initialized:
            await self.initialize()
        
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        
        # Import EpisodeType for proper source handling
        from graphiti_core.nodes import EpisodeType
        
        await self.graphiti.add_episode(
            name=episode_id,
            episode_body=content,
            source=EpisodeType.text,  # Always use text type for our content
            source_description=source,
            reference_time=episode_timestamp
        )
        
        logger.info(f"Added episode {episode_id} to knowledge graph")
    
    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.
        
        Args:
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search
        
        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's search method (simplified parameters)
            results = await self.graphiti.search(query)
            
            # Convert results to dictionaries
            return [
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                    "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None,
                    "source_node_uuid": str(result.source_node_uuid) if hasattr(result, 'source_node_uuid') and result.source_node_uuid else None
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get entities related to a given entity using Graphiti search.
        
        Args:
            entity_name: Name of the entity
            relationship_types: Types of relationships to follow (not used with Graphiti)
            depth: Maximum depth to traverse (not used with Graphiti)
        
        Returns:
            Related entities and relationships
        """
        if not self._initialized:
            await self.initialize()
        
        # Use Graphiti search to find related information about the entity
        results = await self.graphiti.search(f"relationships involving {entity_name}")
        
        # Extract entity information from the search results
        related_entities = set()
        facts = []
        
        for result in results:
            facts.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None
            })
            
            # Simple entity extraction from fact text (could be enhanced)
            if entity_name.lower() in result.fact.lower():
                related_entities.add(entity_name)
        
        return {
            "central_entity": entity_name,
            "related_facts": facts,
            "search_method": "graphiti_semantic_search"
        }
    
    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of facts for an entity using Graphiti.
        
        Args:
            entity_name: Name of the entity
            start_date: Start of time range (not currently used)
            end_date: End of time range (not currently used)
        
        Returns:
            Timeline of facts
        """
        if not self._initialized:
            await self.initialize()
        
        # Search for temporal information about the entity
        results = await self.graphiti.search(f"timeline history of {entity_name}")
        
        timeline = []
        for result in results:
            timeline.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None
            })
        
        # Sort by valid_at if available
        timeline.sort(key=lambda x: x.get('valid_at') or '', reverse=True)
        
        return timeline
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        if not self._initialized:
            await self.initialize()
        
        # For now, return a simple search to verify the graph is working
        # More detailed statistics would require direct Neo4j access
        try:
            test_results = await self.graphiti.search("test")
            return {
                "graphiti_initialized": True,
                "sample_search_results": len(test_results),
                "note": "Detailed statistics require direct Neo4j access"
            }
        except Exception as e:
            return {
                "graphiti_initialized": False,
                "error": str(e)
            }
    
    async def clear_graph(self):
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's proper clear_data function with the driver
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
        except Exception as e:
            logger.error(f"Failed to clear graph using clear_data: {e}")
            # Fallback: Close and reinitialize (this will create fresh indices)
            if self.graphiti:
                await self.graphiti.close()
            
            # Create OpenAI-compatible clients for reinitialization
            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,
                base_url=self.llm_base_url
            )
            
            llm_client = OpenAIClient(config=llm_config)
            
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url
                )
            )
            
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )
            await self.graphiti.build_indices_and_constraints()
            
            logger.warning("Reinitialized Graphiti client (fresh indices created)")


# Global Graphiti client instance
graph_client = GraphitiClient()


async def initialize_graph():
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph():
    """Close graph client."""
    await graph_client.close()


# Convenience functions for common operations
async def add_to_knowledge_graph(
    content: str,
    source: str,
    episode_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add content to the knowledge graph.
    
    Args:
        content: Content to add
        source: Source of the content
        episode_id: Optional episode ID
        metadata: Optional metadata
    
    Returns:
        Episode ID
    """
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"
    
    await graph_client.add_episode(
        episode_id=episode_id,
        content=content,
        source=source,
        metadata=metadata
    )
    
    return episode_id


async def search_knowledge_graph(
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph.
    
    Args:
        query: Search query
    
    Returns:
        Search results
    """
    return await graph_client.search(query)


async def get_entity_relationships(
    entity: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get relationships for an entity.
    
    Args:
        entity: Entity name
        depth: Maximum traversal depth
    
    Returns:
        Entity relationships
    """
    return await graph_client.get_related_entities(entity, depth=depth)


async def test_graph_connection() -> bool:
    """
    Test graph database connection.
    
    Returns:
        True if connection successful
    """
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False