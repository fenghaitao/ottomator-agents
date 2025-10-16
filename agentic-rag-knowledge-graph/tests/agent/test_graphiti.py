"""
Test cases for Graphiti knowledge graph operations.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.graph_utils import GraphitiClient

# Set up environment for GitHub Copilot testing
os.environ.update({
    'LLM_CHOICE': 'github_copilot/gpt-4.1',
    'LLM_API_KEY': 'test-api-key',
    'EMBEDDING_MODEL': 'github_copilot/text-embedding-ada-002',
    'EMBEDDING_API_KEY': 'test-api-key',
    'NEO4J_URI': 'bolt://localhost:7687',
    'NEO4J_USER': 'neo4j',
    'NEO4J_PASSWORD': 'password'
})


class TestGraphitiClient:
    """Test cases for GraphitiClient operations."""
    
    @pytest.fixture
    def client(self):
        """Create a GraphitiClient instance for testing."""
        return GraphitiClient()
    
    @pytest.fixture
    def mock_graphiti(self):
        """Mock Graphiti core functionality with LiteLLM GitHub Copilot support."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class, \
             patch('agent.graph_utils.litellm.acompletion') as mock_llm, \
             patch('agent.graph_utils.litellm.aembedding') as mock_embedding:
            
            mock_instance = AsyncMock()
            mock_graphiti_class.return_value = mock_instance
            
            # Mock successful initialization
            mock_instance.build_indices_and_constraints = AsyncMock()
            mock_instance.close = AsyncMock()
            
            # Mock LiteLLM responses
            mock_llm_response = MagicMock()
            mock_llm_response.choices = [MagicMock()]
            mock_llm_response.choices[0].message.content = '{"entities": [], "relationships": []}'
            mock_llm.return_value = mock_llm_response
            
            # Mock LiteLLM embedding responses
            mock_embedding_response = MagicMock()
            mock_embedding_response.data = [{'embedding': [0.1] * 1536}]
            mock_embedding.return_value = mock_embedding_response
            
            # Mock search results
            mock_result = type('MockResult', (), {
                'fact': 'OpenAI is an AI research company',
                'uuid': 'test-uuid-123',
                'valid_at': None,
                'invalid_at': None
            })()
            
            mock_instance.search = AsyncMock(return_value=[mock_result])
            
            # Mock add_episode
            mock_instance.add_episode = AsyncMock(return_value="episode_id_123")
            
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_initialization(self, client, mock_graphiti):
        """Test Graphiti client initialization."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            # Test initialization
            await client.initialize()
            
            assert client._initialized is True
            assert client.graphiti is not None
            mock_graphiti.build_indices_and_constraints.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_episode(self, client, mock_graphiti):
        """Test adding an episode to the knowledge graph."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            await client.initialize()
            
            # Test adding episode
            result = await client.add_episode(
                episode_id="test_episode",
                content="This is test content about OpenAI and AI research.",
                source="Test source",
                timestamp=datetime.now(timezone.utc)
            )
            
            assert result == "episode_id_123"
            mock_graphiti.add_episode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_graph(self, client, mock_graphiti):
        """Test searching the knowledge graph."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            await client.initialize()
            
            # Test search
            results = await client.search("OpenAI")
            
            assert len(results) == 1
            assert results[0] == "OpenAI is an AI research company"
            mock_graphiti.search.assert_called_once_with("OpenAI")
    
    @pytest.mark.asyncio
    async def test_get_entity_relationships(self, client, mock_graphiti):
        """Test getting entity relationships."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            await client.initialize()
            
            # Test entity relationships
            results = await client.get_related_entities("OpenAI")
            
            assert len(results) >= 0
            mock_graphiti.search.assert_called_once_with("OpenAI related entities connections")
    
    @pytest.mark.asyncio
    async def test_get_entity_timeline(self, client, mock_graphiti):
        """Test getting entity timeline."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            await client.initialize()
            
            # Test entity timeline
            results = await client.get_entity_timeline("OpenAI")
            
            assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_graph_statistics(self, client, mock_graphiti):
        """Test graph statistics functionality."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            await client.initialize()
            
            # Test graph statistics
            stats = await client.get_graph_statistics()
            
            assert stats["graphiti_initialized"] is True
    
    @pytest.mark.asyncio
    async def test_close_connection(self, client, mock_graphiti):
        """Test closing Graphiti connection."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti_class.return_value = mock_graphiti
            
            await client.initialize()
            await client.close()
            
            assert client._initialized is False
            assert client.graphiti is None
            mock_graphiti.close.assert_called_once()
            
            # Ensure all async operations are completed
            await asyncio.sleep(0.01)
    
    @pytest.mark.asyncio
    async def test_error_handling_initialization(self, client):
        """Test error handling during initialization."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            # Mock initialization failure
            mock_graphiti_class.side_effect = Exception("Connection failed")
            
            # Should not raise exception, but should log error
            await client.initialize()
            
            assert client._initialized is False
            assert client.graphiti is None
    
    @pytest.mark.asyncio
    async def test_search_without_initialization(self, client):
        """Test search operations without initialization."""
        # Test search without initialization
        results = await client.search("test query")
        
        # Should return empty results
        assert len(results) == 0


class TestGraphitiIntegration:
    """Integration tests for Graphiti with LiteLLM GitHub Copilot models."""
    
    @pytest.fixture
    def initialized_client(self):
        """Create an initialized GraphitiClient for integration tests."""
        return GraphitiClient()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_litellm_github_copilot_integration(self, initialized_client):
        """Test GitHub Copilot integration with LiteLLM."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            
            # Mock Graphiti instance with proper async cleanup
            mock_graphiti = AsyncMock()
            mock_graphiti_class.return_value = mock_graphiti
            mock_graphiti.build_indices_and_constraints = AsyncMock()
            mock_graphiti.add_episode = AsyncMock(return_value="episode_123")
            mock_graphiti.search = AsyncMock(return_value=[])
            mock_graphiti.close = AsyncMock()
            
            client = initialized_client
            try:
                await client.initialize()
                
                # Test that the client is configured for GitHub Copilot
                assert client.llm_choice == 'github_copilot/gpt-4.1'
                assert client.use_litellm_direct is True
                
                # Test episode addition with GitHub Copilot (this will use mocked Graphiti)
                episode_id = await client.add_episode(
                    episode_id="github_copilot_test",
                    content="GitHub Copilot provides AI-powered code assistance for developers.",
                    source="Test source",
                    timestamp=datetime.now(timezone.utc)
                )
                
                # Verify that add_episode was called on the mock
                mock_graphiti.add_episode.assert_called_once()
                
                # Verify the episode ID is returned
                assert episode_id == "episode_123"
                
            finally:
                # Ensure proper cleanup
                await client.close()
                # Wait for cleanup to complete
                await asyncio.sleep(0.01)
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_github_copilot_configuration(self, initialized_client):
        """Test that GraphitiClient is properly configured for GitHub Copilot."""
        client = initialized_client
        
        # Check configuration
        assert client.llm_choice == 'github_copilot/gpt-4.1'
        assert client.embedding_model == 'github_copilot/text-embedding-ada-002'
        assert hasattr(client, 'use_litellm_direct')
        assert client.use_litellm_direct is True
        
        # Check that the client has the custom LiteLLM components
        assert hasattr(client, 'graphiti_model')
        assert client.graphiti_model == 'github_copilot/gpt-4.1'
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_graph_statistics_with_litellm(self, initialized_client):
        """Test graph statistics with LiteLLM GitHub Copilot setup."""
        with patch('agent.graph_utils.Graphiti') as mock_graphiti_class:
            mock_graphiti = AsyncMock()
            mock_graphiti_class.return_value = mock_graphiti
            mock_graphiti.build_indices_and_constraints = AsyncMock()
            mock_graphiti.close = AsyncMock()
            
            client = initialized_client
            try:
                await client.initialize()
                
                stats = await client.get_graph_statistics()
                
                assert stats["graphiti_initialized"] is True
                assert stats["llm_model"] == "github_copilot/gpt-4.1"
                assert stats["embedding_model"] == "github_copilot/text-embedding-ada-002"
                assert stats["litellm_integration"] is True
                
            finally:
                # Ensure proper cleanup
                await client.close()
                # Wait for cleanup to complete
                await asyncio.sleep(0.01)


