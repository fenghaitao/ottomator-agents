#!/usr/bin/env python3
"""
Neo4j connection test script.
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j database connection"""
    
    # Get Neo4j configuration from environment
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("❌ Neo4j configuration incomplete")
        print("Please check your .env file for:")
        print("  - NEO4J_URI (e.g., bolt://localhost:7687)")
        print("  - NEO4J_USER (e.g., neo4j)")
        print("  - NEO4J_PASSWORD")
        return False
    
    print(f"🔗 Connecting to Neo4j at {neo4j_uri}...")
    
    try:
        # Create driver
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Test connection with a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            
            if record and record["test"] == 1:
                print("✅ Neo4j connection successful")
                
                # Get Neo4j version info
                version_result = session.run("CALL dbms.components() YIELD name, versions, edition")
                for record in version_result:
                    if record["name"] == "Neo4j Kernel":
                        print(f"📋 Neo4j version: {record['versions'][0]} ({record['edition']})")
                
                # Check if APOC plugin is available (useful for graph operations)
                try:
                    apoc_result = session.run("RETURN apoc.version() as version")
                    apoc_record = apoc_result.single()
                    if apoc_record:
                        print(f"🔌 APOC plugin version: {apoc_record['version']}")
                except Exception:
                    print("ℹ️  APOC plugin not detected (optional but recommended)")
                
                # Test basic graph operations
                print("🧪 Testing basic graph operations...")
                
                # Create a test node
                session.run("MERGE (test:TestNode {name: 'connection_test'})")
                
                # Count nodes
                count_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = count_result.single()["node_count"]
                print(f"📊 Total nodes in database: {node_count}")
                
                # Clean up test node
                session.run("MATCH (test:TestNode {name: 'connection_test'}) DELETE test")
                
                print("🎉 Neo4j database is ready!")
                return True
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Neo4j is running")
        print("2. Check your connection URI (bolt://localhost:7687)")
        print("3. Verify username and password")
        print("4. Ensure Neo4j is accepting connections")
        return False

if __name__ == "__main__":
    success = test_neo4j_connection()
    sys.exit(0 if success else 1)