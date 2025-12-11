from qdrant_client import QdrantClient
print(f"QdrantClient has search: {hasattr(QdrantClient, 'search')}")
client = QdrantClient(":memory:")
print(f"Instance has search: {hasattr(client, 'search')}")
try:
    client.search(collection_name="test", query_vector=[0.1]*10)
except Exception as e:
    print(f"Error calling search: {e}")
