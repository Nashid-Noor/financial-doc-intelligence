from qdrant_client import QdrantClient
client = QdrantClient(":memory:")
print(f"Methods: {[m for m in dir(client) if not m.startswith('_')]}")
try:
    print(f"Has query_points: {hasattr(client, 'query_points')}")
except:
    pass
