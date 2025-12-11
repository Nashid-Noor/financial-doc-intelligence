from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="test",
    vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE)
)
client.upsert(
    collection_name="test",
    points=[
        models.PointStruct(id=1, vector=[0.1]*10, payload={"text": "hello"})
    ]
)

try:
    print("Trying query_points...")
    result = client.query_points(
        collection_name="test",
        query=[0.1]*10,
        limit=1
    )
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    if hasattr(result, 'points'):
        print(f"Result.points: {result.points}")
except Exception as e:
    print(f"Error calling query_points: {e}")
