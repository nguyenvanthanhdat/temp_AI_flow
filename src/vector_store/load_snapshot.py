from qdrant import QdrantVectorSpace
import random

def main():
    qdrant_url = "http://localhost:6333"
    token = "huyhoang"
    
    # Initialize QdrantVectorSpace
    vector_space = QdrantVectorSpace(
        qdrant_url=qdrant_url,
        token=token,
        collection_name="test_collection",
        snapshot=True
    )

    # Upload snapshot
    vector_space.load_snapshot("snapshot_collection/sample.snapshot")

    # Query the vector space by using image
    image_vector = [random.random() for _ in range(512)]
    image_results = vector_space.query_by_image(image_vector, top_k=2)

    print("Image Query Results:")
    for result in image_results:
        print(result)

    # Query the vector space by using text
    text_vector = [random.random() for _ in range(512)]
    text_results = vector_space.query_by_text(text_vector, top_k=3)

    print("Text Query Results:")
    for result in text_results:
        print(result)

if __name__ == "__main__":
    main()