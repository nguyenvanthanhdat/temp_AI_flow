from qdrant import QdrantVectorSpace
import random

def main():
    qdrant_url = "http://localhost:6333"
    token = "huyhoang"
    collection_name = "test_collection"
    
    # Initialize QdrantVectorSpace
    vector_space = QdrantVectorSpace(
        qdrant_url=qdrant_url,
        token=token,
        collection_name=collection_name,
        similarity_metric="cosine",
        collection_type="image-text"
    )

    # Json file
    import json
    with open('example/metadata.json', 'r') as file:
        metadata_samples = json.load(file)

    # Add pairs to the vector space
    pairs = []
    for metadata in metadata_samples:
        image_embedding = [random.random() for _ in range(512)]  # Random vector for image
        text_embedding = [random.random() for _ in range(512)]   # Random vector for text

        # Construct the pair
        pair = {
            'image_embedding': image_embedding,
            'text_embedding': text_embedding,
            'metadata': metadata
        }
        pairs.append(pair)
    vector_space.add(pairs)

    vector_space.create_snapshot()

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