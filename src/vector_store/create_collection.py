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
    with open('example/json/sample.json', 'r') as file:
        sample = json.load(file)

    # Add pairs to the vector space
    pairs = []
    for each in sample:
        image_embedding = [random.random() for _ in range(512)]  # Random vector for image
        text_embedding = [random.random() for _ in range(512)]   # Random vector for text

        # Construct the pair
        pair = {
            'video_id': each['video_id'],
            'frame_id': each['frame_id'],
            'image_embedding': image_embedding,
            'text_embedding': text_embedding
        }
        pairs.append(pair)
    vector_space.add_pairs(pairs)

    vector_space.create_snapshot('example/snapshot')

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