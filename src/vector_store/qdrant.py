from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from typing import Any, Dict, List
from qdrant_client.http import models  

import logging
import uuid
import random

logger = logging.getLogger(__name__)

"""
    {
        image_embedding: [0.1, 0.2, 0.3, ...],
        text_embedding: [0.4, 0.5, 0.6, ...],
        metadata: {
            video_id: "video_1",
            keyframe_id: "keyframe_1"
            image: Image.open("image.jpg"),
            text: "A cat is sitting on a table."
        }
    }
"""

# Map similarity metric to Qdrant distance metric
SIMILARITY_METRIC_MAP = {
    "dot": "Dot",
    "manhattan": "Manhattan",
    "euclidean": "Euclid",
    "cosine": "Cosine"
}
        
class QdrantVectorSpace:
    def __init__(
            self, 
            qdrant_url: str, 
            token: str = "",
            collection_name: str = "collection",
            similarity_metric: str = "IP",
            consistency_level: str = "Strong",
            overwrite: bool = False,
            collection_type: str = "image-text",
            image_dim: int = 512,
            text_dim: int = 512
        ) -> None:

        self.collection_name = collection_name
        self.consistency_level = consistency_level
        self.collection_type = collection_type
        self.similarity_metric = SIMILARITY_METRIC_MAP.get(similarity_metric.lower(), "IP")

        # Initialize Qdrant client and collection
        if collection_type == "text":
            vector_params = VectorParams(size=text_dim, distance=self.similarity_metric)
        elif collection_type == "image":
            vector_params = VectorParams(size=image_dim, distance=self.similarity_metric)
        elif collection_type == "image-text":
            vector_params = {
                "image": VectorParams(size=image_dim, distance=self.similarity_metric),
                "text": VectorParams(size=text_dim, distance=self.similarity_metric),
            }

        # self.client = QdrantClient(url=qdrant_url, api_key=token)
        self.client = QdrantClient(":memory:")
        if overwrite:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params
            )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params
            )

    def add(self, pairs: List[Dict[str, Any]], **add_kwargs: Any) -> List[str]:
        points = []
        for pair in pairs:
            # Extract embeddings and metadata
            image_embedding = pair.get('image_embedding')
            text_embedding = pair.get('text_embedding')
            metadata = pair.get('metadata', {})

            video_id = metadata.get('video_id', '')
            keyframe_id = metadata.get('keyframe_id', '')
            idx = self.generate_uuid(video_id, keyframe_id)

            # Ensure both image and text embeddings are provided
            if image_embedding is None or text_embedding is None:
                raise ValueError("Each pair must contain both 'image_embedding' and 'text_embedding'.")

            # Set up vector search
            if self.collection_type == "image-text":
                vector_search = {
                    "image": image_embedding,
                    "text": text_embedding
                }
            elif self.collection_type == "image":
                vector_search = {
                    "image": image_embedding
                }
            elif self.collection_type == "text":
                vector_search = {
                    "text": text_embedding
                }

            point = models.PointStruct(
                id=idx,
                vector=vector_search,
                payload=metadata
            )
            points.append(point)

        # Upload points into Qdrant collection
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

    def delete(self, video_id: str, keyframe_id: str, **delete_kwargs: Any) -> None:
        """
            Delete a point from the Qdrant collection based on video_id and keyframe_id.
        """
        # Delete the point from Qdrant collection
        idx = self.generate_uuid(video_id, keyframe_id)
        self.client.delete(
            collection_name=self.collection_name,
            point_ids=[idx]
        )

    def query_by_image(self, query_vector: List[float], **kwargs: Any) -> List[Dict[str, str]]:
        """
            Query the Qdrant collection based on the input query (image).
        """
        # Parameters
        top_k = kwargs.get('top_k', 10)  # Number of top results to return
        
        # Perform the query
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=("text", query_vector),
            with_payload=True,
            limit=top_k
        )

    def query_by_text(self, query_vector: List[float], **kwargs: Any) -> List[Dict[str, str]]:
        """
            Query the Qdrant collection based on the input query (vector).
        """
        # Parameters
        top_k = kwargs.get('top_k', 10)  # Number of top results to return
        
        # Perform the query
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=("image", query_vector),
            with_payload=True,
            limit=top_k
        )
    
    def generate_uuid(self, video_id: str, keyframe_id: str) -> str:
        combined_string = f"{video_id}_{keyframe_id}"
        
        # Generate a UUID4 based on the combined string
        hash_value = uuid.uuid5(uuid.NAMESPACE_DNS, combined_string)
        
        return str(hash_value)

def main():
    qdrant_url = "http://localhost:6333"
    token = ""
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

    # Example query vectors
    image_vector = [random.random() for _ in range(512)]  # Image vector with 512 dimensions
    text_vector = [random.random() for _ in range(512)]  # Text vector with 768 dimensions

    # Perform queries
    image_results = vector_space.query_by_image(image_vector, top_k=2)
    text_results = vector_space.query_by_text(text_vector, top_k=3)

    # Print results
    print("Image Query Results:")
    for result in image_results:
        print(result)

    print("\nText Query Results:")
    for result in text_results:
        print(result)

if __name__ == "__main__":
    main()