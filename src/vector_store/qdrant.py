from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from typing import Any, Dict, List
from qdrant_client.http import models  

import logging
import uuid
import requests
import os

logger = logging.getLogger(__name__)

"""
    {
        video_id: "1",
        keyframe_id: "1",
        image_embedding: [0.1, 0.2, 0.3, ...],
        text_embedding: [0.4, 0.5, 0.6, ...],
    }
"""

class QdrantVectorSpace:
    def __init__(
            self, 
            qdrant_url: str, 
            token: str = "",
            collection_name: str = "collection",
            similarity_metric: str = "IP",
            consistency_level: str = "Strong",
            snapshot: bool = False,
            collection_type: str = "image-text",
            image_dim: int = 512,
            text_dim: int = 512
        ) -> None:

        # Initialize Qdrant client
        self.token = token
        self.qdrant_url = qdrant_url
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.token)
        self.collection_name = collection_name

        if not snapshot:
            self.consistency_level = consistency_level
            self.collection_type = collection_type

            # Map similarity metric to Qdrant distance metric
            similarity_metric_map = {
                "dot": "Dot",
                "manhattan": "Manhattan",
                "euclidean": "Euclid",
                "cosine": "Cosine"
            }
            self.similarity_metric = similarity_metric_map.get(similarity_metric.lower(), "IP")

            # Initialize Qdrant client and collection
            if collection_type == "text":
                vector_params = {
                    "text": VectorParams(size=text_dim, distance=self.similarity_metric)
                }
            elif collection_type == "image":
                vector_params = {
                    "image": VectorParams(size=image_dim, distance=self.similarity_metric)
                }
            elif collection_type == "image-text":
                vector_params = {
                    "image": VectorParams(size=image_dim, distance=self.similarity_metric),
                    "text": VectorParams(size=text_dim, distance=self.similarity_metric),
                }

            # Create the collection or loading old snapshot
            if self.client.collection_exists(collection_name=self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params
            )

    def index(self):
        """
            Index the Qdrant collection.
        """
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="id",
            field_schema=models.KeywordIndexParams(
                type="keyword",
                is_tenant=True,
            ),
        )

    def add(self, pairs: List[Dict[str, Any]], **add_kwargs: Any) -> List[str]:
        points = []
        for pair in pairs:
            # Extract embeddings and metadata
            image_embedding = pair.get('image_embedding')
            text_embedding = pair.get('text_embedding')

            video_id = pair.get('video_id', '')
            keyframe_id = pair.get('keyframe_id', '')

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
                payload={
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "id": f"{video_id}_{keyframe_id}"
                }
            )
            points.append(point)

        # Upload points into Qdrant collection
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

        self.index()

    # def delete(self, video_id: str, keyframe_id: str, **delete_kwargs: Any) -> None:
    #     """
    #         Delete a point from the Qdrant collection based on video_id and keyframe_id.
    #     """
    #     # Delete the point from Qdrant collection
    #     idx = self.generate_uuid(video_id, keyframe_id)
    #     self.client.delete(
    #         collection_name=self.collection_name,
    #         point_ids=[idx]
    #     )

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
    
    def create_snapshot(self, output_path: str = None) -> None:
        """
            Creates a snapshot of the current Qdrant collection and saves it to the specified path.
        """
        snapshot_info = self.client.create_snapshot(collection_name=self.collection_name)
        print(f"Snapshot created with the name {snapshot_info.name}")

        snapshot_url = f"{self.qdrant_url}/collections/{self.collection_name}/snapshots/{snapshot_info.name}"

        response = requests.get(snapshot_url, headers={"api-key": self.token})
        if output_path is None:
            output_path = "./"
        output_path = os.path.join(output_path, snapshot_info.name)

        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Snapshot saved to {output_path}")

    def load_snapshot(self, snapshot_path: str) -> None:
        """
            Loads a snapshot from the specified path into the current Qdrant collection.
        """
        url = f"{self.qdrant_url}/collections/{self.collection_name}/snapshots/upload"
    
        # Open the snapshot file in binary read mode
        with open(snapshot_path, 'rb') as snapshot_file:
            # Prepare the multipart-form data
            files = {
                'snapshot': (snapshot_path, snapshot_file, 'application/octet-stream')
            }
            
            headers = {
                'api-key': self.token
            }
            
            # Send the POST request
            response = requests.post(url, headers=headers, files=files)
            
            # Check the response
            if response.status_code == 200:
                print("Snapshot uploaded successfully.")
                print("Response:", response.json())
            else:
                print("Failed to upload snapshot.")
                print("Status Code:", response.status_code)
                print("Response:", response.text)

    def generate_uuid(self, video_id: str, keyframe_id: str) -> str:
        combined_string = f"{video_id}_{keyframe_id}"
        
        # Generate a UUID4 based on the combined string
        hash_value = uuid.uuid5(uuid.NAMESPACE_DNS, combined_string)
        return str(hash_value)