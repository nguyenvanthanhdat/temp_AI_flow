from src.modeling import VLLM
from src.embeding import RetrieveModel
from src.vector_store import QdrantVectorSpace
import os
import gradio as gr


class Processor:
    def __init__(
        self,
        llm_model_path,
        embd_model_path,
        qdrant_url,
        token,
        collection_name,
        translate_path=None,
    ):
        self.llm_model = VLLM(llm_model_path, llm_model_path)
        self.retrieve_model = RetrieveModel(embd_model_path)
        self.translate_path = translate_path
        self.qdrant_url = qdrant_url
        self.token = token
        self.collection_name = collection_name
        # TODO: load gradio
        
    def init_db(self):
        """ Reload if find db else create new db
        
        """
        if not os.path.exists("db") or False:
            # TODO: create db image-text, image-embeding, text-embeding
            # os.mkdir("db")
            self.vector_space = QdrantVectorSpace(
                qdrant_url=self.qdrant_url,
                token=self.token,
                collection_name=self.collection_name,
                similarity_metric="cosine",
                collection_type="image-text"
            )
        else:
            # TODO: reload db image-text, image-embeding, text-embeding
            
            pass
        
        self.db = None
    
    def process(self, query):
        """Handling process query pipeline
        
        """
        
        text_retrieval, image_retrieval  = self.retrieve_model.inference(text=query)
        
        # TODO: retrieve the image through the image-embeding
        list_images_output = text_retrieval
        
        # TODO: retrieve the image through the text-embeding
        list_text_output = image_retrieval
        
        self.list_images_output = list_images_output
        self.list_text_output = list_text_output
    
    def postprocess(self):
        """ Hanndling 

        """
        
        results = []
        # TODO: The choose image will place into 1st index of results
        
        # TODO: Calculate the average from self.list_images_embeding and self.list_text_embeding
        
        # TODO: choose the top 99 images most highest into results
        
        self.results = results
    
    def to_csv(self, saved_path):
        
        # TODO: save the results into csv file
        
        pass
    
if __name__ == '__main__':
    processor = Processor(
        llm_model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        embd_model_path="jinaai/jina-clip-v1",
        qdrant_url = "http://localhost:6333",
        token = "",
        collection_name = "test_collection",
    )