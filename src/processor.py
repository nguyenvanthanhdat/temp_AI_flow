from src.modeling import VLLM
from src.embeding import RetrieveModel


class Processor:
    def __init__(
        self,
        llm_model_path,
        embd_model_path,
        translate_path,
    ):
        self.llm_model = VLLM(llm_model_path, llm_model_path)
        self.retrieve_model = RetrieveModel(embd_model_path)
        
    def init_db(self):
        """ Reload if find db else create new db
        
        """
        pass 
    
    def process(self):
        """Handling process query pipeline
        
        """
        pass
    
    def postprocess(self):
        """ Hanndling 

        """
        pass
    
    def to_csv(self, saved_path):
        pass
    
if __name__ == '__main__':
    processor = Processor(
        llm_model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        embd_model_path="jinaai/jina-clip-v1",
    )