import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from transformers import AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RetrieveModel:
    def __init__(
        self,
        model_name_or_path='jinaai/jina-clip-v1',
        device = None
    ):
        self.device = DEVICE if device is None else device
        # load model
        self.load_model(model_name_or_path)
        
        
    def load_model(self, model_name_or_path):
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype="auto", trust_remote_code=True)
        model.eval()
        model = model.to(self.device)
        self.model = model
        
    def inference(self, text=None, image=None):
        # text encoder
        text_embd = None
        if text is not None:
            text_embd = self.model.encode_text(text)
        
        # image encoder
        img_embd = None
        if image is not None:
            img_embd = self.model.encode_image(image)
        return text_embd, img_embd
    
    def get_token_len(self, ):
        

if __name__ == '__main__':
    model = RetrieveModel(device=torch.device("cuda:7"))

    text_embd, img_embd = model.inference(
        text=["Organic skincare products for sensitive skin", "Organic skincare products for sensitive skin"], 
        image= 'sample/joy.jpg'
    )
    print("\ntext_embd: ", text_embd.shape, "\n")
    print("\nimg_embd: ", img_embd.shape, "\n")