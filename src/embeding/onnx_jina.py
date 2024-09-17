import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import base64
import requests
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image

from io import BytesIO
from embeding.embed_utils import normalize

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class JinaTextEmbeding:
    def __init__(
        self,
        model_name_or_path:str = None,
    ):
        self.load_model(model_name_or_path)
        
    def load_model(self, model_name_or_path, provider=["CUDAExecutionProvider","CPUExecutionProvider"]):
        _model_name_or_path = os.path.join(model_name_or_path, 'onnx/text_model.onnx')
        self.model = ort.InferenceSession(
            _model_name_or_path,
            sess_options=sess_options,
            providers=provider
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        self.inputs = [_inp.name for _inp in self.model.get_inputs()]
        self.outputs = [_opt.name for _opt in self.model.get_outputs()]
        
    def inference(self, x, norm_embeds:bool=True):
        inp_tensor = self.tokenizer(x, padding=True, return_tensors='np')['input_ids']
        inp_tensor = np.expand_dims(inp_tensor, axis=0)
        embeddings = self.model.run(self.outputs, dict(zip(self.inputs, inp_tensor)))
        if norm_embeds:
            embeddings = normalize(embeddings)
        
        return embeddings


class JinaImageEmbeding:
    def __init__(
        self,
        model_name_or_path:str = None,
    ):
        self.load_model(model_name_or_path)
        
    def load_model(self, model_name_or_path, provider=["CUDAExecutionProvider","CPUExecutionProvider"]):
        _model_name_or_path = os.path.join(model_name_or_path, 'onnx/vision_model.onnx')
        self.model = ort.InferenceSession(
            _model_name_or_path,
            sess_options=sess_options,
            providers=provider
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        self.inputs = [_inp.name for _inp in self.model.get_inputs()]
        self.outputs = [_opt.name for _opt in self.model.get_outputs()]
        
    def inference(self, examples, norm_embeds:bool=True):
        process_inp = self.preprocess(examples)
        inp_tensor = self.processor(process_inp, return_tensors='np')['pixel_values']
        inp_tensor = np.expand_dims(inp_tensor, axis=0)
        embeddings = self.model.run(self.outputs, dict(zip(self.inputs, inp_tensor)))[0]
        if norm_embeds:
            embeddings = normalize(embeddings)
        
        return embeddings
    
    def preprocess(self, examples):
        processed_inputs = []
        for img in examples:
            if isinstance(img, str):
                if img.startswith('http'):
                    response = requests.get(img)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                elif img.startswith('data:image/'):
                    image = self.decode_data_image(img).convert('RGB')
                else:
                    image = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                image = img.convert('RGB')
            else:
                raise ValueError("Unsupported image format")

            processed_inputs.append(image)
            
        return processed_inputs
            
    def decode_data_image(self, data_image_str):
        _, data = data_image_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))

        
# if __name__ == '__main__':
#     from PIL import Image
    
#     # Image Embeding
#     images = [
#         Image.open('sample/joy.jpg'),
#         Image.open('output/L03_V001/L03_V001_00005.jpg'),
#         Image.open('output/L03_V001/L03_V001_00180.jpg'),
#         Image.open('output/L03_V001/L03_V001_00280.jpg'),
#         Image.open('output/L03_V001/L03_V001_00430.jpg'),
#     ]
#     model = JinaImageEmbeding('weights/jina-clip-v1')
#     result = model.inference(images, norm_embeds=True)
#     print(result)
    
#     print("\n\n")
#     # Text embeding
#     text = [
#         'Hello all my friend',
#         "This is my test",
#         "Run multiprocessing pipeline"
#     ]

#     model = JinaTextEmbeding('weights/jina-clip-v1')
#     result = model.inference(text, norm_embeds=True)
#     print(result)