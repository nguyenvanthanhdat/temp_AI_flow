# import transformers
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info


class VLLM:
    def __init__(self, model_name, tokenizer_name):
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.processor = AutoProcessor.from_pretrained(tokenizer_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            # torch_dtype="auto", 
            # torch_dtype=torch.float16,
            torch_dtype=torch.float16, 
            # load_in_4bit=True,
            device_map="auto"
        )
        self.message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    # {"type": "image", "image": 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg'},
                    {"type": "text", "text": "Describe this image."},
                    # {"type": "text", "text": "What's in the picture? Please draw a picture with the same content and similar style. Replace women with men."}
                ],
            }
        ]
        self.functions = """
            {
                "name": "describe image",
                "description_for_model": "A model that generates a detailed description of the image. Format the arguments as a JSON object.",
                "parameters": {
                    "name": "description_the_image",
                    "type": "string",
                    "description": "Describe the image.",
                    "required": True,
                }
            },
            {
                "name": "get object in image",
                "description_for_model": "A model that generates a list of objects, animals, person in the image. Format the arguments as a JSON object.",
                "parameters": {
                    "name": "list_all_objects",
                    "type": "string",
                    "description": "Name all the objects in the image.",
                    "required": True,
                }
            }
        """
        # ]
        
    def predict(self, image):
        if image == None:
            # image = 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg'
            image = 'http://farm9.staticflickr.com/8453/8045989560_846b28104e_z.jpg'
        self.message[0]["content"][0]["image"] = image
        self.message[0]["content"][1]["text"] = str(self.functions)
        input_message = self.message
        text = self.processor.apply_chat_template(
            input_message, tokenize=False, add_generation_prompt=True
        )
        # print(text)
        # print(input_message)
        image_inputs, video_inputs = process_vision_info(input_message)
        inputs = self.processor(
            text = [text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
        
    
if __name__ == "__main__":
    # model_name = "Qwen/Qwen2-VL-2B-Instruct"
    # tokenizer_name = "Qwen/Qwen2-VL-2B-Instruct"
    # model_name = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
    # tokenizer_name = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
    # model_name = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8"
    # tokenizer_name = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8"
    model_name = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
    tokenizer_name = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
    vllm = VLLM(model_name, tokenizer_name)
    image = None
    output = vllm.predict(image)
    print(output[0])