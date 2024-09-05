import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from transformers import AutoModelForSequenceClassification


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RerankerModel:
    def __init__(
        self,
        model_name_or_path='jinaai/jina-reranker-v2-base-multilingual',
        device = None
    ):
        self.device = DEVICE if device is None else device
        # load model
        self.load_model(model_name_or_path)
        
        
    def load_model(self, model_name_or_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype="auto", trust_remote_code=True)
        model.eval()
        model = model.to(self.device)
        self.model = model
        
    def inference(self, queries, docs, max_length=1024):
        if len(queries) != len(docs):
            sentence_pairs = [[queries, doc] for doc in docs]
        else:
            sentence_pairs = [[query, doc] for query, doc in zip(queries, docs)]
            
        scores = self.model.compute_score(sentence_pairs, max_length=max_length)
        
        return scores
    
    
    
if __name__ == '__main__':
    model = RerankerModel(device=torch.device("cuda:7"))
    query = "Organic skincare products for sensitive skin"

    docs = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
        "针对敏感肌专门设计的天然有机护肤产品",
        "新的化妆趋势注重鲜艳的颜色和创新的技巧",
        "敏感肌のために特別に設計された天然有機スキンケア製品",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
    ]
    
    result = model.inference(query, docs, max_length=128)
    print("\nResult: ", result, "\n")
    