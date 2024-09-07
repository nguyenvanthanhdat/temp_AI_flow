import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional, Sequence, Union, List, Dict, Any, Type, Iterable
from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_embedding import OnnxImageEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.common import OnnxProvider
from embeding import (
    JinaImageEmbeddingWorker,
    JinaImageEmbedding,
    JinaTextEmbeddingWorker,
    JinaTextEmbedding
)


class AICTextEmbedding(TextEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[TextEmbeddingBase]] = [
        OnnxTextEmbedding,
        JinaTextEmbedding,
        JinaTextEmbeddingWorker
    ]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        result = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding.list_supported_models())
        return result

    def __init__(
        self,
        model_name: str = None,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(
                model_name.lower() == model["model"].lower()
                for model in supported_models
            ):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextEmbedding."
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        yield from self.model.embed(documents, batch_size, parallel, **kwargs)
        
        
class AICImageEmbedding(ImageEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[ImageEmbeddingBase]] = [
        OnnxImageEmbedding,
        JinaImageEmbedding,
        JinaImageEmbeddingWorker
    ]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        result = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding.list_supported_models())
        return result

    def __init__(
        self,
        model_name: str = None,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(
                model_name.lower() == model["model"].lower()
                for model in supported_models
            ):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextEmbedding."
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        yield from self.model.embed(documents, batch_size, parallel, **kwargs)
        
        
if __name__ == '__main__':
    jina_embd = AICImageEmbedding(model_name='jina-image')
    jina_embd = AICTextEmbedding(model_name='jina-text')