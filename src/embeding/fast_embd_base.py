import numpy as np
from typing import Iterable, Type, List, Dict, Any, Optional, Sequence

from fastembed.common.utils import normalize
from fastembed.common.utils import define_cache_dir, normalize
from fastembed.common import ImageInput, OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext

from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker
from fastembed.text.pooled_embedding import PooledEmbedding

from fastembed.image.onnx_image_model import ImageEmbeddingWorker
from fastembed.image.onnx_embedding import OnnxImageEmbeddingWorker, OnnxImageEmbedding
from fastembed.image.transform.functional import normalize



supported_models = [
    {
        "model": "jina-image",
        "dim": 768,
        "description": "Text embeddings, MultiModel (text), English, 8192 input tokens truncation",
        "size_in_GB": 0.52,
        "sources": {"hf": "tiennv/AIC-jina-clip-vit"},
        "model_file": "vision_model.onnx",
    },
    {
        "model": "jina-text",
        "dim": 768,
        "description": "Image embeddings, MultiModel (image), English, 8192 input tokens truncation",
        "size_in_GB": 0.52,
        "sources": {"hf": "jinaai/jina-clip-v1"},
        "model_file": "onnx/text_model.onnx",
    },
]

class JinaTextEmbedding(PooledEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return JinaTextEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext
    ) -> Iterable[np.ndarray]:
        embeddings = output.model_output
        attn_mask = output.attention_mask
        return normalize(self.mean_pooling(embeddings, attn_mask)).astype(np.float32)

class JinaTextEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self, model_name: str, cache_dir: str, **kwargs
    ) -> OnnxTextEmbedding:
        return JinaTextEmbedding(
            model_name=model_name, cache_dir=cache_dir, threads=1, **kwargs
        )

class JinaImageEmbedding(OnnxImageEmbedding):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """

        super().__init__(model_name, cache_dir, threads, **kwargs)

        model_description = self._get_model_description(model_name)
        self.cache_dir = define_cache_dir(cache_dir)
        model_dir = self.download_model(
            model_description, self.cache_dir, local_files_only=self._local_files_only
        )

        self.load_onnx_model(
            model_dir=model_dir,
            model_file=model_description["model_file"],
            threads=threads,
            providers=providers,
        )

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_models

    def embed(
        self,
        images: ImageInput,
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        yield from self._embed_images(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            images=images,
            batch_size=batch_size,
            parallel=parallel,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker"]:
        return JinaImageEmbeddingWorker

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """

        return onnx_input

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        return normalize(output.model_output).astype(np.float32)

class JinaImageEmbeddingWorker(OnnxImageEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> JinaImageEmbedding:
        return JinaImageEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1, **kwargs)