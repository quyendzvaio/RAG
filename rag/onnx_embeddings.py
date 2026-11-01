"""Embedding sentence-transformers chay bang ONNX Runtime tren CPU.

Model duoc export mot lan tu sentence-transformers (can torch), toi uu do thi (fuse attention/layernorm) va cache
trong `EMBEDDING_ONNX_DIR/<model>/` cung tokenizer va `embedding.json`; lan sau chi can onnxruntime + tokenizer.
Vector giong ban PyTorch (cosine 1.0000 tren golden), nen dung chung index voi `EMBEDDING_RUNTIME=torch`.
Khong luong tu hoa int8: voi vietnamese-bi-encoder (PhoBERT) tren CPU AVX2, cosine voi ban fp32 chi con 0.71-0.84.
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)

MODEL_FILE = "model.onnx"
META_FILE = "embedding.json"
ONNX_OPSET = 17
POOLING_MODES = ("mean", "cls")


class OnnxEmbeddings(Embeddings):
    def __init__(self, model_name: str, onnx_dir: str, batch_size: int):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        directory = Path(onnx_dir) / model_name.replace("/", "--")
        if not (directory / META_FILE).exists():
            export_onnx(model_name, directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory))
        self.max_seq_length = int(meta["max_seq_length"])
        self.pooling = meta["pooling"]
        self.batch_size = batch_size
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # intra_op_num_threads de 0: onnxruntime dung so nhan vat ly (4 luong nhanh hon 8 tren may 4 nhan / 8 luong).
        self.session = ort.InferenceSession(str(directory / MODEL_FILE), options, providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]
        logger.info("Embedding ONNX Runtime: %s (%s pooling, max %d token)", directory, self.pooling, self.max_seq_length)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Giong HuggingFaceEmbeddings + sentence-transformers: thay xuong dong bang dau cach, cat khoang trang hai dau.
        # Sap theo do dai de moi lo it padding.
        texts = [text.replace("\n", " ").strip() for text in texts]
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        vectors: List[List[float]] = [[] for _ in texts]
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            for index, vector in zip(batch, self._encode([texts[index] for index in batch])):
                vectors[index] = vector.tolist()
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _encode(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_seq_length, return_tensors="np")
        hidden = self.session.run(None, {name: encoded[name].astype(np.int64, copy=False) for name in self.input_names})[0]
        if self.pooling == "cls":
            vectors = hidden[:, 0]
        else:
            mask = encoded["attention_mask"][..., None].astype(np.float32)
            vectors = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def export_onnx(model_name: str, directory: Path) -> None:
    """Export model sentence-transformers (Transformer + Pooling) sang ONNX da toi uu, ghi nguyen tu vao `directory`."""
    import torch
    from onnxruntime.transformers.optimizer import optimize_model
    from sentence_transformers import SentenceTransformer

    logger.info("Dang export %s sang ONNX (chi lam mot lan)...", model_name)
    model = SentenceTransformer(model_name, device="cpu")
    modules = [type(module).__name__ for module in model]
    pooling = model[1].get_pooling_mode_str() if len(model) > 1 else ""
    if modules[:2] != ["Transformer", "Pooling"] or any(name != "Normalize" for name in modules[2:]) \
            or pooling not in POOLING_MODES:
        raise ValueError(f"Khong export duoc {model_name}: can Transformer + Pooling {POOLING_MODES}, co {modules} {pooling}")

    transformer = model[0].auto_model.eval()
    sample = model.tokenizer(["xin chào", "một câu dài hơn một chút"], padding=True, return_tensors="pt")
    directory.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=directory.parent))
    try:
        raw = work / "raw.onnx"
        with torch.no_grad():
            torch.onnx.export(
                transformer, (sample["input_ids"], sample["attention_mask"]), str(raw),
                input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                dynamic_axes={name: {0: "batch", 1: "sequence"} for name in ("input_ids", "attention_mask", "last_hidden_state")},
                opset_version=ONNX_OPSET,
            )
        config = transformer.config
        optimized = optimize_model(str(raw), model_type="bert", num_heads=config.num_attention_heads,
                                   hidden_size=config.hidden_size, opt_level=1)
        optimized.save_model_to_file(str(work / MODEL_FILE))
        raw.unlink()
        model.tokenizer.save_pretrained(str(work))
        meta = {"model": model_name, "pooling": pooling, "max_seq_length": model.max_seq_length,
                "dimensions": model.get_sentence_embedding_dimension(), "opset": ONNX_OPSET}
        (work / META_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(work, directory)
        directory.chmod(0o755)  # mkdtemp tao thu muc 0700
    finally:
        shutil.rmtree(work, ignore_errors=True)
    logger.info("Da export ONNX vao %s", directory)
