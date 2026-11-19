"""Frozen MiniLM encoding, resumable deduplicated cache, symmetric pair features."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = ROOT / "models" / "sbert_encoder"


def load_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Download the encoder first using --download in src.sbert_features")
    model = SentenceTransformer(str(MODEL_DIR), device=device, local_files_only=True)
    model.eval()
    if model.get_embedding_dimension() != 384:
        raise ValueError("Expected 384-dimensional MiniLM embeddings")
    return model


def encode_batches(model, sentences, batch_size=64):
    """Reduce batch size and retry the same slice on CUDA OOM."""
    output = np.empty((len(sentences), 384), dtype=np.float32)
    start = 0
    while start < len(sentences):
        end = min(start + batch_size, len(sentences))
        try:
            output[start:end] = model.encode(sentences[start:end], batch_size=batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            print(f"CUDA OOM: retrying with batch size {batch_size}", flush=True)
            continue
        start = end
    return output


def cached_embeddings(model, sentences, identity, cache_dir=None):
    """One row per unique exact text; cache key includes ordered text and encoder config.

    Chunks are marked complete only after flushing. Interrupted runs resume at the
    last completed chunk. No labels or validation statistics enter the encoder.
    """
    cache_dir = Path(cache_dir or ROOT / "models" / "sbert_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(sentences))
    lookup = {text: i for i, text in enumerate(unique)}
    digest = hashlib.sha256(json.dumps([identity, unique], ensure_ascii=False).encode()).hexdigest()
    array_path, state_path = cache_dir / f"{digest}.npy", cache_dir / f"{digest}.json"
    completed = 0
    if array_path.exists() and state_path.exists():
        embeddings = np.load(array_path, mmap_mode="r+")
        completed = json.loads(state_path.read_text())["completed"]
        if embeddings.shape != (len(unique), 384):
            raise ValueError("Cached embedding dimensions do not match")
    else:
        embeddings = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=(len(unique), 384))
    for start in range(completed, len(unique), 2048):
        end = min(start + 2048, len(unique))
        embeddings[start:end] = encode_batches(model, unique[start:end])
        embeddings.flush()
        state_path.write_text(json.dumps({"completed": end, "unique_sentences": len(unique), "identity": identity}))
        print(f"Encoded {end:,}/{len(unique):,} unique sentences", flush=True)
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
    return embeddings, lookup, {"unique_sentences": len(unique), "cache": str(array_path), "key": digest}


def pair_features(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 384:
        raise ValueError("Expected equal (n, 384) embeddings")
    cosine = np.clip(np.sum(a * b, axis=1, keepdims=True), -1, 1)
    return np.concatenate((np.abs(a - b), a * b, cosine), axis=1)


if __name__ == "__main__":
    import argparse
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        revision = HfApi().model_info(MODEL_ID).sha
        model = SentenceTransformer(MODEL_ID, revision=revision, device="cpu",
                                    cache_folder=str(ROOT / "models" / "sbert_download_cache"))
        model.save_pretrained(str(MODEL_DIR))
        (MODEL_DIR / "source_revision.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": revision}, indent=2))
        print(f"Saved encoder at revision {revision}")
