import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import nltk

# ================== НАСТРОЙКИ ==================
CORPUS_DIR = "corpus"        # Должен совпадать с путем в download_and_build_corpus.py
OUT_DIR = "embeddings"

MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "e5": "intfloat/e5-base-v2"
}

CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
MIN_SENT_LEN = 20
# =================================================

def build_embeddings():
    """
    Строит эмбеддинги для предложений и чанков, используя LaBSE и E5.
    Сохраняет результаты в поддиректории OUT_DIR/{model_name}/
    """
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    os.makedirs(OUT_DIR, exist_ok=True)

    sentence_texts = []
    sentence_meta = []
    chunk_texts = []
    chunk_meta = []

    print("📚 Reading corpus...")

    for root, _, files in os.walk(CORPUS_DIR):
        for fname in files:
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(root, fname)
            rel_path = os.path.relpath(path, CORPUS_DIR)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            sentences = [
                s.strip() for s in sent_tokenize(text)
                if len(s.strip()) >= MIN_SENT_LEN
            ]

            for i, s in enumerate(sentences):
                sentence_texts.append(s)
                sentence_meta.append({
                    "text_id": rel_path,
                    "sentence_id": i,
                    "text": s
                })

            step = CHUNK_SIZE - CHUNK_OVERLAP
            for i in range(0, len(sentences) - CHUNK_SIZE + 1, step):
                chunk = sentences[i:i + CHUNK_SIZE]
                chunk_text = " ".join(chunk)

                chunk_texts.append(chunk_text)
                chunk_meta.append({
                    "text_id": rel_path,
                    "chunk_id": i // step,
                    "start_sentence": i,
                    "end_sentence": i + CHUNK_SIZE - 1,
                    "text": chunk_text
                })

    print(f"🔢 Sentences: {len(sentence_texts)}")
    print(f"🔢 Chunks: {len(chunk_texts)}")

    if not sentence_texts:
        print("⚠️  No sentences found. Check your corpus directory.")
        return

    # Обработка каждой модели
    for model_name, model_id in MODELS.items():
        print(f"\n🧠 Loading model: {model_name} ({model_id})")
        model = SentenceTransformer(model_id)

        print(f"🧠 Encoding sentences for {model_name}...")
        sentence_embeddings = model.encode(
            sentence_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"🧠 Encoding chunks for {model_name}...")
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        model_dir = os.path.join(OUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        np.save(os.path.join(model_dir, "sentence_embeddings.npy"), sentence_embeddings)
        np.save(os.path.join(model_dir, "chunk_embeddings.npy"), chunk_embeddings)

        with open(os.path.join(model_dir, "sentence_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(sentence_meta, f, ensure_ascii=False, indent=2)

        with open(os.path.join(model_dir, "chunk_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, ensure_ascii=False, indent=2)

        print(f"✅ {model_name} done")
        print(f"  Sentence embeddings shape: {sentence_embeddings.shape}")
        print(f"  Chunk embeddings shape: {chunk_embeddings.shape}")

    print("\n🎉 All models processed successfully!")


if __name__ == "__main__":
    build_embeddings()