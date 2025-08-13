import os, json, argparse
import numpy as np
import faiss
from PIL import Image
from .feature_extractor import CLIPEmbedder

class RagIndex:
    def __init__(self, index_path="data/index.faiss", meta_path="data/meta.json"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.meta = []
        self.index = None
        self.dim = None
        self._load()

    def _load(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.index_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.index = faiss.read_index(self.index_path)
            if self.index.ntotal > 0:
                self.dim = self.index.d
        else:
            self.meta = []
            self.index = None
            self.dim = None

    def save(self):
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def rebuild(self, sample_dir="data/samples"):
        emb_model = CLIPEmbedder()
        embs = []
        self.meta = []
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir, exist_ok=True)
        for name in os.listdir(sample_dir):
            if name.lower().endswith((".jpg",".jpeg",".png",".webp")):
                path = os.path.join(sample_dir, name)
                img = Image.open(path).convert("RGB")
                emb = emb_model.encode(img)
                embs.append(emb)
                self.meta.append({"path": path, "caption": f"Sample image {name}", "labels": []})
                print("Indexed:", name)

        if len(embs) == 0:
            print("No images found to index.")
            self.index = None
            self.dim = None
            self.save()
            return

        mat = np.vstack(embs).astype("float32")
        self.dim = mat.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # cosine if normalized
        self.index.add(mat)
        self.save()
        print(f"Saved FAISS index to {self.index_path} with {self.index.ntotal} items.")

    def search(self, query_emb: np.ndarray, k=3):
        if self.index is None or self.index.ntotal == 0 or k <= 0:
            return []
        query = query_emb.reshape(1,-1).astype("float32")
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            results.append(self.meta[int(idx)])
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--sample_dir", default="data/samples")
    parser.add_argument("--index_path", default="data/index.faiss")
    parser.add_argument("--meta_path", default="data/meta.json")
    args = parser.parse_args()

    ri = RagIndex(index_path=args.index_path, meta_path=args.meta_path)
    if args.rebuild:
        ri.rebuild(sample_dir=args.sample_dir)
        print("Done.")
    else:
        print("Nothing to do. Use --rebuild to build the index.")
