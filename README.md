# Image Captioning + Segmentation 

This is an **upgraded** version that uses real models:
- **Captioning:** BLIP (Salesforce/blip-image-captioning-base) via 🤗 Transformers
- **Segmentation:** TorchVision **Mask R-CNN** (ResNet50-FPN) pre-trained on COCO
- **RAG:** CLIP (clip-ViT-B-32) embeddings via sentence-transformers + FAISS index

## Quickstart

### 1) Create venv & install deps
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

> If PyTorch fails: visit https://pytorch.org/get-started/locally/ for the right command (CUDA vs CPU).

### 2) Build the RAG index 
Put a few sample images into `data/samples/` then run:
```bash
python utils/rag.py --rebuild
```

### 3) Launch the app
```bash
streamlit run app.py
```

### 4) In the UI
- Upload an image.
- Toggle **Use RAG** to compare “with vs without retrieval context”.
- You’ll see: **caption** (from BLIP), **segmentation** (Mask R-CNN), **neighbors** used for context.

## Notes
- Everything runs on **CPU** by default; if you have CUDA, PyTorch will use it automatically.
- First run will download model weights (one-time cache).

## Structure
- `app.py` – Streamlit UI.
- `utils/feature_extractor.py` – CLIP embeddings (sentence-transformers).
- `utils/caption_model.py` – BLIP captioner.
- `utils/segmenter.py` – TorchVision Mask R-CNN, drawing utilities.
- `utils/rag.py` – FAISS index builder + search.
- `requirements.txt` – dependencies.
