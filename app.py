import streamlit as st
from PIL import Image
import numpy as np
import os

from utils.feature_extractor import CLIPEmbedder
from utils.caption_model import BLIPCaptioner
from utils.segmenter import Segmenter, draw_instance_predictions
from utils.rag import RagIndex

st.set_page_config(page_title="Caption + Segmentation + RAG (Pro)", layout="wide")
st.title("🖼️ Caption + Segmentation + RAG (Pro)")

with st.sidebar:
    st.header("Options")
    use_rag = st.toggle("Use RAG (retrieval context)", value=True)
    k = st.slider("Neighbors (k)", 0, 10, 3, help="Number of nearest neighbors to retrieve")
    score_thresh = st.slider("Segmentation score threshold", 0.0, 1.0, 0.5, 0.05)
    max_len = st.slider("Max caption length", 8, 64, 30, 2)

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])

@st.cache_resource
def _get_embedder():
    return CLIPEmbedder()

@st.cache_resource
def _get_captioner():
    return BLIPCaptioner()

@st.cache_resource
def _get_segmenter(score_thresh: float):
    return Segmenter(score_thresh=score_thresh)

@st.cache_resource
def _get_index():
    return RagIndex(index_path="data/index.faiss", meta_path="data/meta.json")

col1, col2 = st.columns([3,2])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # Build context via RAG
    rag_context = ""
    neighbors = []
    if use_rag and k > 0:
        try:
            emb = _get_embedder().encode(image)
            neighbors = _get_index().search(emb, k=k)
            ctx_parts = []
            for nb in neighbors:
                cap = nb.get("caption","").strip()
                labels = ", ".join(nb.get("labels", []))
                if cap:
                    ctx_parts.append(f"[NN caption] {cap}")
                if labels:
                    ctx_parts.append(f"[NN labels] {labels}")
            rag_context = "\n".join(ctx_parts)
        except Exception as e:
            st.warning(f"RAG issue: {e}")

    # Caption
    try:
        caption = _get_captioner().generate(image, context=rag_context, max_length=max_len)
    except Exception as e:
        caption = f"(Captioner error: {e})"

    # Segmentation
    try:
        segmenter = _get_segmenter(score_thresh)
        pred = segmenter.predict(image)
        seg_viz = draw_instance_predictions(image, pred)
    except Exception as e:
        seg_viz = f"(Segmentation error: {e})"

    with col2:
        st.subheader("Outputs")
        st.markdown("**Caption:** " + str(caption))
        st.subheader("Segmentation")
        st.image(seg_viz, use_container_width=True)

        st.subheader("RAG Context")
        if rag_context:
            st.code(rag_context)
        else:
            st.write("(no neighbors used)")

    if neighbors:
        st.subheader("Nearest Neighbors")
        ncols = st.columns(min(5, len(neighbors)))
        for i, nb in enumerate(neighbors[:5]):
            with ncols[i]:
                p = nb.get("path","")
                cap = nb.get("caption","")
                if os.path.exists(p):
                    st.image(p, caption=cap, use_container_width=True)
else:
    st.info("Upload an image to get started. Tip: add a few images to data/samples and run `python utils/rag.py --rebuild` to enable RAG.")
