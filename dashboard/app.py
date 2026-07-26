"""
Proof of Concept Dashboard — ResNet-18 vs. Vision Transformer (ViT-B/16)
CIFAR-10 Image Classification

Streamlit app comparing a classical CNN baseline against a Vision Transformer
on an uploaded image, and presenting training results from the notebook.
"""

import json
import os
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import timm
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ViT vs. ResNet-18 — Proof of Concept",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plt_colorize(gray: np.ndarray) -> np.ndarray:
    """Apply viridis colormap to a [0,1] grayscale array → RGB [0,1]."""
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    return cmap(gray)[:, :, :3]

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #8899aa; margin-top: 0.2rem; }
    .model-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2d3250;
        margin-bottom: 0.5rem;
    }
    .highlight { color: #ff6b6b; font-weight: 700; }
    .badge-new {
        background: #ff6b6b22;
        color: #ff6b6b;
        border: 1px solid #ff6b6b55;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-baseline {
        background: #4a9eff22;
        color: #4a9eff;
        border: 1px solid #4a9eff55;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ResNet-18…")
def load_resnet():
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(DEVICE)
    return model


@st.cache_resource(show_spinner="Loading ViT-B/16…")
def load_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    # Disable fused (scaled_dot_product_attention) attention so the attn_drop
    # forward hooks receive the attention matrices — required for the
    # attention rollout visualization (needs all 12 blocks).
    for block in model.blocks:
        block.attn.fused_attn = False
    model.eval()
    model.to(DEVICE)
    return model


@st.cache_data
def load_results():
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ImageNet top-1000 label index → human-readable name (abbreviated list for common animals/vehicles)
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)

@st.cache_data(show_spinner=False)
def load_imagenet_labels():
    try:
        import urllib.request
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def predict(model, tensor: torch.Tensor, top_k: int = 5):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = probs.topk(top_k)
    return top_probs.cpu().numpy(), top_indices.cpu().numpy()


# ─────────────────────────────────────────────
# Attention map extraction (ViT)
# ─────────────────────────────────────────────

def get_vit_attention_map(model, tensor: torch.Tensor) -> np.ndarray:
    """
    Attention rollout (Abnar & Zuidema, 2020): propagate attention through
    all transformer blocks instead of reading a single layer. This avoids
    the last-layer artifact where attention collapses onto a few
    uninformative background patches.
    Returns a 14×14 map of how much the [CLS] token attends to each patch.
    """
    attentions = []

    def hook_fn(module, input, output):
        # output shape: (B, num_heads, seq_len, seq_len)
        attentions.append(output.detach().cpu())

    handles = [
        block.attn.attn_drop.register_forward_hook(hook_fn)
        for block in model.blocks
    ]

    try:
        with torch.no_grad():
            _ = model(tensor.unsqueeze(0).to(DEVICE))
    finally:
        for h in handles:
            h.remove()

    if not attentions:
        return None

    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len)
    for layer_attn in attentions:
        attn = layer_attn[0].mean(0)               # average heads → (197, 197)
        attn = attn + torch.eye(seq_len)           # add residual connection
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout

    cls_attn = rollout[0, 1:].reshape(14, 14).numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


# ─────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────

def bar_chart(labels, probs, color, title):
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=labels,
        orientation="h",
        marker_color=color,
        marker_line_color="rgba(0,0,0,0)",
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="white")),
        xaxis=dict(range=[0, 110], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(color="white", size=12)),
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="white"),
        margin=dict(l=10, r=50, t=40, b=10),
        height=240,
    )
    return fig


def accuracy_bar(resnet_acc, vit_acc):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ResNet-18 (Baseline)",
        x=["ResNet-18"],
        y=[resnet_acc],
        marker_color="#4a9eff",
        text=[f"{resnet_acc:.2f}%"],
        textposition="outside",
        width=0.4,
    ))
    fig.add_trace(go.Bar(
        name="ViT-B/16 (New)",
        x=["ViT-B/16"],
        y=[vit_acc],
        marker_color="#ff6b6b",
        text=[f"{vit_acc:.2f}%"],
        textposition="outside",
        width=0.4,
    ))
    fig.update_layout(
        title=dict(text="Final Test Accuracy on CIFAR-10", font=dict(size=15, color="white")),
        yaxis=dict(range=[min(resnet_acc, vit_acc) - 5, 100],
                   ticksuffix="%", gridcolor="#2d3250", color="white"),
        xaxis=dict(color="white"),
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="white"),
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def training_curve(history_resnet, history_vit, metric="val_acc"):
    epochs = list(range(1, len(history_resnet[metric]) + 1))
    labels = {"val_acc": "Validation Accuracy (%)", "val_loss": "Validation Loss"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history_resnet[metric],
        mode="lines+markers", name="ResNet-18",
        line=dict(color="#4a9eff", width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history_vit[metric],
        mode="lines+markers", name="ViT-B/16",
        line=dict(color="#ff6b6b", width=2), marker=dict(size=6),
    ))
    fig.update_layout(
        title=dict(text=labels.get(metric, metric), font=dict(size=14, color="white")),
        xaxis=dict(title="Epoch", color="white", gridcolor="#2d3250"),
        yaxis=dict(color="white", gridcolor="#2d3250"),
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e2130"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def per_class_f1_chart(resnet_f1s, vit_f1s):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ResNet-18", x=CLASSES, y=list(resnet_f1s.values()),
        marker_color="#4a9eff", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="ViT-B/16", x=CLASSES, y=list(vit_f1s.values()),
        marker_color="#ff6b6b", opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Per-Class F1-Score (%)", font=dict(size=14, color="white")),
        yaxis=dict(range=[0, 105], ticksuffix="%", color="white", gridcolor="#2d3250"),
        xaxis=dict(color="white"),
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e2130"),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 PoC Dashboard")
    st.caption("P9 — DataSpace Technical Test")
    st.divider()
    st.markdown("**Algorithm comparison**")
    st.markdown(
        '<span class="badge-baseline">Baseline</span> ResNet-18 (He et al., 2015)<br><br>'
        '<span class="badge-new">New</span> ViT-B/16 (Dosovitskiy et al., 2020)',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Dataset:** CIFAR-10")
    st.markdown("10 classes · 60,000 images")
    st.divider()
    st.markdown("**Reference**")
    st.markdown(
        "[arXiv:2010.11929](https://arxiv.org/abs/2010.11929) — "
        "*An Image is Worth 16x16 Words*"
    )
    st.markdown(
        "[arXiv:1512.03385](https://arxiv.org/abs/1512.03385) — "
        "*Deep Residual Learning for Image Recognition*"
    )
    st.divider()
    st.caption(f"Running on: `{DEVICE}`")

# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────

st.title("Vision Transformer vs. ResNet-18")
st.markdown(
    "**Proof of Concept** — Demonstrating that a Vision Transformer (ViT-B/16, 2020) "
    "outperforms a classical CNN baseline (ResNet-18, 2015) on image classification."
)

tab_demo, tab_results, tab_about = st.tabs(
    ["🖼️ Live Demo", "📊 Training Results", "📖 About"]
)

# ─────────────────────────────────────────────
# Tab 1: Live Demo
# ─────────────────────────────────────────────

with tab_demo:
    st.markdown("### Upload an image and compare both models in real-time")
    st.caption(
        "Both models are pretrained on ImageNet-1k. Upload any image to see "
        "how each model classifies it and inspect the ViT attention map."
    )

    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB")
        tensor = TRANSFORM(image)
        imagenet_labels = load_imagenet_labels()

        resnet = load_resnet()
        vit    = load_vit()

        resnet_probs, resnet_indices = predict(resnet, tensor)
        vit_probs,    vit_indices    = predict(vit,    tensor)

        resnet_top_labels = [imagenet_labels[i] for i in resnet_indices]
        vit_top_labels    = [imagenet_labels[i] for i in vit_indices]

        col_img, col_resnet, col_vit = st.columns([1, 1.5, 1.5])

        with col_img:
            st.markdown("**Uploaded Image**")
            st.image(image, width='stretch')
            st.caption(f"Size: {image.size[0]}×{image.size[1]} px")

        with col_resnet:
            st.plotly_chart(
                bar_chart(
                    resnet_top_labels[::-1],
                    resnet_probs[::-1],
                    "#4a9eff",
                    "ResNet-18 (Baseline)",
                ),
                width='stretch',
            )
            top_class = resnet_top_labels[0]
            top_conf  = resnet_probs[0] * 100
            st.markdown(
                f'<div class="model-card">'
                f'<span class="badge-baseline">BASELINE</span><br><br>'
                f'<b>Top prediction:</b> {top_class}<br>'
                f'<b>Confidence:</b> {top_conf:.1f}%<br>'
                f'<b>Architecture:</b> ResNet-18 · CNN · 11.7M params<br>'
                f'<b>Published:</b> 2015'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_vit:
            st.plotly_chart(
                bar_chart(
                    vit_top_labels[::-1],
                    vit_probs[::-1],
                    "#ff6b6b",
                    "ViT-B/16 (New Algorithm)",
                ),
                width='stretch',
            )
            top_class = vit_top_labels[0]
            top_conf  = vit_probs[0] * 100
            st.markdown(
                f'<div class="model-card">'
                f'<span class="badge-new">NEW ALGORITHM</span><br><br>'
                f'<b>Top prediction:</b> {top_class}<br>'
                f'<b>Confidence:</b> {top_conf:.1f}%<br>'
                f'<b>Architecture:</b> ViT-B/16 · Transformer · 86M params<br>'
                f'<b>Published:</b> 2020 · arXiv:2010.11929'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Attention map
        st.divider()
        st.markdown("### ViT Attention Map")
        st.caption(
            "Attention rollout (Abnar & Zuidema, 2020): [CLS] token attention "
            "propagated through all 12 transformer blocks. Brighter zones = "
            "regions the model focuses on for classification."
        )

        attn_map = get_vit_attention_map(vit, tensor)
        if attn_map is not None:
            import plotly.express as px

            img_resized = np.array(image.resize((224, 224)))
            attn_resized = np.array(
                Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                    (224, 224), Image.BILINEAR
                )
            ) / 255.0

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Original (224×224)**")
                st.image(img_resized, width='stretch')
            with col_b:
                st.markdown("**Attention Map**")
                fig = px.imshow(attn_resized, color_continuous_scale="viridis")
                fig.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="#0f1117",
                    paper_bgcolor="#0f1117",
                    height=220,
                )
                st.plotly_chart(fig, width='stretch')
            with col_c:
                st.markdown("**Overlay**")
                overlay = (0.5 * img_resized / 255.0 + 0.5 * plt_colorize(attn_resized))
                overlay = np.clip(overlay, 0, 1)
                st.image(overlay, width='stretch')
        else:
            st.warning(
                "Attention map could not be extracted from this model build. "
                "Predictions above are unaffected."
            )
    else:
        st.info(
            "Upload any image above to run inference with both models. "
            "You can use any photo — animals, vehicles, landscapes, etc."
        )


# ─────────────────────────────────────────────
# Tab 2: Training Results
# ─────────────────────────────────────────────

with tab_results:
    results = load_results()

    if results is None:
        st.warning(
            "No `results.json` found. Run the notebook first to generate training results, "
            "then place `results.json` in the `dashboard/` folder."
        )
        st.markdown(
            "**Expected file:** `dashboard/results.json` — generated automatically "
            "by the last cell of `notebook/poc_notebook.ipynb`."
        )
    else:
        rn = results["resnet18"]
        vt = results["vit_b16"]

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#4a9eff">{rn["test_accuracy"]:.2f}%</div>'
                f'<div class="metric-label">ResNet-18 Accuracy</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#ff6b6b">{vt["test_accuracy"]:.2f}%</div>'
                f'<div class="metric-label">ViT-B/16 Accuracy</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            delta = vt["test_accuracy"] - rn["test_accuracy"]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#2ecc71">+{delta:.2f}%</div>'
                f'<div class="metric-label">Accuracy Improvement</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col4:
            f1_delta = vt["f1_macro"] - rn["f1_macro"]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#2ecc71">+{f1_delta:.2f}%</div>'
                f'<div class="metric-label">F1-Macro Improvement</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Accuracy bar + training curves
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(accuracy_bar(rn["test_accuracy"], vt["test_accuracy"]),
                            width='stretch')
        with col_right:
            st.plotly_chart(training_curve(rn["history"], vt["history"], "val_acc"),
                            width='stretch')

        # Loss curve + per-class F1
        col_left2, col_right2 = st.columns(2)
        with col_left2:
            st.plotly_chart(training_curve(rn["history"], vt["history"], "val_loss"),
                            width='stretch')
        with col_right2:
            st.plotly_chart(per_class_f1_chart(rn["per_class_f1"], vt["per_class_f1"]),
                            width='stretch')

        # Summary table
        st.divider()
        st.markdown("### Model Comparison Summary")
        st.markdown(
            f"""
| Property | ResNet-18 (Baseline) | ViT-B/16 (New Algorithm) |
|---|---|---|
| Architecture | CNN (residual connections) | Transformer (self-attention) |
| Published | 2015 | 2020 (arXiv:2010.11929) |
| Parameters | 11.7M | 86.6M |
| Receptive field | Local | Global |
| Inductive bias | Strong (translation equiv.) | Weak |
| Test Accuracy (CIFAR-10) | {rn['test_accuracy']:.2f}% | **{vt['test_accuracy']:.2f}%** |
| Macro F1 | {rn['f1_macro']:.2f}% | **{vt['f1_macro']:.2f}%** |
"""
        )


# ─────────────────────────────────────────────
# Tab 3: About
# ─────────────────────────────────────────────

with tab_about:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## What is a Vision Transformer?")
        st.markdown(
            """
Dosovitskiy et al. (2020) adapted the Transformer architecture — originally designed
for NLP — to image classification. The core idea:

1. **Patch Embedding** — The image is split into fixed-size 16×16 patches.
   Each patch is linearly projected into a 768-dimensional embedding vector.

2. **[CLS] Token** — A learnable classification token is prepended to the sequence
   (identical to BERT in NLP).

3. **Position Embeddings** — Learnable 1D embeddings are added to preserve spatial order.

4. **Transformer Encoder** — 12 blocks of Multi-Head Self-Attention (MHSA) + MLP.
   Self-attention lets every patch attend to every other patch globally.

5. **Classification Head** — The [CLS] token output is passed through an MLP
   to produce class probabilities.

**Key advantage over CNNs**: CNNs have a limited local receptive field per layer.
ViT captures global context from the very first layer, which is beneficial when
classification depends on overall object structure rather than local textures.
"""
        )

    with col2:
        st.markdown("## Experimental Setup")
        st.markdown(
            """
**Dataset:** CIFAR-10 (Krizhevsky, 2009)
- 50,000 training images, 10,000 test images
- 10 balanced classes, 32×32 px (upscaled to 224×224 for both models)

**Baseline:** ResNet-18 pretrained on ImageNet → fine-tuned on CIFAR-10

**New algorithm:** ViT-B/16 pretrained on ImageNet-21k → fine-tuned on CIFAR-10

**Training details:**
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Loss: Cross-Entropy with label smoothing (0.1)
- Epochs: 10
- Batch size: 64
- Augmentation: random horizontal flip, random crop, color jitter

**Why fine-tuning from ImageNet pretrained weights?**
Training ViT from scratch on CIFAR-10 requires very large amounts of data.
Fine-tuning from pretrained weights is the standard practice in production
and aligns with how ViT is used in real-world applications.
"""
        )

    st.divider()
    st.markdown("## References")
    st.markdown(
        """
1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020).
   *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
   ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2015).
   *Deep Residual Learning for Image Recognition*.
   CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

3. Krizhevsky, A. (2009).
   *Learning Multiple Layers of Features from Tiny Images*.
   Technical Report. University of Toronto.

4. Vaswani, A., et al. (2017).
   *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

5. Wightman, R. (2019). *PyTorch Image Models (timm)*.
   [GitHub](https://github.com/huggingface/pytorch-image-models)
"""
    )
