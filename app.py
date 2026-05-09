"""Gradio entry point for the plant disease classifier.

Local:        python app.py
HF Space:     uploaded as the Space's app_file (see deploy_to_hf.py).

The model weights live in a GitHub Release, not in this Space's image.
``ensure_model_present`` downloads them on cold start.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from app.model_utils import ensure_model_present
from app.predict import MODEL_PATH, predict

# Trigger the weights download (or no-op if already on disk) before the UI
# accepts traffic. On HF Spaces this happens once per cold boot.
ensure_model_present(MODEL_PATH)

REPO_URL = "https://github.com/tejaswirajgit/Plant-Disease-Detection"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def _list_examples() -> list[list[str]] | None:
    if not EXAMPLES_DIR.is_dir():
        return None
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(EXAMPLES_DIR.glob(ext)))
    return [[str(f)] for f in files] if files else None


def classify(pil_image):
    if pil_image is None:
        return {}, "_Upload a leaf photo to get a prediction._"
    result = predict(pil_image)
    label_dict = {
        f"{p['crop']} - {p['condition']}": p["prob"] for p in result["top_3"]
    }
    header = (
        result["condition"]
        if result["is_healthy"]
        else f"{result['crop']} - {result['condition']}"
    )
    md = (
        f"### {header}\n\n"
        f"- **Crop:** {result['crop']}\n"
        f"- **Condition:** {result['condition']}\n"
        f"- **Healthy:** {'yes' if result['is_healthy'] else 'no'}\n"
        f"- **Confidence:** {result['confidence'] * 100:.2f}%\n\n"
        f"<sub>Raw label: <code>{result['raw_label']}</code></sub>"
    )
    return label_dict, md


with gr.Blocks(title="Plant Disease Detection") as demo:
    gr.Markdown(
        "# Plant Disease Detection\n\n"
        "Upload a leaf photo. The model identifies the crop and classifies it "
        "across 38 disease classes (including healthy) for 14 plant species. "
        "Built on EfficientNetB0 fine-tuned with TensorFlow / Keras."
    )

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Leaf photo")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Top 3 predictions")
            out_md = gr.Markdown(value="_Upload a leaf photo to get a prediction._")

    btn.click(fn=classify, inputs=inp, outputs=[out_label, out_md])
    inp.change(fn=classify, inputs=inp, outputs=[out_label, out_md])

    _examples = _list_examples()
    if _examples:
        gr.Examples(
            examples=_examples,
            inputs=inp,
            outputs=[out_label, out_md],
            fn=classify,
            cache_examples=False,
            label="Examples",
        )

    gr.Markdown(
        f"---\n\nSource: [{REPO_URL}]({REPO_URL})  •  "
        "Model: EfficientNetB0 transfer-learned on the New Plant Diseases Dataset (38 classes)."
    )


if __name__ == "__main__":
    demo.launch()
