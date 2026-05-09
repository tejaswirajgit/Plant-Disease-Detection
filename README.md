# Plant Disease Detection

Multi-class plant leaf disease classifier — **38 classes** across 14 plant species — built on transfer-learned **EfficientNetB0** with TensorFlow / Keras 2.12. Reference run reached ~99.84% validation accuracy.

## Dataset

[New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle (~88k labeled images, train / valid / test splits).

The dataset is **not** committed to this repo (see `.gitignore`). Download it via Kaggle:

```bash
pip install kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset --unzip -p .
```

Or via `opendatasets` from inside the notebook (requires the same credentials).

## Setup

```bash
# 1. Clone
git clone https://github.com/tejaswirajgit/Plant-Disease-Detection.git
cd Plant-Disease-Detection

# 2. Create a virtual env (Python 3.10+ recommended; TF 2.12 supports up to 3.11)
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Provide Kaggle credentials for dataset download
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
# then fill in KAGGLE_USERNAME and KAGGLE_KEY
```

## Run

Open `Plant Leaf Disease Detection.ipynb` in Jupyter or VS Code and execute cells top-to-bottom.

> ⚠ **Patch the dataset paths first.** The notebook hardcodes the original author's `E:/MINICONDA_FILES/...` paths. Replace them with the in-repo layout — see [CLAUDE.md](CLAUDE.md#important-hardcoded-dataset-paths) for the exact rewrite.

## Architecture

EfficientNetB0 (ImageNet weights, `include_top=False`) → `GlobalAveragePooling2D` → `Dense(38, softmax)`.

- Input: 224×224×3, batch size 32, `label_mode='categorical'`
- Fine-tuning: last 20 backbone layers unfrozen
- Loss: `categorical_crossentropy`, optimizer: Adam
- Callbacks: `EarlyStopping(patience=3)`, `ReduceLROnPlateau(factor=0.2, patience=2)`, `ModelCheckpoint(save_best_only=True, save_weights_only=True)`, `TensorBoard`
- Reference val accuracy: **~99.84%**

Full pipeline notes in [CLAUDE.md](CLAUDE.md).

## Project layout

```
.
├── CLAUDE.md                            # context for Claude Code instances
├── LICENSE                              # MIT
├── README.md                            # this file
├── requirements.txt                     # pinned deps (TF 2.12)
├── .env.example                         # template for Kaggle creds (copy to .env)
├── .gitignore                           # excludes dataset, weights, secrets
├── .claude/
│   └── agents/
│       └── ml-engineering.md            # ML engineer subagent definition
├── Plant Leaf Disease Detection.ipynb   # main pipeline notebook
└── New Plant Diseases Dataset/          # gitignored — Kaggle data
```

## License

[MIT License](LICENSE) © 2026 Tejaswi Raj
