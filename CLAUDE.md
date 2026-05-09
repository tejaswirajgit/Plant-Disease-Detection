# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit & contributor rules

This is a personal repo and the user (`tejaswirajgit`) wants to be the **sole contributor** on GitHub's Contributors graph.

- **Never append a `Co-Authored-By:` trailer** to any commit message in this repo, even though Claude Code's default commit instructions normally add `Co-Authored-By: Claude ...`. The commit body ends with the change description.
- Author identity = the user's git config only (`tejaswirajgit <tejaswirajmgr@gmail.com>` — verified on GitHub as Primary).
- Don't change `git config user.email` without asking — it already points at a verified GitHub email.
- Approval-gated commits: never run `git commit`, `git push`, `git tag`, `gh pr create`, or `gh release create` without explicit user approval. Stage and prepare the message, then stop and ask.

## Repository contents

This repo is a single Jupyter notebook ML project — there is no source package, build system, or test suite.

- [Plant Leaf Disease Detection.ipynb](Plant%20Leaf%20Disease%20Detection.ipynb) — the entire pipeline (data loading, model definition, training, evaluation, prediction helpers).
- [New Plant Diseases Dataset/](New%20Plant%20Diseases%20Dataset/) — the Kaggle "New Plant Diseases Dataset (Augmented)" image data.
  - `New Plant Diseases Dataset(Augmented)/train/` — 38 class folders, ~70k training images
  - `New Plant Diseases Dataset(Augmented)/valid/` — 38 class folders, ~17.5k validation images
  - `test/test/` — flat folder of unlabeled JPEGs (filename encodes the true class, e.g. `AppleCedarRust1.JPG`)

Class folder naming uses the `<Plant>___<Condition>` convention (three underscores), e.g. `Tomato___Early_blight`, `Apple___healthy`.

## Running the notebook

The notebook expects TensorFlow 2.12.0 and uses Keras' `image_dataset_from_directory`. Open it in Jupyter / VS Code and run cells top-to-bottom. There is no `requirements.txt` — install on demand: `tensorflow==2.12`, `opendatasets`, `matplotlib`.

There is no `pytest` / `npm test` / build step — validation is via `feature_model.evaluate(test_data)` inside the notebook.

## Important: hardcoded dataset paths

The notebook's paths point at the original author's machine, **not** at the data checked into this repo:

```python
train_dir = 'E:/MINICONDA_FILES/PROJECT3/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
```

When running here, rewrite these to the in-repo locations before executing the data-loading cells:

```python
train_dir = 'New Plant Diseases Dataset/New Plant Diseases Dataset(Augmented)/train'
test_dir  = 'New Plant Diseases Dataset/New Plant Diseases Dataset(Augmented)/valid'
# unlabeled holdout images:
data_dir  = 'New Plant Diseases Dataset/test/test'
```

Note the doubled `New Plant Diseases Dataset(Augmented)/` segment in the original is an artifact of how Kaggle's `opendatasets` extracts the archive — the in-repo layout has it only once.

## Model architecture (high level)

Transfer learning on **EfficientNetB0** (ImageNet weights, `include_top=False`):

1. Input `(224, 224, 3)` → frozen `EfficientNetB0` backbone
2. `GlobalAveragePooling2D` → `Dense(38, softmax)` head
3. Fine-tuning unfreezes the **last 20 layers** of the backbone (`base_model.layers[:-20]` stay frozen)
4. Loss: `categorical_crossentropy`, optimizer: `Adam`, `image_size=(224,224)`, `batch_size=32`, `label_mode='categorical'`

Training callbacks: `EarlyStopping(patience=3, monitor=val_loss)`, `ReduceLROnPlateau(factor=0.2, patience=2)`, `ModelCheckpoint(save_weights_only=True, save_best_only=True)` writing to `fine_tune_checkpoints/`, plus a TensorBoard callback that logs to `plant_disease_model/<experiment>/<timestamp>/`. The reference run reached ~99.84% validation accuracy.

The trained checkpoint (`fine_tune_checkpoints/`) is **not** in the repo — re-running the fit cell is required to reproduce, and a single epoch takes ~30 minutes on CPU per the original output. To reuse a prior run, drop the checkpoint files in place and skip straight to `feature_model.load_weights(checkpoint_path)`.

## Inference helpers

Two helpers near the bottom of the notebook:

- `load_prep(img_path)` — reads + resizes to 224×224 (no normalization; EfficientNet handles scaling internally).
- `random_image_predict(model, test_dir, ...)` — picks a random class folder under the labeled `valid/` tree and plots predicted vs. actual.

The final cell predicts against `test/test/` (the unlabeled holdout) — there `class_names` indexing relies on `pred.argmax()` over the same 38-class softmax used in training, so the train/valid `class_names` order must be preserved.
