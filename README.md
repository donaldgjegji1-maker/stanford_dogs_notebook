# Stanford Dogs Breed Classifier
### Fine-Grained Visual Classification via Transfer Learning (EfficientNetB3 + Keras)

---

## Overview

This project trains a Convolutional Neural Network to classify **120 dog breeds** from the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) using transfer learning on top of **EfficientNetB3** pre-trained on ImageNet. The final model achieves **≥ 87% validation accuracy**.

The full pipeline — data loading, preprocessing, training, evaluation, and reporting — is contained in a single self-documented Jupyter/Colab notebook structured as a scientific research report.

---

## Project Structure

```
.
├── stanford_dogs_classification.ipynb   # Main notebook (training + report)
├── stanford_dogs.h5                     # Saved trained model (compiled)
├── model_summary.txt                    # Full layer-by-layer model summary
├── training_curves.png                  # Accuracy & loss plots (both phases)
├── head_architecture.png                # Classification head diagram
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Stanford Dogs / ImageNet |
| Classes | 120 dog breeds |
| Train images | 12,000 |
| Test images | 8,580 |
| Total | 20,580 |
| Input resolution | 300 × 300 × 3 |

Loaded automatically via `tensorflow_datasets`:
```python
import tensorflow_datasets as tfds
ds, info = tfds.load("stanford_dogs", split=["train", "test"], with_info=True)
```

---

## Model Architecture

```
Input (300, 300, 3)
    │
    ▼
Rescaling (scale=1.0)          ← Hint 2: first layer scales input
    │
    ▼
EfficientNetB3 (ImageNet)      ← frozen in Phase 1 / top 30% unfrozen in Phase 2
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
BatchNormalization
    │
    ▼
Dense(512, relu) + L2(1e-4)
    │
    ▼
Dropout(0.4)
    │
    ▼
Dense(120, softmax)
```

---

## Training Strategy

Training is split into two phases to maximize speed and accuracy:

### Phase 1 — Feature Extraction (frozen base)
All EfficientNetB3 layers are frozen. Bottleneck features are **computed once** and cached as NumPy arrays. Only the classification head is trained on the cached features — reducing epoch time from ~15 min to ~30 seconds.

### Phase 2 — Fine-Tuning (partial unfreeze)
The top ~30% of the base model's layers are unfrozen and trained end-to-end with data augmentation at a very low learning rate (`1e-5`) to prevent catastrophic forgetting.

| Setting | Phase 1 | Phase 2 |
|---------|---------|---------|
| Base frozen | Fully | Top 30% unfrozen |
| Learning rate | 1e-3 | 1e-5 |
| Augmentation | No | Yes |
| Max epochs | 20 | 30 |
| Early stopping patience | 5 | 7 |

---

## Data Augmentation (Phase 2)

- Random horizontal flip  
- Random rotation ± 10°  
- Random zoom ± 10%  
- Random contrast adjustment  

---

## `preprocess_data` Function

```python
def preprocess_data(X, Y):
    """
    Args:
        X: numpy.ndarray of images, shape (m, H, W, 3), any resolution
        Y: numpy.ndarray of integer labels, shape (m,), values in [0, 119]

    Returns:
        X_p: numpy.ndarray, shape (m, 300, 300, 3), dtype float32
        Y_p: numpy.ndarray, shape (m, 120), one-hot encoded, dtype float32
    """
```

Resizes all images to 300×300 and one-hot encodes labels into 120 classes.

---

## Results

| Metric | Value |
|--------|-------|
| Final Validation Accuracy | ≥ 87% |
| Base Architecture | EfficientNetB3 |
| Parameters (total) | ~12M |
| Parameters (trained, Phase 1) | ~400K (head only) |
| Parameters (trained, Phase 2) | ~3M (top layers + head) |

---

## Requirements

```
tensorflow >= 2.12
tensorflow-datasets
numpy
matplotlib
seaborn
scikit-learn
```

Install in Colab:
```bash
pip install tensorflow tensorflow-datasets
```

---

## Usage

### Run in Google Colab (recommended)
1. Open `stanford_dogs_classification.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run all cells: `Runtime → Run all`
4. The trained model is saved to `/content/stanford_dogs.h5`

### Save model
```python
from google.colab import drive
drive.mount("/content/drive")

import shutil
shutil.copy("stanford_dogs.h5", "/content/drive/MyDrive/stanford_dogs.h5")
```

### Load the saved model
```python
import keras
model = keras.models.load_model("stanford_dogs.h5", safe_mode=False)
```

### Use `preprocess_data` externally
```python
import numpy as np
# X: raw images as uint8 numpy array, Y: integer labels
X_p, Y_p = preprocess_data(X, Y)
predictions = model.predict(X_p)
```

---

## Notebook Report Structure

The notebook doubles as a full research report, organized as follows:

| Section | Content |
|---------|---------|
| Abstract | Summary of approach and results |
| Introduction | Problem definition and motivation |
| Materials & Methods | Architecture, training strategy, augmentation, regularization |
| Results | Training curves, final metrics, confusion matrix, sample predictions |
| Discussion | Experimental log, key insights, limitations, future work |
| Acknowledgments | Tools and platforms used |
| Literature Cited | Referenced papers |
| Appendices | Model summary, hyperparameter table, `preprocess_data` unit test |

---

## References

1. Khosla et al. (2011). *Novel dataset for Fine-Grained Image Categorization.* CVPR Workshop.
2. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML. arXiv:1905.11946.
3. Deng et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR.
4. Tan et al. (2018). *A Survey on Deep Transfer Learning.* LNCS 11141.

---

## Notes

- The `.h5` format is required by the project specification. Load it with `safe_mode=False` due to Keras's default Lambda layer deserialization restriction.
- `keras.utils.plot_model` will crash on the full model (Graphviz overflow on 500+ nodes). The notebook plots only the classification head and saves the full summary to `model_summary.txt`.
- Do not run this notebook as an imported module — the training script is guarded and only executes in `__main__` context.
