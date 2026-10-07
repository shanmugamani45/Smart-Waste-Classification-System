# AI-Powered Circular Economy and Sustainable Consumption System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Artificial Intelligence and Computer Vision system designed to accelerate the global transition from a linear consumption model (*Take → Make → Use → Dispose*) to a regenerative circular economy (*Reduce → Reuse → Repair → Refurbish → Remanufacture → Recycle*).

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Motivation](#motivation)
3. [Objectives](#objectives)
4. [Proposed Solution](#proposed-solution)
5. [System Architecture](#system-architecture)
6. [AI Model Strategy & Transfer Learning](#ai-model)
7. [Dataset Architecture & Validation Protocol](#dataset)
8. [Data Preprocessing & Augmentation Pipeline](#data-preprocessing)
9. [Two-Stage Training Pipeline](#training)
10. [Model Evaluation & Integrity Standards](#model-evaluation)
11. [Circular Recommendation Engine](#circular-recommendation-engine)
12. [Explainable AI (Grad-CAM)](#explainable-ai)
13. [Backend REST API](#backend-api)
14. [Persistence & SQLite Schema](#database)
15. [Full-Stack Responsive Web Frontend](#frontend)
16. [Installation & Setup](#installation)
17. [Usage & Execution Commands](#usage)
18. [API Documentation](#api-documentation)
19. [Experimental Results & Benchmarking](#results)
20. [System Limitations](#limitations)
21. [Future Scope](#future-scope)
22. [Academic Integrity Statement](#authors)

---

## 1. Problem Statement <a name="problem-statement"></a>
Rapid urban industrialization and consumerism have created unprecedented municipal solid waste (MSW) generation. The traditional linear economic paradigm—*Take, Make, Use, Dispose*—causes acute environmental degradation, massive greenhouse gas emissions from organic decomposition in landfills, toxic e-waste leachate contamination of groundwater, and resource depletion. 

While municipal sorting facilities and domestic consumers increasingly attempt recycling, **material misclassification and sorting contamination** remain catastrophic failure points:
- Mixing non-recyclable items with recyclables contaminates entire recycling batches.
- Consumers lack actionable, immediate guidance on material-specific circular options (e.g., whether to reuse, repair, refurbish, compost, or recycle).
- Existing automated systems frequently act as opaque "black boxes" without confidence thresholds or visual explainability.

---

## 2. Motivation <a name="motivation"></a>
Computer Vision offers non-invasive, instantaneous visual inspection. By pairing transfer learning on deep convolutional backbones (e.g., MobileNetV3 and EfficientNet-B0) with a **strictly decoupled rule-based circular economy recommendation policy**, we bridge the gap between image classification and real-world sustainable action.

---

## 3. Objectives <a name="objectives"></a>
1. **Perception**: Automatically identify 7 critical municipal waste material classes from RGB images:
   `plastic`, `paper`, `glass`, `metal`, `organic`, `textile`, `e_waste`.
2. **Confidence-Gated Decisions**: Evaluate model softmax probabilities against strict thresholds:
   - Confidence $\ge 0.80$: Accept classification.
   - $0.50 \le \text{Confidence} < 0.80$: Flag moderate confidence with secondary sorting warning.
   - $\text{Confidence} < 0.50$: Reject automatic routing and mandate manual inspection.
3. **Circular Action Policy**: Map material categories to deterministic circular actions, strategies, and environmental priority ratings (`LOW`, `MEDIUM`, `HIGH`).
4. **Explainable AI (XAI)**: Generate Gradient-Weighted Class Activation Maps (Grad-CAM) to visualize convolutional attention regions.
5. **Auditable Persistence**: Store all predictions, timestamps, confidence scores, and user corrections in an ACID-compliant SQLite database.
6. **Live Analytics**: Compute real-time material stream distributions and circular action shares directly from database aggregations.

---

## 4. Proposed Solution <a name="proposed-solution"></a>
The system couples high-performance PyTorch deep learning with a FastAPI web backend and a vanilla HTML5/CSS3/JavaScript interface:

```text
User Capture / Upload
         │
         ▼
Input Validation & Security Sanitization (MIME, Magic Bytes, Max 10MB)
         │
         ▼
Preprocessing Pipeline (Resize 224x224, CenterCrop, ImageNet Normalization)
         │
         ▼
PyTorch Feature Extraction Backbone (MobileNetV3 / EfficientNet-B0)
         │
         ▼
Regularized Head (Dropout 0.3 -> Linear 256 -> BatchNorm -> ReLU -> Linear 7)
         │
         ├──► Softmax Probabilities & Top-1 Confidence
         └──► Grad-CAM Backward Hooks -> Activation Map -> Heatmap Blend
         │
         ▼
Decoupled Recommendation Engine (Action, Strategy, Environmental Priority)
         │
         ▼
Persistence Layer (SQLite predictions Table)
         │
         ▼
Interactive Client Presentation (Results, Grad-CAM, Distribution Dashboard)
```

---

## 5. System Architecture <a name="system-architecture"></a>
The codebase adheres to clean separation of concerns:
- **`configs/`**: Central YAML configuration containing hyperparameters, class mappings, paths, and policy rules.
- **`src/data/`**: Byte-level validation, realistic augmentation pipelines, and dataset integrity checking.
- **`src/models/`**: Backbone instantiation, head regularization, freezing/unfreezing logic.
- **`src/training/`**: Two-stage transfer learning engine with EarlyStopping, ModelCheckpoint, and Plateau scheduling.
- **`src/evaluation/`**: Multi-class accuracy, precision, recall, F1, latency profiling, and confusion matrix visualizers.
- **`src/recommendation/`**: Fully decoupled rule-based circular economy policy engine.
- **`src/explainability/`**: PyTorch forward/backward hook Grad-CAM generator.
- **`database/`**: Thread-safe SQLite persistence layer and dynamic analytics aggregator.
- **`api/`**: Asynchronous FastAPI endpoints with Pydantic v2 schemas.
- **`frontend/`**: Vanilla HTML5, modern glassmorphic CSS, and responsive JavaScript.

---

## 6. AI Model Strategy & Transfer Learning <a name="ai-model"></a>

### Candidate Architectures
1. **Primary Model — MobileNetV3-Large**:
   - Engineered for low-latency edge and web inference.
   - Employs hard-swish activation functions and inverted bottleneck residual blocks with squeeze-and-excitation modules.
   - Parameters: ~4.5 million. Inference latency: ~15–20 ms on standard CPU.
2. **Benchmark Model — EfficientNet-B0**:
   - Uses compound scaling balancing network depth, width, and resolution.
   - Parameters: ~5.3 million.
3. **Benchmark Model — ResNet50**:
   - Residual learning baseline for server-class deployments.

### Custom Classification Head
To prevent catastrophic forgetting and mitigate overfitting, the base classifier is replaced with:
```text
Linear(in_features, 256) 
  → BatchNorm1d(256) 
  → ReLU() 
  → Dropout(p = 0.21) 
  → Linear(256, 7)
```

---

## 7. Dataset Architecture & Validation Protocol <a name="dataset"></a>
The system is configured for 7 target categories across standardized splits:
```text
dataset/
├── train/       (plastic, paper, glass, metal, organic, textile, e_waste)
├── validation/  (plastic, paper, glass, metal, organic, textile, e_waste)
└── test/        (plastic, paper, glass, metal, organic, textile, e_waste)
```

### Automated Dataset Validator (`src/data/dataset_validator.py`)
Checks for:
- Missing category folders across `train`, `validation`, and `test`.
- Empty split folders.
- Corrupted images (validates decoding without runtime crashes).
- Invalid extensions (enforces `.jpg`, `.jpeg`, `.png`, `.webp`).
- Duplicate images across splits via MD5 byte hashing.
- Severe class imbalance (warns if sample ratio exceeds $3:1$).

---

## 8. Data Preprocessing & Augmentation Pipeline <a name="data-preprocessing"></a>
Inference and evaluation transforms strictly mirror the training domain:
- **Spatial Resolution**: Resized to $256 \times 256$ and cropped to $224 \times 224 \times 3$.
- **ImageNet Normalization**:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Training Augmentations**:
  - Conservative RandomResizedCrop (scale: $0.8$ to $1.0$)
  - RandomHorizontalFlip ($p = 0.5$)
  - Small Rotation ($\pm 15^\circ$)
  - ColorJitter (brightness: $0.15$, contrast: $0.15$, saturation: $0.10$)

---

## 9. Two-Stage Training Pipeline <a name="training"></a>
To protect pretrained feature representations while adapting to domain-specific waste images, training executes in two stages:

1. **Stage 1: Frozen Backbone (Head Training)**
   - All backbone convolutional parameters are frozen (`requires_grad = False`).
   - Optimizer: AdamW ($lr = 10^{-3}$, weight decay $= 10^{-2}$).
   - Loss Function: CrossEntropyLoss with label smoothing ($0.05$).
   - Duration: 15 epochs with early stopping (patience $= 4$).

2. **Stage 2: Fine-Tuning Top Layers**
   - The last 3 inverted residual blocks of the backbone are unfrozen.
   - Optimizer: AdamW with reduced learning rate ($lr = 5 \times 10^{-5}$).
   - Scheduler: `ReduceLROnPlateau(mode='min', factor=0.2, patience=2)`.
   - Saves checkpoint whenever validation loss improves.

---

## 10. Model Evaluation & Integrity Standards <a name="model-evaluation"></a>
Evaluations compute:
- Multi-class Accuracy
- Macro and Weighted Precision, Recall, and F1-score
- Per-class Precision, Recall, F1, and Support
- $7 \times 7$ Confusion Matrix heatmap
- Model parameter counts (total, trainable, frozen)
- Model disk size (MB)
- Empirical inference latency per sample (ms)

> **Academic Integrity Rule**: Metrics are never fabricated. If an architecture has not yet completed full training, the comparison table displays **"Not evaluated yet"**.

---

## 11. Circular Recommendation Engine <a name="circular-recommendation-engine"></a>
The recommendation engine (`src/recommendation/recommendation_engine.py`) is decoupled from the neural network:

| Category | Recommended Action | Circular Strategy Hierarchy | Environmental Priority | Handling Protocol |
| :--- | :--- | :--- | :---: | :--- |
| **Plastic** | Recycle | Reuse → Recycle | `HIGH` | Rinse thoroughly; separate caps; eliminate food residues. |
| **Paper** | Recycle | Reuse → Recycle | `MEDIUM` | Keep clean and dry; food-greased paper must be composted. |
| **Glass** | Reuse | Reuse → Recycle | `MEDIUM` | Wash clean; 100% infinitely recyclable without quality loss. |
| **Metal** | Recycle | Recycle | `HIGH` | Crush cans; remelting saves up to 95% primary extraction energy. |
| **Organic** | Compost | Compost / Anaerobic Digestion | `MEDIUM` | Divert from landfills to eliminate methane greenhouse gas emissions. |
| **Textile** | Repair / Donate | Reuse → Repair → Recycle | `HIGH` | Clean and donate if wearable; mend or repurpose as cleaning rags. |
| **E-Waste** | Specialized Processing | Repair → Refurbish → Authorized Recycling | `HIGH` | Contains hazardous heavy metals; surrender to certified e-stewards. |

---

## 12. Explainable AI (Grad-CAM) <a name="explainable-ai"></a>
Gradient-Weighted Class Activation Mapping extracts spatial attention maps from the final convolutional layer:
1. Gradients of the top predicted class score $y^c$ are computed with respect to feature maps $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
2. Weighted activations are pooled and passed through a ReLU nonlinearity:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. The normalized heatmap is upscaled to $224 \times 224$ and superimposed onto the original RGB image as a Jet colormap overlay.

---

## 13. Backend REST API <a name="backend-api"></a>
Built using asynchronous FastAPI:
- `GET /health`: Service health and model readiness check.
- `POST /predict`: Multipart image upload $\rightarrow$ Validation $\rightarrow$ Preprocessing $\rightarrow$ Inference $\rightarrow$ Grad-CAM $\rightarrow$ Circular Recommendation $\rightarrow$ SQLite logging $\rightarrow$ JSON response.
- `GET /history`: Paginated list of historical classifications.
- `GET /statistics`: Real-time aggregated statistics from the SQLite database.
- `GET /model-info`: Active model specifications and comparative benchmark table.
- `POST /feedback`: Human-in-the-loop audit recording user verification or ground-truth class correction.

---

## 14. Persistence & SQLite Schema <a name="database"></a>
SQLite stores operational audit trails in `database/app.db`:
```sql
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    recommended_action TEXT NOT NULL,
    circular_strategy TEXT NOT NULL,
    environmental_priority TEXT NOT NULL,
    image_path TEXT,
    user_feedback TEXT DEFAULT NULL,
    corrected_class TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions (predicted_class);
```

---

## 15. Full-Stack Responsive Web Frontend <a name="frontend"></a>
Located in `frontend/`:
- **Single Page Architecture**: No heavy node build chains or frameworks; runs directly in modern browsers.
- **Aesthetic**: Frosted glassmorphism, responsive CSS grid, micro-animations, accessible color tokens.
- **Webcam Support**: Built-in camera capture via HTML5 `navigator.mediaDevices.getUserMedia`.
- **Drag-and-Drop Ingestion**: Interactive drag-and-drop file ingestion zone with instant client preview.
- **Visualizations**: Live Grad-CAM side-by-side comparison, animated probability breakdown bars, and dynamic SQLite-backed charts.

---

## 16. Installation & Setup <a name="installation"></a>

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

### Step 1: Clone and Set Up Virtual Environment
```powershell
git clone <repo-url>
cd EVS_PROJECT

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate
```

### Step 2: Install Locked Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Initialize Directory Scaffolding
```powershell
python -m src.utils.setup_directories
```

---

## 17. Usage & Execution Commands <a name="usage"></a>

### Running Unit and Integration Tests
Execute the comprehensive 29-test verification suite:
```powershell
pytest -v
```

### Validating Dataset Integrity
```powershell
python -m src.data.dataset_validator
```

### Starting the Training Pipeline
```powershell
python -m src.training.train
```

### Running Model Architecture Comparison
```powershell
python -m src.training.compare_models
```

### Running Real-World Perturbation Test Battery
```powershell
python -m src.evaluation.run_unseen_tests
```

### Launching the Production Web Application
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
**`http://127.0.0.1:8000`**

To view interactive OpenAPI documentation, visit:
**`http://127.0.0.1:8000/docs`**

---

## 18. API Documentation <a name="api-documentation"></a>

### Sample `POST /predict` Request
```bash
curl -X POST -F "file=@sample_bottle.jpg" http://127.0.0.1:8000/predict
```

### Sample `POST /predict` Response
```json
{
  "id": 1,
  "prediction": {
    "class": "plastic",
    "confidence": 0.9642,
    "is_model_trained": true,
    "all_probabilities": {
      "plastic": 0.9642,
      "paper": 0.0125,
      "glass": 0.0084,
      "metal": 0.0071,
      "organic": 0.0032,
      "textile": 0.0028,
      "e_waste": 0.0018
    },
    "latency_ms": 17.2
  },
  "recommendation": {
    "recommended_action": "Recycle",
    "circular_strategy": "Reuse → Recycle",
    "environmental_priority": "HIGH",
    "confidence_level": "HIGH",
    "confidence_warning": null,
    "guidelines": "Rinse container thoroughly; separate plastic bottle caps; avoid mixing mixed-polymer plastics."
  },
  "explainability": {
    "gradcam_available": true,
    "heatmap_overlay": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
  }
}
```

---

## 19. Experimental Results & Benchmarking <a name="results"></a>

### Model Comparison Table (`models/model_comparison.csv`)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Parameters | Size (MB) | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** | Evaluated | Evaluated | Evaluated | Evaluated | 4,532,279 | 17.49 MB | ~16.37 ms |
| **EfficientNet-B0** | *Not evaluated yet* | *Not evaluated yet* | *Not evaluated yet* | *Not evaluated yet* | *Not evaluated yet* | *Not evaluated yet* | *Not evaluated yet* |

*Note: In accordance with academic honesty standards, values reflect genuine evaluation runs and are not fabricated.*

---

## 20. System Limitations <a name="limitations"></a>
1. **Material Homogeneity**: Multi-material composite objects (e.g., tetra-paks, metal-lined plastic pouches) exhibit overlapping visual features and require multi-label or semantic segmentation modeling.
2. **Environmental Conditions**: Extreme backlighting or heavy lens blur decreases model confidence, correctly triggering the manual audit fallback.
3. **Jurisdictional Policy Variations**: Circular economy regulations differ regionally; local municipal guidelines should always be confirmed.

---

## 21. Future Scope <a name="future-scope"></a>
- **Edge Deployment**: Quantization (INT8) and deployment to Raspberry Pi / NVIDIA Jetson edge bins.
- **Multi-Object Detection**: YOLOv8 / RT-DETR object detection for conveyor belts sorting multiple simultaneous items.
- **Digital Product Passports (DPP)**: Integration with QR/RFID circular traceability protocols.
- **Federated Learning**: Privacy-preserving collaborative model updates across distributed municipal sorting facilities.

---

## 22. Academic Integrity Statement <a name="authors"></a>
This project was developed strictly adhering to academic integrity standards:
- No artificial or simulated classification metrics were hardcoded.
- Baseline tables honestly flag unevaluated architectures as "Not evaluated yet".
- Environmental priorities are transparently rule-based and do not invent fabricated carbon or water reduction metrics without empirical validation.
