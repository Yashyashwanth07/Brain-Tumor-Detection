import json
import os
import copy
from pathlib import Path

# Load original notebook
with open('swin-tiny-original.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

orig_cells = nb['cells']

def make_md_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.strip().split("\n")]
    }

def make_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

# Build the refined list of cells
refined_cells = []

# --- 1. TITLE & CLINICAL MOTIVATION ---
refined_cells.append(make_md_cell("""# 🧠 Brain Tumor MRI Classification — Swin-Tiny Pipeline
### Deep Learning via Hierarchical Vision Transformer with Shifted Windows (Swin-Tiny)

---

## Executive Summary & Clinical Context
Brain tumor diagnosis from Magnetic Resonance Imaging (MRI) requires distinguishing between three primary intracranial neoplasms:
1. **Glioma** (typically invasive, intra-axial glial tissue neoplasms)
2. **Meningioma** (extra-axial tumors arising from meningeal coverings)
3. **Pituitary Tumor** (sellar/suprasellar adenomas affecting the endocrine system)
4. **No Tumor** (healthy brain control scans)

Traditional Convolutional Neural Networks (CNNs) rely on local receptive fields, which can struggle to capture long-range contextual anatomical relationships across multi-planar MRI slices. Standard Vision Transformers (ViTs), while possessing global attention, suffer from quadratic computational complexity $\mathcal{O}(N^2)$ with respect to image resolution and lack multi-scale feature hierarchies.

**The Swin Transformer (Shifted Window Transformer)** resolves this by computing self-attention within local non-overlapping windows and allowing cross-window connections via shifted window partitioning. This provides:
- **Linear computational complexity $\mathcal{O}(M \times N)$** to image size
- **Hierarchical multi-scale representation** akin to feature pyramids in CNNs
- **Superior spatial sensitivity** critical for identifying tumor boundaries and subtle mass effects

---

### Pipeline Architecture & Best Practices
This end-to-end pipeline incorporates rigorous medical imaging and ML engineering best practices:
1. **Dataset Ingestion & Quality Audit**: Automatic class discovery, corrupted image detection, and dimension profiling.
2. **Leakage-Safe Perceptual Hashing (pHash)**: Duplicate/near-duplicate detection ensuring identical or near-identical slices remain strictly within a single partition.
3. **Stratified Partitioning**: Group-aware 80/10/10 split preserving exact class ratios without data snooping.
4. **Medically Safe Data Augmentation**: Controlled affine transforms, subtle rotations, and ImageNet normalization.
5. **Multi-GPU Distributed Acceleration**: `torch.nn.DataParallel` across 2× Tesla T4 GPUs with Automatic Mixed Precision (AMP).
6. **Cosine Annealing with Linear Warmup**: Smooth gradient optimization preventing early representation collapse.
7. **Comprehensive Diagnostics**: Multi-class ROC-AUC (One-vs-Rest), Precision-Recall curves, Calibration Reliability Diagram (Expected Calibration Error - ECE), and Confidence Error Grids.
8. **Explainable AI (XAI)**: Grad-CAM adapted for Swin Transformer spatial feature maps to visualize attentional saliency.
9. **Architectural Benchmarking & Ablation**: Comparison against ResNet-50 and ConvNeXt-Tiny baselines."""))

# --- CELL 1: INSTALL / IMPORTS ---
refined_cells.append(make_md_cell("""## 1. Environment Setup & Dependency Management
We inspect and install only necessary external packages (`timm`, `captum`) dynamically while maintaining core PyTorch and CUDA runtime integrity."""))

# Keep original Cell 1
refined_cells.append(orig_cells[1])

# --- CELL 2: GPU / ACCELERATOR CHECK ---
refined_cells.append(make_md_cell("""## 2. Hardware Acceleration & Multi-GPU Diagnostics
Verifying visible CUDA hardware. The pipeline dynamically detects and leverages single or dual Tesla T4 GPUs via `torch.nn.DataParallel` with graceful CPU fallback."""))

# Keep original Cell 2
refined_cells.append(orig_cells[2])

# --- CELL 3: GLOBAL CONFIGURATION ---
refined_cells.append(make_md_cell("""## 3. Global Pipeline Configuration
Centralized hyperparameter control using Python `@dataclass`. Includes dynamic dataset path resolution for seamless execution across Kaggle environments and local development environments."""))

# Update Cell 3 with dynamic path resolution
c3_src = """# ============================================================
# CELL 3 — GLOBAL CONFIGURATION
# ============================================================
def resolve_dataset_path() -> str:
    \"\"\"Automatically locate the dataset directory across Kaggle or local environments.\"\"\"
    candidates = [
        "/kaggle/input/datasets/yashwanthboggavarapu/braintumor",
        "/kaggle/input/braintumor",
        "/kaggle/input/brain-tumor-mri-dataset",
        "./Dataset_MRI_Brain_Tumor",
        "../Dataset_MRI_Brain_Tumor",
        "./braintumor"
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return "/kaggle/input/datasets/yashwanthboggavarapu/braintumor"

@dataclass
class Config:
    # paths
    DATASET_PATH: str = field(default_factory=resolve_dataset_path)
    WORK_DIR: str = "/kaggle/working" if os.path.exists("/kaggle") else "./working"
    RESULTS_DIR: str = "/kaggle/working/results" if os.path.exists("/kaggle") else "./results"
    XAI_DIR: str = "/kaggle/working/results/xai" if os.path.exists("/kaggle") else "./results/xai"

    # reproducibility
    SEED: int = 42

    # data
    IMAGE_SIZE: int = 224
    NUM_CLASSES: int = 4          # updated automatically from discovered folders
    VALID_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    TRAIN_FRAC: float = 0.80
    VAL_FRAC: float = 0.10
    TEST_FRAC: float = 0.10

    # dataloader
    BATCH_SIZE: int = 32          # batch size per step
    NUM_WORKERS: int = 4 if platform.system() != "Windows" else 0

    # training
    EPOCHS: int = 80              # early stopping halts if validation macro-F1 plateaus
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.05
    WARMUP_EPOCHS: int = 3
    GRAD_CLIP_NORM: float = 1.0
    EARLY_STOP_PATIENCE: int = 12  # tolerance before triggering early stopping
    USE_AMP: bool = True
    USE_CLASS_WEIGHTS: Optional[bool] = None   # automatic calculation if imbalance detected

    # model
    MODEL_NAME: str = "swin_tiny_patch4_window7_224"

    # optional comparison & ablation
    RUN_BASELINES: bool = True
    BASELINE_EPOCHS: int = 5      # optimized for swift transfer-learning comparison
    BASELINE_MODELS: Tuple[str, ...] = ("resnet50", "convnext_tiny")
    RUN_ENSEMBLE: bool = False
    RUN_ABLATION: bool = False

    # duplicate detection via perceptual hashing
    PHASH_SIZE: int = 8           # 64-bit perceptual hash
    PHASH_HAMMING_THRESHOLD: int = 4  # Hamming distance <= 4 indicates duplicate slice


CFG = Config()
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
os.makedirs(CFG.XAI_DIR, exist_ok=True)

print("Pipeline Configuration Initialized:")
for k, v in asdict(CFG).items():
    print(f"  {k:<28}: {v}")

with open(os.path.join(CFG.RESULTS_DIR, "config.json"), "w") as f:
    json.dump(asdict(CFG), f, indent=2)"""

refined_cells.append(make_code_cell(c3_src))

# --- CELL 4: REPRODUCIBILITY ---
refined_cells.append(make_md_cell("""## 4. Reproducibility & cuDNN Benchmarking
Fixing deterministic seeds across Python, NumPy, and PyTorch while enabling `cudnn.benchmark` for optimized convolution and attention kernel execution on NVIDIA T4 Tensor Cores."""))

refined_cells.append(orig_cells[4])

# --- CELL 5: DATASET DISCOVERY ---
refined_cells.append(make_md_cell("""## 5. Dataset Discovery & Structure Scanning
Recursive discovery of class directories to dynamically register label mappings without hardcoding."""))

refined_cells.append(orig_cells[5])

# --- CELL 6: DATASET INSPECTION & INTEGRITY AUDIT ---
refined_cells.append(make_md_cell("""## 6. Dataset Audit: Integrity, Corrupt Image Detection & Dimensions
A single unified pass over the dataset checking file headers, image readability (with truncated image handling), resolution statistics, and class distribution."""))

refined_cells.append(orig_cells[6])

# --- CELL 7: DUPLICATE / NEAR-DUPLICATE DETECTION (PHASH) ---
refined_cells.append(make_md_cell("""## 7. Data Leakage Prevention: Perceptual Hashing (pHash)
In medical imaging datasets, identical or nearly-identical adjacent MRI slices frequently appear across subfolders. If one slice enters the training set and its near-twin enters the test set, metrics become artificially inflated. 

We compute 64-bit DCT perceptual hashes and group identical/near-identical images into connected components, ensuring every group is assigned atomically to a single split."""))

refined_cells.append(orig_cells[7])

# --- CELL 8: LEAKAGE-SAFE TRAIN/VAL/TEST SPLIT ---
refined_cells.append(make_md_cell("""## 8. Stratified Group-Aware Train / Val / Test Split
Partitioning data into **80% Train**, **10% Validation**, and **10% Test** sets while respecting duplicate group integrity and maintaining stratification."""))

refined_cells.append(orig_cells[8])

# --- CELL 9: DATASET VISUALIZATION ---
refined_cells.append(make_md_cell("""## 9. Exploratory Data Visualization
Inspecting representative MRI scans across all four diagnostic categories (Glioma, Meningioma, Pituitary, No Tumor)."""))

refined_cells.append(orig_cells[9])

# --- CELL 10: TRANSFORMS & AUGMENTATION ---
refined_cells.append(make_md_cell("""## 10. Medically Safe Data Augmentation
MRI scans require realistic geometrical invariances without creating non-physiological artifacts. We apply:
- Random horizontal flip (bilateral anatomical symmetry)
- Subtle affine rotations ($\pm 10^\circ$) and scaling ($0.95 - 1.05$)
- Slight color jitter (contrast/brightness variations across scanner calibrations)
- ImageNet standardization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$)"""))

refined_cells.append(orig_cells[10])

# --- CELL 11: DATALOADERS ---
refined_cells.append(make_md_cell("""## 11. High-Throughput PyTorch DataLoaders
Wrapping training, validation, and testing partitions with pinned memory and multi-worker prefetching."""))

refined_cells.append(orig_cells[11])

# --- CELL 12: CLASS IMBALANCE HANDLING ---
refined_cells.append(make_md_cell("""## 12. Class Imbalance Assessment
Inspecting effective number of samples and automatically computing inverse-frequency loss weights if needed."""))

refined_cells.append(orig_cells[12])

# --- CELL 13: SWIN-TINY ARCHITECTURE ---
refined_cells.append(make_md_cell("""## 13. Swin Transformer Model Instantiation
Loading `swin_tiny_patch4_window7_224` pretrained on ImageNet-1k, replacing the classification head with a 4-class projection linear layer."""))

refined_cells.append(orig_cells[13])

# --- CELL 14: MULTI-GPU WRAPPING ---
refined_cells.append(make_md_cell("""## 14. Multi-GPU DataParallel Setup
Deploying model across visible GPUs with automatic batch scatter and gather."""))

refined_cells.append(orig_cells[14])

# --- CELL 15: AMP MIXED PRECISION ---
refined_cells.append(make_md_cell("""## 15. Automatic Mixed Precision (AMP)
Enabling FP16 execution on Tensor Cores with dynamic loss scaling for faster throughput and reduced memory consumption."""))

refined_cells.append(orig_cells[15])

# --- CELL 16: OPTIMIZER & LR SCHEDULER ---
refined_cells.append(make_md_cell("""## 16. Optimization Strategy: AdamW + Cosine Annealing with Linear Warmup
Linear warmup for the first 3 epochs stabilizes transformer attention weights, followed by half-cycle cosine decay down to minimum learning rate."""))

refined_cells.append(orig_cells[16])

# --- CELL 17: LOSS FUNCTION ---
refined_cells.append(make_md_cell("""## 17. Regularized Cross-Entropy Loss
Cross-entropy criterion with label smoothing ($\epsilon = 0.05$) to mitigate overconfidence and enhance generalization."""))

refined_cells.append(orig_cells[17])

# --- CELL 18: TRAINING LOOP ---
refined_cells.append(make_md_cell("""## 18. Model Training & Early Stopping Loop
Full training loop with gradient clipping ($\|\mathbf{g}\|_2 \le 1.0$), real-time validation tracking (loss, accuracy, macro-F1), and atomic checkpointing."""))

refined_cells.append(orig_cells[18])

# --- CELL 19: TRAINING CURVES ---
refined_cells.append(make_md_cell("""## 19. Convergence Diagnostics: Loss, Accuracy, F1 & Learning Rate
Plotting learning trajectories to diagnose over-fitting, under-fitting, and optimization stability."""))

refined_cells.append(orig_cells[19])

# --- CELL 20: FINAL TEST EVALUATION ---
refined_cells.append(make_md_cell("""## 20. Unbiased Test Set Evaluation
Loading the highest validation macro-F1 checkpoint to evaluate generalization on the untouched test partition."""))

refined_cells.append(orig_cells[20])

# --- CELL 21: CONFUSION MATRIX ---
refined_cells.append(make_md_cell("""## 21. Confusion Matrix & Per-Class Diagnostic Performance
Detailed breakdown of true positives, false positives, false negatives, sensitivity, and specificity across all four tumor categories."""))

refined_cells.append(orig_cells[21])

# --- CELL 22: ROC-AUC ---
refined_cells.append(make_md_cell("""## 22. Receiver Operating Characteristic (ROC) & Multi-Class AUC
One-vs-Rest ROC curves evaluating discriminatory power across different clinical decision thresholds."""))

refined_cells.append(orig_cells[22])

# --- CELL 23: PRECISION-RECALL CURVES ---
refined_cells.append(make_md_cell("""## 23. Precision-Recall Curves & Average Precision (AP)
Evaluating precision-recall trade-offs especially valuable under skewed operational clinical priors."""))

refined_cells.append(orig_cells[23])

# --- CELL 24: CALIBRATION & ECE ---
refined_cells.append(make_md_cell("""## 24. Probability Calibration & Expected Calibration Error (ECE)
Assessing clinical trustworthiness: do predicted probabilities accurately mirror real-world empirical risk?"""))

refined_cells.append(orig_cells[24])

# --- CELL 25: CONFIDENCE ANALYSIS ---
refined_cells.append(make_md_cell("""## 25. Diagnostic Confidence & Uncertainty Profiling
Stratifying model predictions by entropy and softmax confidence."""))

refined_cells.append(orig_cells[25])

# --- CELL 26: ERROR ANALYSIS ---
refined_cells.append(make_md_cell("""## 26. Clinical Error Analysis: High-Confidence vs Borderline Misclassifications
Visual inspection of failure modes to identify subtle visual ambiguities, mass-effect edge cases, and atypical scan patterns."""))

refined_cells.append(orig_cells[26])

# --- CELL 27: EXPLAINABLE AI (GRAD-CAM) ---
refined_cells.append(make_md_cell("""## 27. Explainable AI (XAI): Grad-CAM Saliency Maps
Projecting spatial gradients from Swin Transformer's final stage feature maps back to image space to verify that the model attends to pathological tumor tissue rather than background noise."""))

refined_cells.append(orig_cells[27])

# --- CELL 28: ATTENTION ANALYSIS ---
refined_cells.append(make_md_cell("""## 28. Attention Map Profiling
Analyzing internal multi-head attention patterns across Swin stages."""))

refined_cells.append(orig_cells[28])

# --- CRITICAL REORDERING: CELL 32, 33, 34, 35 (PRIMARY RESULTS MUST SAVE FIRST) ---
refined_cells.append(make_md_cell("""## 29. Computational Resource & Performance Summary
Summary of total wall-clock training time, throughput per epoch, parameter count, and GPU VRAM utilization."""))

refined_cells.append(orig_cells[32])  # Resource summary

refined_cells.append(make_md_cell("""## 30. Consolidated Results Persistence
Saving all benchmark metrics, classification reports, and hardware footprints to disk in `results/`."""))

refined_cells.append(orig_cells[33])  # Save all results

refined_cells.append(make_md_cell("""## 31. Master Results Table
Executive tabulation of primary diagnostic metrics on the test cohort."""))

refined_cells.append(orig_cells[34])  # Final results table

refined_cells.append(make_md_cell("""## 32. Comprehensive Clinical & Research Summary
Formal research summary detailing data audit outcomes, architectural parameters, diagnostic sensitivity/specificity, and translation implications."""))

refined_cells.append(orig_cells[35])  # Research summary report

# --- OPTIONAL EXTENSIONS (BASELINES, ENSEMBLE, ABLATION) AT THE VERY END ---
refined_cells.append(make_md_cell("""---
# Extended Experimental Validation & Architectural Benchmarking
The following sections provide optional comparative benchmarks (ResNet-50 and ConvNeXt-Tiny), multi-architecture ensembling, and ablation studies. These are positioned at the end so all primary Swin-Tiny findings and checkpoints remain safely preserved."""))

refined_cells.append(make_md_cell("""### 33. Architectural Baseline Comparison
Comparative transfer learning across distinct model families:
- **ResNet-50**: Classic deep residual CNN architecture
- **ConvNeXt-Tiny**: Modernized pure convolutional network integrating ViT design elements
- **Swin-Tiny**: Hierarchical Shifted Window Transformer (Primary Architecture)"""))

# Cell 29 is the comparison cell
refined_cells.append(orig_cells[29])

refined_cells.append(make_md_cell("""### 34. Multi-Architecture Ensembling (Optional)
Evaluating soft voting / probability averaging between Swin-Tiny and ConvNeXt-Tiny."""))

refined_cells.append(orig_cells[30])

refined_cells.append(make_md_cell("""### 35. Component Ablation Study (Optional)
Ablating individual pipeline components (augmentations, label smoothing, loss weighting)."""))

refined_cells.append(orig_cells[31])

# Final conclusion markdown
refined_cells.append(make_md_cell("""---
## Conclusion & Production Readiness
This refined pipeline delivers:
- **Reliable End-to-End Execution**: Guaranteed checkpoint and metric persistence before optional benchmarks.
- **Leakage Immunity**: Perceptual hashing and duplicate group containment prevents test score inflation.
- **Clinically Interpretable Insights**: Grad-CAM overlays provide transparent reasoning for radiologist decision support.
- **Broad Portability**: Auto-resolves dataset locations across Kaggle GPU environments and local execution."""))

# Save refined notebook
new_nb = copy.deepcopy(nb)
new_nb['cells'] = refined_cells

with open('swin-tiny-refined.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1)

print(f"Successfully generated swin-tiny-refined.ipynb with {len(refined_cells)} cells!")
