"""
Generate high-resolution End-to-End Methodology diagram for Brain Tumor Detection.
100% correct spelling, precise technical syntax and semantics, publication-quality graphics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageEnhance, ImageFilter

PNG_OUT = r"c:\Users\bogga\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\Brain Tumor Detection\Images\end_to_end_methodology.png"
JPG_OUT = r"c:\Users\bogga\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\Brain Tumor Detection\Images\end_to_end_methodology.jpg"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Segoe UI"]

fig, ax = plt.subplots(figsize=(26, 15), dpi=220)
ax.set_xlim(0, 240)
ax.set_ylim(0, 150)
ax.axis("off")
fig.patch.set_facecolor("#F8FAFC")

# Title Banner
title_box = patches.FancyBboxPatch(
    (5, 137), 230, 10.5,
    boxstyle="round,pad=0.2,rounding_size=1.5",
    facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5,
    zorder=2
)
ax.add_patch(title_box)

ax.text(120, 143.8, "Brain Tumor Detection & Classification Workflow", 
        fontsize=24, fontweight="bold", ha="center", va="center", color="#0F172A")
ax.text(120, 139.5, "End-to-End Deep Learning Framework: Preprocessing • Hybrid ResNet50-CBAM-BiFPN • YOLOv8 Localization • Grad-CAM XAI", 
        fontsize=12, fontweight="medium", ha="center", va="center", color="#475569")

# 5 Columns (Cards)
cols = [
    {"x": 6,   "w": 43, "bg": "#FFFBEB", "border": "#D97706", "hdr_bg": "#FDE68A", "hdr_txt": "#92400E", "title": "1. Dataset Ingestion"},
    {"x": 52.5,"w": 43, "bg": "#F0F9FF", "border": "#0284C7", "hdr_bg": "#BAE6FD", "hdr_txt": "#075985", "title": "2. Image Preprocessing"},
    {"x": 99,  "w": 44, "bg": "#FFF7ED", "border": "#EA580C", "hdr_bg": "#FED7AA", "hdr_txt": "#9A3412", "title": "3. Augmentation & Model"},
    {"x": 146.5,"w": 43, "bg": "#FAF5FF", "border": "#7C3AED", "hdr_bg": "#DDD6FE", "hdr_txt": "#5B21B6", "title": "4. Performance Evaluation"},
    {"x": 193, "w": 43, "bg": "#F0FDF4", "border": "#16A34A", "hdr_bg": "#BBF7D0", "hdr_txt": "#166534", "title": "5. Clinical Output Decision"}
]

card_y = 6
card_h = 127

for col in cols:
    outer_card = patches.FancyBboxPatch(
        (col["x"], card_y), col["w"], card_h,
        boxstyle="round,pad=0.2,rounding_size=2.0",
        facecolor=col["bg"], edgecolor=col["border"], linewidth=2.0,
        zorder=2
    )
    ax.add_patch(outer_card)
    
    hdr_box = patches.FancyBboxPatch(
        (col["x"]+0.8, card_y + card_h - 9), col["w"]-1.6, 8,
        boxstyle="round,pad=0.2,rounding_size=1.2",
        facecolor=col["hdr_bg"], edgecolor=col["border"], linewidth=1.2,
        zorder=3
    )
    ax.add_patch(hdr_box)
    ax.text(col["x"] + col["w"]/2, card_y + card_h - 5, col["title"],
            fontsize=13, fontweight="bold", ha="center", va="center", color=col["hdr_txt"])

# Connecting Arrows between Cards
arrow_y = 70
for i in range(4):
    ax1 = cols[i]["x"] + cols[i]["w"] + 0.3
    ax2 = cols[i+1]["x"] - 0.3
    ax.annotate("", xy=(ax2, arrow_y), xytext=(ax1, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.5, mutation_scale=18),
                zorder=10)

def add_subcard(x, y, w, h, bg="#FFFFFF", border="#CBD5E1", lw=1.2, r=1.0):
    patch = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={r}",
        facecolor=bg, edgecolor=border, linewidth=lw, zorder=4
    )
    ax.add_patch(patch)
    return patch

dataset_base = r"Dataset_MRI_Brain_Tumor/Dataset_MRI_Brain_Tumor"

# ==========================================
# COLUMN 1: DATASET INGESTION
# ==========================================
c1 = cols[0]
cx = c1["x"]

add_subcard(cx+2, 95, 39, 26, bg="#FFFFFF", border="#FCD34D")
ax.text(cx+21.5, 117.5, "Kaggle MRI Dataset Merging", fontsize=11, fontweight="bold", ha="center", color="#92400E", zorder=5)
ax.text(cx+21.5, 112.5, "Combined Multi-Source Clinical Scans", fontsize=8.5, ha="center", color="#78350F", zorder=5)
ax.text(cx+5, 107, "• Total Scans: 10,287 Images", fontsize=9, fontweight="bold", color="#1E293B", zorder=5)
ax.text(cx+5, 102.5, "• Split Ratio: 80% Train | 20% Test", fontsize=8.5, color="#334155", zorder=5)
ax.text(cx+5, 98, "• Resolution: 224 × 224 × 3 RGB", fontsize=8.5, color="#334155", zorder=5)

add_subcard(cx+2, 60, 39, 32, bg="#FFFFFF", border="#FCD34D")
ax.text(cx+21.5, 88.5, "Balanced 4-Class Breakdown", fontsize=10.5, fontweight="bold", ha="center", color="#92400E", zorder=5)

classes_info = [
    ("Glioma Tumor", "2,547 Scans", "#DC2626"),
    ("Meningioma Tumor", "2,582 Scans", "#2563EB"),
    ("Pituitary Tumor", "2,658 Scans", "#D97706"),
    ("No Tumor (Normal)", "2,500 Scans", "#16A34A"),
]
table_y = 82
for cls_name, cls_cnt, cls_col in classes_info:
    bullet = patches.Circle((cx+5, table_y), 0.9, facecolor=cls_col, edgecolor="none", zorder=5)
    ax.add_patch(bullet)
    ax.text(cx+7.5, table_y, cls_name, fontsize=8.5, fontweight="semibold", color="#1E293B", va="center", zorder=5)
    ax.text(cx+39, table_y, cls_cnt, fontsize=8.5, fontweight="bold", color=cls_col, ha="right", va="center", zorder=5)
    table_y -= 6

add_subcard(cx+2, 10, 39, 47, bg="#FFFFFF", border="#FCD34D")
ax.text(cx+21.5, 53.5, "Sample MRI Modalities", fontsize=10.5, fontweight="bold", ha="center", color="#92400E", zorder=5)

samples = [
    ("glioma", "Glioma", cx+11.5, 40),
    ("meningioma", "Meningioma", cx+31.5, 40),
    ("pituitary", "Pituitary", cx+11.5, 21),
    ("no_tumor", "No Tumor", cx+31.5, 21),
]

for folder, label, sx, sy in samples:
    fdir = os.path.join(dataset_base, folder)
    if os.path.exists(fdir):
        flist = os.listdir(fdir)
        if flist:
            img_p = os.path.join(fdir, flist[0])
            try:
                raw_img = Image.open(img_p).resize((110, 110))
                oi = OffsetImage(raw_img, zoom=0.62)
                ab = AnnotationBbox(oi, (sx, sy+2), frameon=True,
                                    bboxprops=dict(edgecolor="#94A3B8", facecolor="black", boxstyle="square,pad=0", lw=1.2), zorder=5)
                ax.add_artist(ab)
                ax.text(sx, sy-8.5, label, fontsize=8.2, fontweight="bold", ha="center", color="#1E293B", zorder=6)
            except Exception as e:
                print("Error loading scan:", e)

# ==========================================
# COLUMN 2: IMAGE PREPROCESSING
# ==========================================
c2 = cols[1]
cx2 = c2["x"]

add_subcard(cx2+2, 69, 39, 52, bg="#FFFFFF", border="#38BDF8")
ax.text(cx2+21.5, 117.5, "Sequential Enhancement Pipeline", fontsize=11, fontweight="bold", ha="center", color="#0369A1", zorder=5)

filters = [
    ("1. Noise Suppression", "Gaussian Blur Filter (σ=1.0)", "Removes high-frequency scanner noise", "#0284C7"),
    ("2. Contrast Adjustment", "Brightness Increase (+20%)", "Improves illumination in dark tumor regions", "#0284C7"),
    ("3. Sharpness Enhancement", "Sharpening Filter (+50%)", "Clarifies boundary texture and tissue edges", "#0284C7"),
    ("4. Edge Enhancement", "Boundary Gradient Contouring", "Emphasizes microvascular tumor margins", "#0284C7"),
]

fy = 110.5
for ftitle, ftech, fdesc, fcol in filters:
    fbox = patches.FancyBboxPatch(
        (cx2+3.5, fy-8), 36, 8.5,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        facecolor="#F0F9FF", edgecolor=fcol, linewidth=1.0, zorder=5
    )
    ax.add_patch(fbox)
    ax.text(cx2+5, fy-1.8, ftitle, fontsize=8.5, fontweight="bold", color="#0C4A6E", zorder=6)
    ax.text(cx2+5, fy-4.6, ftech, fontsize=7.5, fontweight="semibold", color="#0284C7", zorder=6)
    ax.text(cx2+5, fy-7.0, fdesc, fontsize=6.8, color="#475569", zorder=6)
    fy -= 10.2

add_subcard(cx2+2, 10, 39, 56, bg="#FFFFFF", border="#38BDF8")
ax.text(cx2+21.5, 62.5, "Raw vs Preprocessed MRI", fontsize=10.5, fontweight="bold", ha="center", color="#0369A1", zorder=5)

sample_mri_p = os.path.join(dataset_base, "glioma", os.listdir(os.path.join(dataset_base, "glioma"))[0])
enh_pil = None
if os.path.exists(sample_mri_p):
    orig_pil = Image.open(sample_mri_p).convert("RGB").resize((130, 130))
    enh_pil = orig_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    enh_pil = ImageEnhance.Brightness(enh_pil).enhance(1.2)
    enh_pil = ImageEnhance.Sharpness(enh_pil).enhance(1.5)
    
    oi1 = OffsetImage(orig_pil, zoom=0.68)
    ab1 = AnnotationBbox(oi1, (cx2+11.5, 38), frameon=True,
                         bboxprops=dict(edgecolor="#EF4444", facecolor="black", boxstyle="square,pad=0", lw=1.5), zorder=5)
    ax.add_artist(ab1)
    ax.text(cx2+11.5, 23.5, "Original Scan\n(Low Contrast)", fontsize=8, fontweight="bold", ha="center", color="#991B1B", zorder=6)
    
    ax.annotate("", xy=(cx2+22.8, 38), xytext=(cx2+19.2, 38),
                arrowprops=dict(arrowstyle="-|>", color="#0284C7", lw=2.5, mutation_scale=14), zorder=6)
    
    oi2 = OffsetImage(enh_pil, zoom=0.68)
    ab2 = AnnotationBbox(oi2, (cx2+31.5, 38), frameon=True,
                         bboxprops=dict(edgecolor="#10B981", facecolor="black", boxstyle="square,pad=0", lw=1.5), zorder=5)
    ax.add_artist(ab2)
    ax.text(cx2+31.5, 23.5, "Enhanced Scan\n(Delineated Margins)", fontsize=8, fontweight="bold", ha="center", color="#065F46", zorder=6)

ax.text(cx2+21.5, 14, "Standardized Input Tensor: 224 × 224 × 3", fontsize=8, fontweight="bold", ha="center", color="#0369A1", zorder=6)

# ==========================================
# COLUMN 3: DATA AUGMENTATION & ARCHITECTURE
# ==========================================
c3 = cols[2]
cx3 = c3["x"]

add_subcard(cx3+2, 83, 40, 38, bg="#FFFFFF", border="#FB923C")
ax.text(cx3+22, 117.5, "Offline Dataset Augmentation (4×)", fontsize=11, fontweight="bold", ha="center", color="#9A3412", zorder=5)
ax.text(cx3+22, 113.0, "8,229 Train Scans  →  32,916 Total Scans", fontsize=9.5, fontweight="heavy", ha="center", color="#EA580C", zorder=5)

aug_x = [cx3+8.5, cx3+22, cx3+35.5]
if enh_pil is not None:
    aug_ops = [
        ("Rotate 60°", enh_pil.rotate(60)),
        ("Rotate 120°", enh_pil.rotate(120)),
        ("Horiz. Flip", enh_pil.transpose(Image.FLIP_LEFT_RIGHT)),
    ]
    for ax_pos, (alabel, aimg) in zip(aug_x, aug_ops):
        oi = OffsetImage(aimg.resize((90, 90)), zoom=0.55)
        ab = AnnotationBbox(oi, (ax_pos, 98), frameon=True,
                            bboxprops=dict(edgecolor="#FDBA74", facecolor="black", boxstyle="square,pad=0", lw=1.2), zorder=5)
        ax.add_artist(ab)
        ax.text(ax_pos, 87.5, alabel, fontsize=7.8, fontweight="bold", ha="center", color="#431407", zorder=6)

add_subcard(cx3+2, 10, 40, 70, bg="#FFFFFF", border="#FB923C")
ax.text(cx3+22, 76.5, "Hybrid 4-Class Deep Architecture", fontsize=11, fontweight="bold", ha="center", color="#9A3412", zorder=5)

arch_blocks = [
    ("1. ResNet-50 Backbone", "Deep Feature Extractor (Conv1 - Stage 4)", "#1E293B", "#F1F5F9"),
    ("2. 1×1 Conv Reduction", "Projects multi-level channels to 256", "#1E293B", "#F1F5F9"),
    ("3. CBAM Attention Module", "Channel Attention + Spatial Attention Maps", "#C2410C", "#FFEDD5"),
    ("4. SPPF+ Pooling Module", "Multi-Kernel Max Pooling (5×5, 9×9, 13×13)", "#0284C7", "#E0F2FE"),
    ("5. BiFPN Feature Fusion", "Bidirectional Top-Down & Bottom-Up Paths", "#7C3AED", "#F3E8FF"),
    ("6. Dense Classifier Head", "GAP (768-D) → Dense 512 → Dense 128 → Softmax", "#15803D", "#DCFCE7"),
]

by = 70.0
for btitle, bdesc, btext_c, bbg_c in arch_blocks:
    bbox = patches.FancyBboxPatch(
        (cx3+3.5, by-7.2), 37, 7.5,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        facecolor=bbg_c, edgecolor="#CBD5E1", linewidth=1.0, zorder=5
    )
    ax.add_patch(bbox)
    ax.text(cx3+5, by-2.3, btitle, fontsize=8.5, fontweight="bold", color=btext_c, zorder=6)
    ax.text(cx3+5, by-5.4, bdesc, fontsize=7.2, color="#475569", zorder=6)
    
    if by > 32:
        ax.annotate("", xy=(cx3+22, by-7.2), xytext=(cx3+22, by-9.0),
                    arrowprops=dict(arrowstyle="<-", color="#94A3B8", lw=1.5), zorder=6)
    by -= 9.5

ax.text(cx3+22, 13.5, "Trained with Focal Loss & OneCycleLR", fontsize=8.5, fontweight="bold", ha="center", color="#9A3412", zorder=6)

# ==========================================
# COLUMN 4: PERFORMANCE EVALUATION
# ==========================================
c4 = cols[3]
cx4 = c4["x"]

add_subcard(cx4+2, 73, 39, 48, bg="#FFFFFF", border="#C084FC")
ax.text(cx4+21.5, 117.5, "Evaluation Metrics (Test Set)", fontsize=11, fontweight="bold", ha="center", color="#6B21A8", zorder=5)

metrics_list = [
    ("Overall Test Accuracy", "99.37%", "#15803D"),
    ("Precision", "99.37%", "#0284C7"),
    ("Recall / Sensitivity", "99.36%", "#D97706"),
    ("Specificity", "99.79%", "#7C3AED"),
    ("F1-Score", "99.37%", "#DC2626"),
]

my = 111.0
for mlabel, mval, mcol in metrics_list:
    mbox = patches.FancyBboxPatch(
        (cx4+3.5, my-6.5), 36, 7.0,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor="#FAF5FF", edgecolor="#E9D5FF", linewidth=1.0, zorder=5
    )
    ax.add_patch(mbox)
    ax.text(cx4+5.5, my-3.2, mlabel, fontsize=8.5, fontweight="semibold", color="#1E293B", va="center", zorder=6)
    ax.text(cx4+37.5, my-3.2, mval, fontsize=10.0, fontweight="bold", color=mcol, ha="right", va="center", zorder=6)
    my -= 7.8

add_subcard(cx4+2, 10, 39, 60, bg="#FFFFFF", border="#C084FC")
ax.text(cx4+21.5, 66.5, "ROC-AUC & Confusion Matrix", fontsize=10.5, fontweight="bold", ha="center", color="#6B21A8", zorder=5)

roc_p = r"Images/AUC_roc.png"
cm_p = r"Images/confusion_matrix.png"

if os.path.exists(roc_p):
    im_roc = Image.open(roc_p)
    oi_roc = OffsetImage(im_roc, zoom=0.085)
    ab_roc = AnnotationBbox(oi_roc, (cx4+11.5, 47), frameon=True,
                            bboxprops=dict(edgecolor="#DDD6FE", facecolor="white", boxstyle="square,pad=0.05", lw=1.2), zorder=5)
    ax.add_artist(ab_roc)
    ax.text(cx4+11.5, 33, "ROC Curves (AUC ≥ 0.998)", fontsize=7.5, fontweight="bold", ha="center", color="#5B21B6", zorder=6)

if os.path.exists(cm_p):
    im_cm = Image.open(cm_p)
    oi_cm = OffsetImage(im_cm, zoom=0.065)
    ab_cm = AnnotationBbox(oi_cm, (cx4+31.5, 47), frameon=True,
                           bboxprops=dict(edgecolor="#DDD6FE", facecolor="white", boxstyle="square,pad=0.05", lw=1.2), zorder=5)
    ax.add_artist(ab_cm)
    ax.text(cx4+31.5, 33, "Confusion Matrix (Counts/%)", fontsize=7.5, fontweight="bold", ha="center", color="#5B21B6", zorder=6)

ax.text(cx4+21.5, 27.5, "Class-Wise Discrimination Power:", fontsize=8, fontweight="bold", ha="center", color="#4B5563", zorder=6)
auc_badges = [
    ("No Tumor: 1.000", cx4+12, 21),
    ("Pituitary: 1.000", cx4+31, 21),
    ("Meningioma: 0.999", cx4+12, 14.5),
    ("Glioma: 0.998", cx4+31, 14.5)
]
for atxt, axpos, aypos in auc_badges:
    abox = patches.FancyBboxPatch(
        (axpos-8.5, aypos-3.0), 17, 5.5,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        facecolor="#F3E8FF", edgecolor="#C084FC", linewidth=0.8, zorder=5
    )
    ax.add_patch(abox)
    ax.text(axpos, aypos-0.2, atxt, fontsize=7.2, fontweight="bold", ha="center", va="center", color="#6B21A8", zorder=6)

# ==========================================
# COLUMN 5: CLINICAL OUTPUT DECISION
# ==========================================
c5 = cols[4]
cx5 = c5["x"]

add_subcard(cx5+2, 69, 39, 52, bg="#FFFFFF", border="#4ADE80")
ax.text(cx5+21.5, 117.5, "Tumor Localization & XAI", fontsize=11, fontweight="bold", ha="center", color="#15803D", zorder=5)

yolo_sample_p = r"Images/yolo_confidence.png"
if os.path.exists(yolo_sample_p):
    im_yolo = Image.open(yolo_sample_p)
    oi_y = OffsetImage(im_yolo, zoom=0.17)
    ab_y = AnnotationBbox(oi_y, (cx5+21.5, 99), frameon=True,
                          bboxprops=dict(edgecolor="#86EFAC", facecolor="black", boxstyle="square,pad=0.05", lw=1.2), zorder=5)
    ax.add_artist(ab_y)

ax.text(cx5+12, 82.5, "Grad-CAM Heatmap\n(Explainable AI)", fontsize=7.8, fontweight="bold", ha="center", color="#047857", zorder=6)
ax.text(cx5+31, 82.5, "YOLOv8 Tight BBox\n(Tumor ROI: 99%)", fontsize=7.8, fontweight="bold", ha="center", color="#047857", zorder=6)

ax.text(cx5+21.5, 73.0, "U-Net Segmentation: IoU 0.452 | Dice 0.621", fontsize=7.5, fontweight="bold", ha="center", color="#1E293B", zorder=6)

add_subcard(cx5+2, 10, 39, 56, bg="#FFFFFF", border="#4ADE80")
ax.text(cx5+21.5, 62.5, "Final Diagnostic Output", fontsize=10.5, fontweight="bold", ha="center", color="#15803D", zorder=5)

decisions = [
    ("Glioma", "Detected (High Confidence: 99.4%)", "#DC2626", "#FEE2E2"),
    ("Meningioma", "Detected (High Confidence: 99.2%)", "#2563EB", "#DBEAFE"),
    ("Pituitary Tumor", "Detected (High Confidence: 100%)", "#D97706", "#FEF3C7"),
    ("No Tumor", "Normal Brain Scan Verified (100%)", "#16A34A", "#DCFCE7"),
]

dy = 54.5
for dname, dstatus, dcolor, dbg in decisions:
    dbox = patches.FancyBboxPatch(
        (cx5+3.5, dy-7), 36, 7.8,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        facecolor=dbg, edgecolor=dcolor, linewidth=1.2, zorder=5
    )
    ax.add_patch(dbox)
    ax.text(cx5+5.5, dy-2.2, dname, fontsize=8.8, fontweight="bold", color=dcolor, zorder=6)
    ax.text(cx5+5.5, dy-5.3, dstatus, fontsize=7.2, color="#334155", zorder=6)
    dy -= 9.5

ax.text(cx5+21.5, 14.5, "Clinical Decision Support System (CAD)", fontsize=8.2, fontweight="heavy", ha="center", color="#166534", zorder=6)

plt.tight_layout()
plt.savefig(PNG_OUT, dpi=220, bbox_inches="tight")

im_saved = Image.open(PNG_OUT)
im_saved.convert("RGB").save(JPG_OUT, "JPEG", quality=95)

print("Generated refined methodology image with enhanced visual assets!")
