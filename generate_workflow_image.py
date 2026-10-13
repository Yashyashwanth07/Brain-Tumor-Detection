"""
Generate pixel-perfect replica of the Brain Tumor Detection Workflow diagram
incorporating YOLOv12 as the proposed model with Area-Attention, Decoupled Head,
classification metrics, localization, and 100% correct spelling.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

PNG_OUT = r"c:\Users\bogga\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\Brain Tumor Detection\Images\workflow.png"
JPG_OUT = r"c:\Users\bogga\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\Brain Tumor Detection\Images\workflow.jpg"

# 1. Prepare panel crops safely from yolo_tumor.png
yolo_p = r"c:\Users\bogga\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\Brain Tumor Detection\Images\yolo_tumor.png"
if os.path.exists(yolo_p):
    yolo = Image.open(yolo_p)
    w, h = yolo.size
    cw = w // 3
    rh = h // 4
    yolo.crop((cw, 0, cw*2, rh)).resize((220, 220), Image.Resampling.LANCZOS).save(r"Images/panel_bbox.png")
    yolo.crop((cw*2, 0, w, rh)).resize((220, 220), Image.Resampling.LANCZOS).save(r"Images/panel_heat.png")
    yolo.crop((cw, rh, cw*2, rh*2)).resize((220, 220), Image.Resampling.LANCZOS).save(r"Images/panel_edge.png")
    yolo.crop((cw*2, rh, w, rh*2)).resize((220, 220), Image.Resampling.LANCZOS).save(r"Images/panel_mask.png")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "Helvetica"]

fig, ax = plt.subplots(figsize=(24, 16), dpi=220)
ax.set_xlim(0, 240)
ax.set_ylim(0, 160)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")

# -------------------------------------------------------------
# Main Title Banner
# -------------------------------------------------------------
ax.text(120, 151.5, "Brain Tumor Detection Workflow", 
        fontsize=27, fontweight="bold", ha="center", va="center", color="#0F172A")
ax.plot([75, 165], [146.5, 146.5], color="#334155", lw=1.8)

# -------------------------------------------------------------
# 5 Vertical Pillars Setup
# -------------------------------------------------------------
col_w = 40.5
col_gap = 4.2
base_x = 8.5

cols = [
    {
        "id": 1, "x": base_x, "w": col_w,
        "title": "Dataset Ingestion", "border": "#FBBF24", "hdr_bg": "#FEF3C7", "hdr_txt": "#1E293B",
        "bg": "#FFFBEB"
    },
    {
        "id": 2, "x": base_x + (col_w + col_gap), "w": col_w,
        "title": "Image Preprocessing", "border": "#38BDF8", "hdr_bg": "#E0F2FE", "hdr_txt": "#1E293B",
        "bg": "#F0F9FF"
    },
    {
        "id": 3, "x": base_x + (col_w + col_gap)*2 - 0.5, "w": 46.5,
        "title": "Offline Data Augmentation", "border": "#C2410C", "hdr_bg": "#FFEDD5", "hdr_txt": "#1E293B",
        "bg": "#FFF7ED"
    },
    {
        "id": 4, "x": base_x + (col_w + col_gap)*3 + 5.5, "w": 38,
        "title": "Evaluation", "border": "#A855F7", "hdr_bg": "#F3E8FF", "hdr_txt": "#1E293B",
        "bg": "#FAF5FF"
    },
    {
        "id": 5, "x": base_x + (col_w + col_gap)*4 + 3.5, "w": 39.5,
        "title": "Tumor Classification\n(Output Decision)", "border": "#4ADE80", "hdr_bg": "#DCFCE7", "hdr_txt": "#1E293B",
        "bg": "#F0FDF4"
    }
]

pillar_y = 26
pillar_h = 113

for c in cols:
    pillar = patches.FancyBboxPatch(
        (c["x"], pillar_y), c["w"], pillar_h,
        boxstyle="round,pad=0.2,rounding_size=2.2",
        facecolor=c["bg"], edgecolor=c["border"], linewidth=2.0, zorder=2
    )
    ax.add_patch(pillar)
    
    h_h = 13.5 if "\n" in c["title"] else 9.5
    h_y = pillar_y + pillar_h - h_h
    hdr = patches.FancyBboxPatch(
        (c["x"] + 0.2, h_y), c["w"] - 0.4, h_h,
        boxstyle="round,pad=0.1,rounding_size=1.8",
        facecolor=c["hdr_bg"], edgecolor="none", zorder=3
    )
    ax.add_patch(hdr)
    ax.text(c["x"] + c["w"]/2, h_y + h_h/2, c["title"],
            fontsize=13.5, fontweight="bold", ha="center", va="center", color=c["hdr_txt"], zorder=4)

# =============================================================
# COLUMN 1: DATASET INGESTION
# =============================================================
c1 = cols[0]
folder_icon_p = r"Images/folder_icon.png"
if os.path.exists(folder_icon_p):
    im_folder = Image.open(folder_icon_p)
    oi_f = OffsetImage(im_folder, zoom=0.68)
    ab_f = AnnotationBbox(oi_f, (c1["x"] + c1["w"]/2, 88), frameon=False, zorder=5)
    ax.add_artist(ab_f)

ax.text(c1["x"] + c1["w"]/2, 68, "Dataset\nMerging & Annotation", 
        fontsize=13.5, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=5)

b1 = patches.FancyBboxPatch(
    (c1["x"] - 0.5, 7), c1["w"] + 1.0, 14,
    boxstyle="round,pad=0.2,rounding_size=1.2",
    facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=1.5, zorder=4
)
ax.add_patch(b1)
ax.text(c1["x"] + c1["w"]/2, 14, "Dataset\nMerging & Annotation", 
        fontsize=11.5, fontweight="bold", ha="center", va="center", color="#78350F", zorder=5)

ax.annotate("", xy=(c1["x"] + c1["w"]/2, 21.5), xytext=(c1["x"] + c1["w"]/2, 25.5),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.2, mutation_scale=16), zorder=5)

ax.annotate("", xy=(cols[1]["x"] - 0.3, 76), xytext=(c1["x"] + c1["w"] + 0.3, 76),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.4, mutation_scale=18), zorder=6)

# =============================================================
# COLUMN 2: IMAGE PREPROCESSING
# =============================================================
c2 = cols[1]

p_steps = [
    ("Noise\nReduction", 114),
    ("Brightness\nIncrease", 97),
    ("Sharpness\nEnhancement", 80),
    ("Edge\nEnhancement", 63)
]

for p_text, p_y in p_steps:
    pbox = patches.FancyBboxPatch(
        (c2["x"] + 3.5, p_y - 6), c2["w"] - 7, 12.5,
        boxstyle="round,pad=0.2,rounding_size=1.2",
        facecolor="#1E4E79", edgecolor="#0C2D48", linewidth=1.8, zorder=5
    )
    ax.add_patch(pbox)
    ax.text(c2["x"] + c2["w"]/2, p_y + 0.3, p_text,
            fontsize=11.2, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=6)

for y_start in [107.5, 90.5, 73.5]:
    ax.annotate("", xy=(c2["x"] + c2["w"]/2, y_start - 3.0), xytext=(c2["x"] + c2["w"]/2, y_start - 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.0, mutation_scale=14), zorder=6)

b2 = patches.FancyBboxPatch(
    (c2["x"] + 2.5, 30.5), c2["w"] - 5, 12,
    boxstyle="round,pad=0.2,rounding_size=1.2",
    facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=1.5, zorder=5
)
ax.add_patch(b2)
ax.text(c2["x"] + c2["w"]/2, 36.5, "Image\nPreprocessing",
        fontsize=12, fontweight="bold", ha="center", va="center", color="#0369A1", zorder=6)

ax.annotate("", xy=(c2["x"] + c2["w"]/2, 43), xytext=(c2["x"] + c2["w"]/2, 56.5),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.2, mutation_scale=16), zorder=6)

for y_conn in [114, 76, 36.5]:
    ax.annotate("", xy=(cols[2]["x"] - 0.3, y_conn), xytext=(c2["x"] + c2["w"] + 0.3, y_conn),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.2, mutation_scale=16), zorder=6)

# =============================================================
# COLUMN 3: PROPOSED MODEL (YOLOv12) & OFFLINE AUGMENTATION
# =============================================================
c3 = cols[2]

arch_w = c3["w"] - 4.5
arch_x = c3["x"] + 2.25
arch_y = 66
arch_h = 67

abox = patches.FancyBboxPatch(
    (arch_x, arch_y), arch_w, arch_h,
    boxstyle="round,pad=0.2,rounding_size=1.5",
    facecolor="#FFFFFF", edgecolor="#FB923C", linewidth=1.5, zorder=4
)
ax.add_patch(abox)

ax.text(arch_x + arch_w/2, arch_y + arch_h - 4.2, "Proposed Model: YOLOv12\n4-Class Classifier & Detector",
        fontsize=11.6, fontweight="bold", ha="center", va="center", color="#C2410C", zorder=5)

arch_rows = [
    ("Image Preprocessing", "#1E4E79", "640×640", 118.5),
    ("YOLOv12 Backbone", "#15803D", "A-EFDN", 106.5),
    ("Area-Attention (A²)", "#0284C7", "FlashAttn", 94.5),
    ("R-ELAN Neck & SPPF", "#1E4E79", "MultiScale", 82.5),
]

for r_name, r_col, r_code, r_y in arch_rows:
    r_patch = patches.FancyBboxPatch(
        (arch_x + 2, r_y - 4.2), 22, 8.5,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        facecolor=r_col, edgecolor="none", zorder=5
    )
    ax.add_patch(r_patch)
    ax.text(arch_x + 13, r_y, r_name, fontsize=8.6, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=6)
    
    ax.annotate("", xy=(arch_x + 26.5, r_y), xytext=(arch_x + 24.5, r_y),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=1.6, mutation_scale=10), zorder=6)
    
    m_patch = patches.FancyBboxPatch(
        (arch_x + 27, r_y - 3.8), 8.5, 7.6,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor="#0284C7", edgecolor="none", zorder=5
    )
    ax.add_patch(m_patch)
    ax.text(arch_x + 31.25, r_y, r_code, fontsize=7.2, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=6)
    
    ax.annotate("", xy=(arch_x + 38, r_y), xytext=(arch_x + 36, r_y),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=1.6, mutation_scale=10), zorder=6)

sample_scan_p = r"Dataset_MRI_Brain_Tumor/Dataset_MRI_Brain_Tumor/glioma/ds1_Testing_glioma_Te-glTr_0000.jpg"
if os.path.exists(sample_scan_p):
    im_sc = Image.open(sample_scan_p).resize((60, 60))
    for r_y in [118.5, 106.5, 94.5, 82.5]:
        oi_s = OffsetImage(im_sc, zoom=0.35)
        ab_s = AnnotationBbox(oi_s, (arch_x + 39.5, r_y), frameon=True,
                              bboxprops=dict(edgecolor="#0284C7", facecolor="black", boxstyle="square,pad=0", lw=1.2), zorder=6)
        ax.add_artist(ab_s)

ax.annotate("", xy=(arch_x + 42.0, 106.5), xytext=(arch_x + 42.0, 118.5),
            arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.8, connectionstyle="arc3,rad=-0.5"), zorder=7)
ax.text(arch_x + 44.0, 112.5, "60°", fontsize=7.6, fontweight="bold", color="#0284C7", ha="left", zorder=8)

ax.annotate("", xy=(arch_x + 42.0, 82.5), xytext=(arch_x + 42.0, 94.5),
            arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.8, connectionstyle="arc3,rad=-0.5"), zorder=7)
ax.text(arch_x + 44.0, 88.5, "120°", fontsize=7.6, fontweight="bold", color="#0284C7", ha="left", zorder=8)

# Decoupled Head for YOLOv12
head_y = 71.5
h_patch = patches.FancyBboxPatch(
    (arch_x + 2, head_y - 4.2), 22, 8.5,
    boxstyle="round,pad=0.1,rounding_size=0.8",
    facecolor="#1E4E79", edgecolor="none", zorder=5
)
ax.add_patch(h_patch)
ax.text(arch_x + 13, head_y, "Decoupled Head", fontsize=8.4, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=6)

h_m = patches.FancyBboxPatch(
    (arch_x + 25.5, head_y - 3.8), 16.5, 7.6,
    boxstyle="round,pad=0.1,rounding_size=0.6",
    facecolor="#15803D", edgecolor="none", zorder=5
)
ax.add_patch(h_m)
ax.text(arch_x + 33.75, head_y, "Classify + Detect", fontsize=7.5, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=6)

# Lower Augmentation Section
ax.text(c3["x"] + c3["w"]/2, 59.5, "Dataset Expansion (4× original size):",
        fontsize=11.5, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=5)

ax.text(c3["x"] + c3["w"]/2, 53.5, "8229  ➔  32916 images",
        fontsize=14, fontweight="heavy", ha="center", va="center", color="#EA580C", zorder=5)

if os.path.exists(sample_scan_p):
    im_base = Image.open(sample_scan_p).resize((85, 85))
    transforms = [
        (c3["x"] + 8, "Rotate 60°", im_base.rotate(60)),
        (c3["x"] + 23.25, "Rotate 120°", im_base.rotate(120)),
        (c3["x"] + 38.5, "Flip", im_base.transpose(Image.FLIP_LEFT_RIGHT))
    ]
    for tx, tlabel, timg in transforms:
        oi_t = OffsetImage(timg, zoom=0.55)
        ab_t = AnnotationBbox(oi_t, (tx, 39), frameon=True,
                              bboxprops=dict(edgecolor="#475569", facecolor="black", boxstyle="square,pad=0", lw=1.2), zorder=5)
        ax.add_artist(ab_t)
        ax.text(tx, 27.5, tlabel, fontsize=9.2, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=6)

ax.annotate("", xy=(cols[3]["x"] - 0.3, 118.5), xytext=(c3["x"] + c3["w"] + 0.3, 118.5),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.4, mutation_scale=18), zorder=7)

ax.annotate("", xy=(cols[3]["x"] - 0.3, 76), xytext=(c3["x"] + c3["w"] + 0.3, 76),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.4, mutation_scale=18), zorder=7)

ax.annotate("", xy=(cols[4]["x"] - 0.3, 76), xytext=(cols[3]["x"] + cols[3]["w"] + 0.3, 76),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.4, mutation_scale=18), zorder=7)

# =============================================================
# COLUMN 4: EVALUATION & CLASSIFICATION METRICS
# =============================================================
c4 = cols[3]

eval1 = patches.FancyBboxPatch(
    (c4["x"] + 2, 85), c4["w"] - 4, 52,
    boxstyle="round,pad=0.2,rounding_size=1.5",
    facecolor="#FFFFFF", edgecolor="#C084FC", linewidth=1.5, zorder=4
)
ax.add_patch(eval1)

bar_p = r"Images/mini_metrics_bar.png"
if os.path.exists(bar_p):
    im_b = Image.open(bar_p)
    oi_b = OffsetImage(im_b, zoom=0.55)
    ab_b = AnnotationBbox(oi_b, (c4["x"] + c4["w"]/2, 122), frameon=False, zorder=5)
    ax.add_artist(ab_b)

eval_metrics = [
    ("Accuracy", "99.4%"),
    ("Precision", "99.2%"),
    ("Recall", "99.1%"),
    ("F1-Score", "99.1%")
]
ey = 104
for m_name, m_val in eval_metrics:
    ax.text(c4["x"] + 6, ey, m_name, fontsize=10, fontweight="bold", color="#1E293B", va="center", zorder=6)
    ax.text(c4["x"] + c4["w"] - 6, ey, m_val, fontsize=10, fontweight="bold", color="#15803D", ha="right", va="center", zorder=6)
    ey -= 5.8

roc_p = r"Images/mini_roc.png"
if os.path.exists(roc_p):
    im_r = Image.open(roc_p)
    oi_r = OffsetImage(im_r, zoom=0.58)
    ab_r = AnnotationBbox(oi_r, (c4["x"] + c4["w"]/2, 70), frameon=True,
                          bboxprops=dict(edgecolor="#374151", facecolor="#111827", boxstyle="square,pad=0.05", lw=1.2), zorder=5)
    ax.add_artist(ab_r)

eval2 = patches.FancyBboxPatch(
    (c4["x"] + 2, 28.5), c4["w"] - 4, 25.5,
    boxstyle="round,pad=0.2,rounding_size=1.5",
    facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.5, zorder=4
)
ax.add_patch(eval2)

summary_badges = [
    ("99.4%", "Accuracy", 48.5),
    ("99.2%", "Precision", 41.5),
    ("99.1%", "F1-Score", 34.5)
]

for s_val, s_lbl, s_y in summary_badges:
    p_box = patches.FancyBboxPatch(
        (c4["x"] + 4.5, s_y - 2.8), 12, 5.8,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor="#10B981", edgecolor="none", zorder=5
    )
    ax.add_patch(p_box)
    ax.text(c4["x"] + 10.5, s_y + 0.1, s_val, fontsize=9.2, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=6)
    ax.text(c4["x"] + 18.5, s_y + 0.1, s_lbl, fontsize=9.8, fontweight="bold", va="center", color="#1E293B", zorder=6)

# =============================================================
# COLUMN 5: TUMOR CLASSIFICATION & LOCALIZATION (OUTPUT DECISION)
# =============================================================
c5 = cols[4]

out_box = patches.FancyBboxPatch(
    (c5["x"] + 2, 28), c5["w"] - 4, 109,
    boxstyle="round,pad=0.2,rounding_size=1.5",
    facecolor="#FFFFFF", edgecolor="#86EFAC", linewidth=1.5, zorder=4
)
ax.add_patch(out_box)

panels = [
    (r"Images/panel_bbox.png", c5["x"] + 11.5, 114),
    (r"Images/panel_heat.png", c5["x"] + 28, 114),
    (r"Images/panel_edge.png", c5["x"] + 11.5, 87),
    (r"Images/panel_mask.png", c5["x"] + 28, 87)
]

for p_path, px, py in panels:
    if os.path.exists(p_path):
        p_img = Image.open(p_path)
        oi_p = OffsetImage(p_img, zoom=0.46)
        ab_p = AnnotationBbox(oi_p, (px, py), frameon=True,
                              bboxprops=dict(edgecolor="#475569", facecolor="black", boxstyle="square,pad=0", lw=1.2), zorder=5)
        ax.add_artist(ab_p)

ax.text(c5["x"] + 28, 126, "Tumor Region", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=6)
ax.text(c5["x"] + 28, 75.5, "Edge Output", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=6)

d_card = patches.FancyBboxPatch(
    (c5["x"] + 4, 32), c5["w"] - 8, 38,
    boxstyle="round,pad=0.15,rounding_size=1.0",
    facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=1.2, zorder=5
)
ax.add_patch(d_card)

dec_classes = [
    ("Glioma", "#DC2626"),
    ("Meningioma", "#2563EB"),
    ("Pituitary", "#D97706"),
    ("No Tumor", "#16A34A")
]
d_y = 63.5
for d_name, d_col in dec_classes:
    dot = patches.Circle((c5["x"] + 8.5, d_y), 0.9, facecolor=d_col, edgecolor="none", zorder=6)
    ax.add_patch(dot)
    ax.text(c5["x"] + 11.5, d_y, d_name, fontsize=10.5, fontweight="bold", color="#1E293B", va="center", zorder=6)
    d_y -= 8.2

ax.annotate("", xy=(c5["x"] + c5["w"]/2, 17), xytext=(c5["x"] + c5["w"]/2, 26),
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.4, mutation_scale=18), zorder=6)

plt.tight_layout()
plt.savefig(PNG_OUT, dpi=220, bbox_inches="tight")

im_saved = Image.open(PNG_OUT)
im_saved.convert("RGB").save(JPG_OUT, "JPEG", quality=95)

print("SUCCESS: YOLOv12 workflow image saved to Images/workflow.png and Images/workflow.jpg!")
