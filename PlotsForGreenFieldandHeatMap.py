"""
Task 4: Urban Heat Island Analysis and Green Space Optimization
Visualization Script

This script produces 5 key visualizations from neighborhood-level
satellite data (Sentinel-2 NDVI/NDBI and Landsat 8 LST) for İzmir.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ============================================================
# SETUP
# ============================================================

# Output folder
os.makedirs("outputs", exist_ok=True)

# Load data
lst = pd.read_csv("izmir_mahalle_lst_final_latest_clean.csv")
s2 = pd.read_csv("izmir_mahalle_s2_final_latest_clean.csv")

# Drop duplicates and merge
df = pd.merge(
    s2[["name", "ilce_adi", "NDVI", "NDBI"]],
    lst[["name", "ilce_adi", "LST_C", "NDVI_L8", "NDBI_L8"]],
    on=["name", "ilce_adi"]
)

# City-wide LST mean (used for hotspot threshold)
LST_MEAN = df["LST_C"].mean()
LST_THRESHOLD = LST_MEAN + 2.0  # 2°C above mean = thermal hotspot

print(f"Total neighborhoods: {len(df)}")
print(f"City-wide LST mean: {LST_MEAN:.2f}°C")
print(f"Hotspot threshold (mean + 2°C): {LST_THRESHOLD:.2f}°C")
print("-" * 50)

# Color palette
PALETTE = {
    "red":    "#E63946",
    "orange": "#F4A261",
    "yellow": "#E9C46A",
    "green":  "#2A9D8F",
    "dark":   "#264653",
    "light":  "#F1FAEE",
    "gray":   "#ADB5BD",
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# ============================================================
# 1. CORRELATION MATRIX HEATMAP
# ============================================================
print("1/5 Generating correlation matrix...")

corr_cols = ["NDVI", "NDBI", "LST_C"]
corr_labels = ["NDVI (Sentinel-2)", "NDBI (Sentinel-2)", "LST °C (Landsat 8)"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor(PALETTE["light"])
ax.set_facecolor(PALETTE["light"])

mask = np.zeros_like(corr_matrix, dtype=bool)  # show all cells

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn_r",
    center=0,
    vmin=-1, vmax=1,
    linewidths=1.5,
    linecolor="white",
    annot_kws={"size": 15, "weight": "bold"},
    ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"}
)

ax.set_title(
    "Correlation Matrix: NDVI, NDBI and LST\n"
    "İzmir Neighborhood-Level Analysis  (n=1,074)",
    fontsize=12, fontweight="bold", pad=14, color=PALETTE["dark"]
)
ax.set_xticklabels(corr_labels, fontsize=10, rotation=15, ha="right")
ax.set_yticklabels(corr_labels, fontsize=10, rotation=0)

plt.tight_layout()
plt.savefig("outputs/1_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: outputs/1_correlation_matrix.png")


# ============================================================
# 2. SCATTER PLOT: NDVI vs LST  (colored by NDBI)
# ============================================================
print("2/5 Generating NDVI vs LST scatter plot...")

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(PALETTE["light"])
ax.set_facecolor(PALETTE["light"])

sc = ax.scatter(
    df["NDVI"], df["LST_C"],
    c=df["NDBI"],
    cmap="RdYlGn_r",
    alpha=0.55,
    s=18,
    edgecolors="none"
)

# Trend line
z = np.polyfit(df["NDVI"], df["LST_C"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["NDVI"].min(), df["NDVI"].max(), 200)
ax.plot(x_line, p(x_line), color=PALETTE["red"], lw=2,
        linestyle="--", label=f"Trend (slope={z[0]:.2f})")

# Threshold lines
ax.axhline(LST_THRESHOLD, color=PALETTE["red"], lw=1,
           linestyle=":", alpha=0.7, label=f"Hotspot threshold ({LST_THRESHOLD:.1f}°C)")
ax.axvline(0.2, color=PALETTE["orange"], lw=1,
           linestyle=":", alpha=0.7, label="Low vegetation (NDVI=0.2)")

cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("NDBI (Sentinel-2)", fontsize=10)

ax.set_xlabel("NDVI — Sentinel-2  (higher = more vegetation)", fontsize=11)
ax.set_ylabel("LST — Landsat 8  (°C)", fontsize=11)
ax.set_title(
    "NDVI vs Land Surface Temperature\n"
    "Each point = one neighborhood  |  Color = built-up intensity (NDBI)",
    fontsize=12, fontweight="bold", color=PALETTE["dark"]
)
ax.legend(fontsize=9, framealpha=0.5)

# Correlation annotation
r = df["NDVI"].corr(df["LST_C"])
ax.text(0.97, 0.97, f"Pearson r = {r:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plt.tight_layout()
plt.savefig("outputs/2_scatter_ndvi_vs_lst.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: outputs/2_scatter_ndvi_vs_lst.png")


# ============================================================
# 3. DISTRICT AVERAGE LST BAR CHART
# ============================================================
print("3/5 Generating district average LST bar chart...")

district_avg = (
    df.groupby("ilce_adi")["LST_C"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
district_avg.columns = ["District", "Avg_LST"]

# Color bars: red if above threshold, green otherwise
bar_colors = [
    PALETTE["red"] if v >= LST_THRESHOLD else PALETTE["green"]
    for v in district_avg["Avg_LST"]
]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(PALETTE["light"])
ax.set_facecolor(PALETTE["light"])

bars = ax.barh(
    district_avg["District"], district_avg["Avg_LST"],
    color=bar_colors, edgecolor="white", linewidth=0.5
)

# Value labels
for bar, val in zip(bars, district_avg["Avg_LST"]):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}°C", va="center", fontsize=8.5)

ax.axvline(LST_MEAN, color=PALETTE["dark"], lw=1.5,
           linestyle="--", label=f"City mean ({LST_MEAN:.1f}°C)")
ax.axvline(LST_THRESHOLD, color=PALETTE["red"], lw=1.5,
           linestyle=":", label=f"Hotspot threshold ({LST_THRESHOLD:.1f}°C)")

legend_patches = [
    mpatches.Patch(color=PALETTE["red"], label="Thermal hotspot (≥ mean + 2°C)"),
    mpatches.Patch(color=PALETTE["green"], label="Below threshold"),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color=PALETTE["dark"], lw=1.5, linestyle="--",
               label=f"City mean ({LST_MEAN:.1f}°C)"),
    plt.Line2D([0], [0], color=PALETTE["red"], lw=1.5, linestyle=":",
               label=f"Hotspot threshold ({LST_THRESHOLD:.1f}°C)")
], fontsize=9, loc="lower right")

ax.set_xlabel("Average Land Surface Temperature (°C)", fontsize=11)
ax.set_title(
    "Average LST by District — İzmir\n"
    "Source: Landsat 8 Level-2  |  2023–2024 Median Composite",
    fontsize=12, fontweight="bold", color=PALETTE["dark"]
)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/3_district_avg_lst_bar.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: outputs/3_district_avg_lst_bar.png")


# ============================================================
# 4. NDVI CATEGORY DISTRIBUTION (BAR)
# ============================================================
print("4/5 Generating NDVI category distribution...")

def ndvi_category(v):
    if v < 0.2:
        return "Low (<0.2)\nInsufficient green cover"
    elif v < 0.4:
        return "Moderate (0.2–0.4)\nPartial green cover"
    else:
        return "High (>0.4)\nHealthy vegetation"

df["NDVI_Category"] = df["NDVI"].apply(ndvi_category)

cat_order = [
    "Low (<0.2)\nInsufficient green cover",
    "Moderate (0.2–0.4)\nPartial green cover",
    "High (>0.4)\nHealthy vegetation",
]
cat_colors = [PALETTE["red"], PALETTE["yellow"], PALETTE["green"]]
cat_counts = df["NDVI_Category"].value_counts().reindex(cat_order)
cat_pct = (cat_counts / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(PALETTE["light"])
ax.set_facecolor(PALETTE["light"])

bars = ax.bar(cat_order, cat_counts, color=cat_colors,
              edgecolor="white", linewidth=1, width=0.55)

for bar, cnt, pct in zip(bars, cat_counts, cat_pct):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{cnt}\n({pct}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Number of Neighborhoods", fontsize=11)
ax.set_title(
    "Neighborhood Distribution by NDVI Category — İzmir\n"
    "Source: Sentinel-2 MSI Level-2A  |  2023–2024 Median Composite",
    fontsize=12, fontweight="bold", color=PALETTE["dark"]
)
ax.set_ylim(0, cat_counts.max() * 1.18)
ax.tick_params(axis="x", labelsize=10)

plt.tight_layout()
plt.savefig("outputs/4_ndvi_category_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: outputs/4_ndvi_category_distribution.png")


# ============================================================
# 5. LST HOTSPOT TABLE (Top 20 priority neighborhoods)
# ============================================================
print("5/5 Generating LST hotspot table...")

hotspots = (
    df[df["LST_C"] >= LST_THRESHOLD]
    .sort_values("LST_C", ascending=False)
    [["name", "ilce_adi", "LST_C", "NDVI", "NDBI"]]
    .head(20)
    .reset_index(drop=True)
)
hotspots.index += 1
hotspots.columns = ["Neighborhood", "District", "LST (°C)", "NDVI", "NDBI"]
hotspots["LST (°C)"] = hotspots["LST (°C)"].round(2)
hotspots["NDVI"] = hotspots["NDVI"].round(3)
hotspots["NDBI"] = hotspots["NDBI"].round(3)

total_hotspots = len(df[df["LST_C"] >= LST_THRESHOLD])

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor(PALETTE["light"])
ax.axis("off")

table = ax.table(
    cellText=hotspots.values,
    colLabels=hotspots.columns,
    rowLabels=hotspots.index,
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 1.6)

# Style header
for j in range(len(hotspots.columns)):
    table[(0, j)].set_facecolor(PALETTE["dark"])
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# Alternating row colors
for i in range(1, len(hotspots) + 1):
    for j in range(len(hotspots.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#E8F4F8")

# Row label column
for i in range(1, len(hotspots) + 1):
    table[(i, -1)].set_facecolor(PALETTE["red"])
    table[(i, -1)].set_text_props(color="white", fontweight="bold")

ax.set_title(
    f"Top 20 Thermal Hotspot Neighborhoods — İzmir\n"
    f"LST ≥ {LST_THRESHOLD:.1f}°C  (city mean + 2°C threshold)  |  "
    f"Total hotspot neighborhoods: {total_hotspots} / {len(df)}",
    fontsize=12, fontweight="bold", color=PALETTE["dark"],
    pad=20
)

plt.tight_layout()
plt.savefig("outputs/5_lst_hotspot_table.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: outputs/5_lst_hotspot_table.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 50)
print(f"Total neighborhoods analyzed : {len(df)}")
print(f"City-wide mean LST           : {LST_MEAN:.2f}°C")
print(f"Hotspot threshold            : {LST_THRESHOLD:.2f}°C")
print(f"Total hotspot neighborhoods  : {total_hotspots} ({total_hotspots/len(df)*100:.1f}%)")
print(f"NDVI–LST correlation (r)     : {df['NDVI'].corr(df['LST_C']):.3f}")
print(f"NDBI–LST correlation (r)     : {df['NDBI'].corr(df['LST_C']):.3f}")
low_ndvi = len(df[df["NDVI"] < 0.2])
print(f"Low vegetation neighborhoods : {low_ndvi} ({low_ndvi/len(df)*100:.1f}%)")
print("=" * 50)
print("Outputs saved in: ./outputs/")