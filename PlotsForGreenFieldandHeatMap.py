"""
Task 4: Urban Heat Island Analysis and Green Space Optimization
Visualization Script v3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

os.makedirs("LST-NDVI-NDBI-Charts", exist_ok=True)

lst = pd.read_csv("izmir_mahalle_lst_final_latest_clean.csv")
s2  = pd.read_csv("izmir_mahalle_s2_final_latest_clean.csv")

df = pd.merge(
    s2[["name", "ilce_adi", "NDVI", "NDBI"]],
    lst[["name", "ilce_adi", "LST_C", "NDVI_L8", "NDBI_L8"]],
    on=["name", "ilce_adi"]
)

LST_MEAN      = df["LST_C"].mean()
LST_THRESHOLD = LST_MEAN + 2.0

print(f"Total neighborhoods: {len(df)}")
print(f"City-wide LST mean: {LST_MEAN:.2f}°C")
print(f"Hotspot threshold (mean + 2°C): {LST_THRESHOLD:.2f}°C")
print("-" * 50)

PALETTE = {
    "red":    "#E63946",
    "orange": "#F4A261",
    "yellow": "#E9C46A",
    "green":  "#2A9D8F",
    "dark":   "#264653",
    "gray":   "#ADB5BD",
}

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.right"]  = False


# ============================================================
# 1. CORRELATION MATRIX HEATMAP
# ============================================================
print("1/5 Generating correlation matrix...")

corr_cols   = ["NDVI", "NDBI", "LST_C"]
corr_labels = ["NDVI (Sentinel-2)", "NDBI (Sentinel-2)", "LST °C (Landsat 8)"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

sns.heatmap(
    corr_matrix, annot=True, fmt=".3f", cmap="RdYlGn_r",
    center=0, vmin=-1, vmax=1, linewidths=1.5, linecolor="white",
    annot_kws={"size": 15, "weight": "bold"}, ax=ax,
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
plt.savefig("LST-NDVI-NDBI-Charts/1_correlation_matrix.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("   Saved: 1_correlation_matrix.png")


# ============================================================
# 2. SCATTER PLOT: NDVI vs LST
# ============================================================
print("2/5 Generating NDVI vs LST scatter plot...")
 
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
 
sc = ax.scatter(df["NDVI"], df["LST_C"], c=df["NDBI"],
                cmap="RdYlGn_r", alpha=0.55, s=18, edgecolors="none")
 
z      = np.polyfit(df["NDVI"], df["LST_C"], 1)
p      = np.poly1d(z)
x_line = np.linspace(df["NDVI"].min(), df["NDVI"].max(), 200)
ax.plot(x_line, p(x_line), color=PALETTE["red"], lw=2,
        linestyle="--", label=f"Trend (slope={z[0]:.2f})")
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
 
r = df["NDVI"].corr(df["LST_C"])
ax.text(0.97, 0.97, f"Pearson r = {r:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
 
# --- Cluster annotation: Hottest (Çiğli + Bayındır) ---
from matplotlib.patches import Ellipse
hot_cluster = df[(df["LST_C"] > 39) & (df["NDVI"] < 0.25)]
hot_x = hot_cluster["NDVI"].mean()
hot_y = hot_cluster["LST_C"].mean()
hot_w = hot_cluster["NDVI"].std() * 4.5
hot_h = hot_cluster["LST_C"].std() * 4.5

ellipse_hot = Ellipse((hot_x, hot_y), width=hot_w, height=hot_h,
                       angle=0, edgecolor=PALETTE["red"], facecolor="none",
                       linewidth=1.8, linestyle="--", zorder=3)
ax.add_patch(ellipse_hot)
# Label placed to the LEFT of the red ellipse — shifted up
ax.text(hot_x - hot_w / 2 - 0.03, hot_y + 1.2,
        "Çiğli &\nBayındır",
        fontsize=9, color=PALETTE["red"], fontweight="bold",
        va="center", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

# --- Cluster annotation: Coolest (Kemalpaşa + Ödemiş) ---
cool_cluster = df[(df["LST_C"] < 29) & (df["NDVI"] > 0.55)]
cool_x = cool_cluster["NDVI"].mean()
cool_y = cool_cluster["LST_C"].mean()
cool_w = cool_cluster["NDVI"].std() * 4.5
cool_h = cool_cluster["LST_C"].std() * 4.5

ellipse_cool = Ellipse((cool_x, cool_y), width=cool_w, height=cool_h,
                        angle=0, edgecolor=PALETTE["green"], facecolor="none",
                        linewidth=1.8, linestyle="--", zorder=3)
ax.add_patch(ellipse_cool)
# Label placed to the BOTTOM-LEFT of the green ellipse — shifted up
ax.text(cool_x - cool_w / 2 - 0.03, cool_y - cool_h / 2 + 1.0,
        "Kemalpaşa &\nÖdemiş",
        fontsize=9, color=PALETTE["green"], fontweight="bold",
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

plt.tight_layout()
plt.savefig("LST-NDVI-NDBI-Charts/2_scatter_ndvi_vs_lst.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("   Saved: 2_scatter_ndvi_vs_lst.png")

 

# ============================================================
# 3. DISTRICT AVERAGE LST BAR CHART  (x starts at 30)
# ============================================================
print("3/5 Generating district average LST bar chart...")

district_avg = (
    df.groupby("ilce_adi")["LST_C"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
district_avg.columns = ["District", "Avg_LST"]

bar_colors = [
    PALETTE["red"] if v >= LST_THRESHOLD else PALETTE["green"]
    for v in district_avg["Avg_LST"]
]

# Narrower figure, x starts at 30
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.barh(
    district_avg["District"], district_avg["Avg_LST"],
    color=bar_colors, edgecolor="white", linewidth=0.5,
    left=0  # actual bar values; xlim set below
)

for bar, val in zip(bars, district_avg["Avg_LST"]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}°C", va="center", fontsize=8)

# X axis starts at 30
x_min = 30
x_max = district_avg["Avg_LST"].max() + 1.5
ax.set_xlim(x_min, x_max)

ax.axvline(LST_MEAN, color=PALETTE["dark"], lw=1.5, linestyle="--")
ax.axvline(LST_THRESHOLD, color=PALETTE["red"], lw=1.5, linestyle=":")

ax.set_xlabel("Average Land Surface Temperature (°C)", fontsize=10)
ax.set_title(
    "Average LST by District — İzmir\n"
    "Source: Landsat 8 Level-2  |  2023–2024 Median Composite",
    fontsize=11, fontweight="bold", color=PALETTE["dark"]
)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=8.5)

# Legend — placed outside chart area to avoid overlap
legend_handles = [
    mpatches.Patch(color=PALETTE["red"],   label="Thermal hotspot (≥ mean + 2°C)"),
    mpatches.Patch(color=PALETTE["green"], label="Below threshold"),
    plt.Line2D([0], [0], color=PALETTE["dark"], lw=1.5, linestyle="--",
               label=f"City mean ({LST_MEAN:.1f}°C)"),
    plt.Line2D([0], [0], color=PALETTE["red"], lw=1.5, linestyle=":",
               label=f"Hotspot threshold ({LST_THRESHOLD:.1f}°C)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
          bbox_to_anchor=(1.0, 0.0), framealpha=0.8)

plt.tight_layout()
plt.savefig("LST-NDVI-NDBI-Charts/3_district_avg_lst_bar.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("   Saved: 3_district_avg_lst_bar.png")


# ============================================================
# 4. NDVI CATEGORY DISTRIBUTION
# ============================================================
print("4/5 Generating NDVI category distribution...")

def ndvi_category(v):
    if v < 0.2:   return "Low (<0.2)\nInsufficient green cover"
    elif v < 0.4: return "Moderate (0.2–0.4)\nPartial green cover"
    else:          return "High (>0.4)\nHealthy vegetation"

df["NDVI_Category"] = df["NDVI"].apply(ndvi_category)

cat_order  = ["Low (<0.2)\nInsufficient green cover",
              "Moderate (0.2–0.4)\nPartial green cover",
              "High (>0.4)\nHealthy vegetation"]
cat_colors = [PALETTE["red"], PALETTE["yellow"], PALETTE["green"]]
cat_counts = df["NDVI_Category"].value_counts().reindex(cat_order)
cat_pct    = (cat_counts / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(cat_order, cat_counts, color=cat_colors,
              edgecolor="white", linewidth=1, width=0.55)
for bar, cnt, pct in zip(bars, cat_counts, cat_pct):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8, f"{cnt}\n({pct}%)",
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
plt.savefig("LST-NDVI-NDBI-Charts/4_ndvi_category_distribution.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("   Saved: 4_ndvi_category_distribution.png")


# ============================================================
# 5. TOP 10 HOTTEST + TOP 10 COOLEST TABLE
# ============================================================
print("5/5 Generating LST hotspot/coolspot table...")

top10 = (df.sort_values("LST_C", ascending=False)
           [["name", "ilce_adi", "LST_C", "NDVI", "NDBI"]]
           .head(10).reset_index(drop=True))
top10.index = range(1, 11)

bottom10 = (df.sort_values("LST_C", ascending=True)
              [["name", "ilce_adi", "LST_C", "NDVI", "NDBI"]]
              .head(10).reset_index(drop=True))
bottom10.index = range(1, 11)  # 1-10 for coolest too

combined_top    = top10.copy()
combined_bottom = bottom10.copy()

for df_part in [combined_top, combined_bottom]:
    df_part.columns = ["Neighborhood", "District", "LST (°C)", "NDVI", "NDBI"]
    df_part["LST (°C)"] = df_part["LST (°C)"].round(2)
    df_part["NDVI"]     = df_part["NDVI"].round(3)
    df_part["NDBI"]     = df_part["NDBI"].round(3)

total_hotspots = len(df[df["LST_C"] >= LST_THRESHOLD])

# Narrow column widths
col_widths = [0.36, 0.17, 0.13, 0.11, 0.11]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8))
fig.patch.set_facecolor("white")

for ax in [ax_top, ax_bot]:
    ax.axis("off")

# --- Top 10 hottest ---
tbl_top = ax_top.table(
    cellText=combined_top.values,
    colLabels=combined_top.columns,
    rowLabels=combined_top.index,
    cellLoc="center",
    loc="center",
    colWidths=col_widths
)
tbl_top.auto_set_font_size(False)
tbl_top.set_fontsize(9)
tbl_top.scale(1, 1.35)

for j in range(len(combined_top.columns)):
    tbl_top[(0, j)].set_facecolor(PALETTE["red"])
    tbl_top[(0, j)].set_text_props(color="white", fontweight="bold")

for i in range(1, 11):
    tbl_top[(i, -1)].set_facecolor(PALETTE["red"])
    tbl_top[(i, -1)].set_text_props(color="white", fontweight="bold")
    for j in range(len(combined_top.columns)):
        if i % 2 == 0:
            tbl_top[(i, j)].set_facecolor("#FFE8E8")

ax_top.set_title("Top 10 Hottest Neighborhoods",
                 fontsize=10, fontweight="bold", color=PALETTE["red"],
                 pad=6, loc="center")

# --- Top 10 coolest ---
tbl_bot = ax_bot.table(
    cellText=combined_bottom.values,
    colLabels=combined_bottom.columns,
    rowLabels=combined_bottom.index,
    cellLoc="center",
    loc="center",
    colWidths=col_widths
)
tbl_bot.auto_set_font_size(False)
tbl_bot.set_fontsize(9)
tbl_bot.scale(1, 1.35)

for j in range(len(combined_bottom.columns)):
    tbl_bot[(0, j)].set_facecolor(PALETTE["green"])
    tbl_bot[(0, j)].set_text_props(color="white", fontweight="bold")

for i in range(1, 11):
    tbl_bot[(i, -1)].set_facecolor(PALETTE["green"])
    tbl_bot[(i, -1)].set_text_props(color="white", fontweight="bold")
    for j in range(len(combined_bottom.columns)):
        if i % 2 == 0:
            tbl_bot[(i, j)].set_facecolor("#E8F4F0")

ax_bot.set_title("Top 10 Coolest Neighborhoods",
                 fontsize=10, fontweight="bold", color=PALETTE["green"],
                 pad=6, loc="center")

fig.suptitle(
    f"Hottest vs Coolest Neighborhoods — İzmir\n"
    f"Ranked by Land Surface Temperature  |  "
    f"Hotspot threshold: {LST_THRESHOLD:.1f}°C  |  "
    f"Total hotspot neighborhoods: {total_hotspots} / {len(df)}",
    fontsize=11, fontweight="bold", color=PALETTE["dark"], y=1.01
)

plt.tight_layout()
plt.savefig("LST-NDVI-NDBI-Charts/5_lst_hotspot_coolspot_table.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("   Saved: 5_lst_hotspot_coolspot_table.png")


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
print("Outputs saved in: ./LST-NDVI-NDBI-Charts/")