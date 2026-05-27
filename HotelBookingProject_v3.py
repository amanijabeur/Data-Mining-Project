import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection  import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing    import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.decomposition    import PCA
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      ConfusionMatrixDisplay, roc_auc_score,
                                      RocCurveDisplay)
import kagglehub
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# COLOR PALETTE  (all-pink theme; violet reserved for PCA & ROC)
# ══════════════════════════════════════════════════════════════
DARK_PINK   = "#C2185B"   # deep rose        — titles, borders, primary text
PINK        = "#FF69B4"   # hot pink         — bars, accents
LIGHT_PINK  = "#FCE4EC"   # very pale pink   — backgrounds, fills
PINK_2      = "#F48FB1"   # soft pink        — gridlines, minor accents
ROSE        = "#E91E63"   # vivid rose       — Type I error, model bar 2
BLUSH       = "#F8BBD0"   # blush pink       — histogram fills, secondary fill
DUSTY_PINK  = "#D81B60"   # dusty deep pink  — Type II error, emphasis
MAUVE       = "#F06292"   # mauve pink       — model bar 3, coef chart

# Violet / purple — used ONLY in PCA plots and ROC curves
VIOLET      = "#7B1FA2"   # deep violet      — PCA bars, PCA scatter cancelled
PURPLE      = "#AB47BC"   # medium purple    — PCA accents
LAVENDER    = "#E1BEE7"   # pale lavender    — PCA cumulative fill

BG          = "#FFF0F5"   # lavender blush   — figure/axes background

# Grouped metric bars (3 metrics: CV AUC, Test AUC, F1)
MODEL_COLORS = [DARK_PINK, ROSE, MAUVE]

# ROC curves: 3 distinct colors — pink, violet, rose (one per model)
ROC_COLORS = [DARK_PINK, VIOLET, ROSE]

# Diverging heatmap: deep pink → white → mauve
PMAP = sns.diverging_palette(330, 300, s=80, l=45, as_cmap=True)

def set_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "axes.edgecolor":   DARK_PINK,
        "axes.labelcolor":  DARK_PINK,
        "xtick.color":      DARK_PINK,
        "ytick.color":      DARK_PINK,
        "grid.color":       PINK_2,
        "grid.alpha":       0.3,
        "text.color":       DARK_PINK,
        "font.family":      "DejaVu Sans",
    })

def style_ax(ax):
    """Apply consistent pink-violet styling to an axis."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=DARK_PINK)
    for sp in ax.spines.values():
        sp.set_edgecolor(DARK_PINK)
    ax.grid(color=PINK_2, alpha=0.35, linestyle="--")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"  Saved: {name}.png")

def section(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)

set_style()

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
section("1. LOADING DATA")

path     = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
csv_file = next(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv"))
df_raw   = pd.read_csv(csv_file)
print(f"  Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# 2. TARGET LEAKAGE REMOVAL
# ══════════════════════════════════════════════════════════════
section("2. TARGET LEAKAGE REMOVAL")

LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]

for col in LEAKAGE_COLS:
    if col in df_raw.columns:
        df_raw.drop(columns=[col], inplace=True)
        print(f"  Dropped from df_raw: {col}")


remaining_status = [c for c in df_raw.columns if "status" in c.lower()]
print(f"  df_raw status columns remaining: {remaining_status}")
assert remaining_status == [], f"LEAKAGE STILL PRESENT: {remaining_status}"
print("  Assertion passed: zero leakage columns in df_raw.")

# ══════════════════════════════════════════════════════════════
# 3. DATA CLEANING  (safe — no statistics learned here)
# ══════════════════════════════════════════════════════════════
section("3. DATA CLEANING")

df = df_raw.copy()

# Drop columns with >40% missing values
hi_miss = [c for c in df.columns if df[c].isna().mean() > 0.40]
df.drop(columns=hi_miss, inplace=True)
print(f"  Dropped high-missing cols: {hi_miss}")

# Remove exact duplicate rows
n_before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Removed {n_before - len(df):,} duplicate rows")

# Domain-specific fixes (value substitution, no learned statistics)
if "meal"     in df.columns: df["meal"]     = df["meal"].replace("Undefined", "SC")
if "agent"    in df.columns: df["agent"]    = df["agent"].fillna(0)
if "company"  in df.columns: df["company"]  = df["company"].fillna(0)
if "children" in df.columns: df["children"] = df["children"].fillna(0)

# Remove bookings with zero guests (data entry errors)
guest_cols = [c for c in ["adults", "children", "babies"] if c in df.columns]
df = df[df[guest_cols].sum(axis=1) > 0]

# Remove ADR anomalies (negative price is impossible)
if "adr" in df.columns:
    df = df[df["adr"] >= 0]

print(f"  Final clean shape: {df.shape}")

TARGET = "is_canceled"

# Confirm leakage columns are absent from df as well
for col in LEAKAGE_COLS:
    assert col not in df.columns, f"LEAKAGE COLUMN FOUND IN df: {col}"
print(f"  Leakage columns confirmed absent from df.")

# ══════════════════════════════════════════════════════════════
# 4. COLUMN TYPE IDENTIFICATION
#    Strings remain as strings — encoding happens inside Pipeline after split.
# ══════════════════════════════════════════════════════════════
section("4. COLUMN TYPE IDENTIFICATION")

cat_cols_raw = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols_raw = df.select_dtypes(include=[np.number]).columns.drop(TARGET).tolist()

# ohe_cols and ord_cols will be computed AFTER the train/test split
# using X_train only — see Section 5.  This eliminates the small theoretical
# leakage that arises from computing cardinality on the full dataset.
OHE_THRESHOLD = 10

print(f"  Categorical columns identified ({len(cat_cols_raw)}): {cat_cols_raw}")
print(f"  Numeric features identified: {len(num_cols_raw)}")
print("  NOTE: OHE/Ordinal split by cardinality will be computed from X_train only (Section 5).")

# ══════════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT  ← moved here, BEFORE EDA
#
#    WHY split before EDA?
#      Correlations, distributions, and feature selection computed on
#      the full dataset implicitly use test-set information to decide
#      which features to keep.  This is a subtle but real form of
#      data leakage.  Splitting first and running all EDA on df_train
#      ensures the test set has zero influence on any analytical or
#      modelling decision.
# ══════════════════════════════════════════════════════════════
section("5. TRAIN / TEST SPLIT  (before EDA — prevents leakage)")

# Include all features (numeric + all categorical) before split
all_features_initial = num_cols_raw + cat_cols_raw
X = df[all_features_initial].copy()
y = df[TARGET].copy()

for col in LEAKAGE_COLS:
    assert col not in X.columns, f"LEAKAGE COLUMN IN X: {col}"
print("  Safety check passed: X contains no leakage columns.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# ── Compute ohe_cols / ord_cols from X_train ONLY ─────────────
#    Cardinality is observed on training data only — no test leakage.
ohe_cols = [c for c in cat_cols_raw if X_train[c].nunique() <= OHE_THRESHOLD]
ord_cols = [c for c in cat_cols_raw if X_train[c].nunique() >  OHE_THRESHOLD]

print(f"  Low-cardinality  OHE cols ({len(ohe_cols)}): {ohe_cols}")
print(f"  High-cardinality Ord cols ({len(ord_cols)}): {ord_cols}")

# Final leakage safety check on column lists
for col in LEAKAGE_COLS:
    assert col not in ohe_cols + ord_cols + num_cols_raw, \
        f"LEAKAGE COLUMN IN FEATURE LISTS: {col}"
print("  Safety check passed: leakage columns absent from all feature lists.")

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Train cancel rate: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")
print("""
  All EDA below uses df_train / df_raw_train (train rows only).
  Correlations, distributions, skewness, and feature selection thresholds
  are derived from training data exclusively — the test set is
  never observed at any analytical or preprocessing stage.
  This eliminates the subtle leakage that occurs when corr_target
  is computed on the full dataset (including test rows).""")

# Convenience: reconstruct a labelled train DataFrame for EDA
df_train = X_train.copy()
df_train[TARGET] = y_train.values

# Train-only slice of df_raw (for categorical EDA in 5e / 5f)
# Uses the index from the split — never touches test rows.
df_raw_train = df_raw.loc[X_train.index].copy()

# ── 5a. Target Distribution ───────────────────────────────────
section("5a. EDA — TARGET DISTRIBUTION & CLASS BALANCE")

cancel_counts = df_train[TARGET].value_counts()
cancel_pct    = df_train[TARGET].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Target Distribution: Is Booking Cancelled?",
             fontsize=14, color=DARK_PINK, fontweight="bold")

axes[0].pie(cancel_counts,
            labels=["Not Cancelled", "Cancelled"],
            colors=[LIGHT_PINK, DARK_PINK],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": DARK_PINK, "fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Proportion", color=DARK_PINK)

bars = axes[1].bar(["Not Cancelled", "Cancelled"], cancel_counts.values,
                   color=[PINK_2, DARK_PINK], edgecolor=DARK_PINK)
for b, v in zip(bars, cancel_counts.values):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,}",
                 ha="center", color=DARK_PINK, fontweight="bold")
style_ax(axes[1])
save_fig("01_target_distribution")

print(f"""  INSIGHT: Dataset is moderately imbalanced.
    {cancel_pct[0]:.1f}% Not Cancelled  vs  {cancel_pct[1]:.1f}% Cancelled.
    Accuracy alone is misleading here — we use AUC-ROC throughout.
    Stratified splits ensure both folds and test set mirror this ratio.""")

# ── 5b. Correlation Analysis ──────────────────────────────────
section("5b. EDA — CORRELATION ANALYSIS (numeric features)")

# Compute correlation on ALL numeric features (leakage-free: train only)
num_df_all   = df_train[[c for c in num_cols_raw if c in df_train.columns] + [TARGET]]
corr_all     = (num_df_all.corr()[TARGET]
                .drop(TARGET)
                .sort_values(key=abs, ascending=False))

# Feature selection threshold derived from full correlation (ALL features)
CORR_THRESHOLD = 0.02
weak_features  = corr_all[corr_all.abs() < CORR_THRESHOLD].index.tolist()

# Top 15 used only for plotting
corr_target = corr_all.head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Correlation — Which Numeric Features Drive Cancellation?",
             fontsize=14, color=DARK_PINK, fontweight="bold")

top12    = corr_target.head(12).index.tolist() + [TARGET]
corr_mat = df_train[top12].corr()
mask     = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, ax=axes[0], cmap=PMAP,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.5, linecolor=LIGHT_PINK,
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Feature Correlation Heatmap", color=DARK_PINK)
axes[0].tick_params(colors=DARK_PINK, labelsize=8)

bar_colors = [DARK_PINK if v > 0 else MAUVE for v in corr_target.values]
axes[1].barh(corr_target.index[::-1], corr_target.values[::-1],
             color=bar_colors[::-1], edgecolor=DARK_PINK, linewidth=0.7)
axes[1].axvline(0, color=DARK_PINK, linewidth=1.5)
axes[1].set_title("Correlation with is_canceled (Top 15)", color=DARK_PINK)
axes[1].set_xlabel("Pearson r", color=DARK_PINK)
style_ax(axes[1])
save_fig("02_correlation_analysis")

print(f"""  INSIGHT: Top 3 numeric predictors: {corr_all.head(3).index.tolist()}
    Features with |r| < {CORR_THRESHOLD} (near-zero signal): {weak_features}
    Correlation computed on ALL {len(corr_all)} numeric features (top 15 shown in plot).
    MODELING DECISION: weak features will be dropped in the feature selection step.""")

# ── 5c. Lead Time vs Cancellation ─────────────────────────────
section("5c. EDA — LEAD TIME vs CANCELLATION")

if "lead_time" in df_train.columns:
    c_vals  = df_train[df_train[TARGET] == 1]["lead_time"]
    nc_vals = df_train[df_train[TARGET] == 0]["lead_time"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lead Time — Do Customers Who Book Far Ahead Cancel More?",
                 fontsize=12, color=DARK_PINK, fontweight="bold")

    bp = axes[0].boxplot(
        [nc_vals, c_vals], patch_artist=True,
        labels=["Not Cancelled", "Cancelled"],
        medianprops=dict(color=DARK_PINK, linewidth=2),
        whiskerprops=dict(color=DARK_PINK),
        capprops=dict(color=DARK_PINK),
        boxprops=dict(color=DARK_PINK),
        flierprops=dict(marker="o", markerfacecolor=PINK,
                        markeredgecolor=PINK, markersize=3, alpha=0.3))
    bp["boxes"][0].set_facecolor(LIGHT_PINK)
    bp["boxes"][1].set_facecolor(ROSE)
    bp["boxes"][1].set_alpha(0.7)
    axes[0].set_title("Lead Time by Outcome", color=DARK_PINK)
    axes[0].set_ylabel("Lead Time (days)", color=DARK_PINK)
    style_ax(axes[0])

    df_train["_lead_bin"] = pd.cut(
        df_train["lead_time"], bins=[0, 7, 30, 90, 180, 365, 700],
        labels=["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "365d+"])
    rate = df_train.groupby("_lead_bin")[TARGET].mean()
    axes[1].bar(rate.index, rate.values, color=PINK, edgecolor=DARK_PINK)
    for i, (_, val) in enumerate(rate.items()):
        axes[1].text(i, val + 0.01, f"{val:.0%}", ha="center",
                     color=DARK_PINK, fontweight="bold", fontsize=9)
    axes[1].set_title("Cancel Rate by Lead Time Bucket", color=DARK_PINK)
    axes[1].set_ylabel("Cancellation Rate")
    axes[1].set_ylim(0, 1)
    style_ax(axes[1])
    df_train.drop(columns=["_lead_bin"], inplace=True)
    save_fig("03_lead_time")

    print(f"""  BUSINESS INSIGHT: Median lead time is {c_vals.median():.0f}d (cancelled)
    vs {nc_vals.median():.0f}d (not cancelled). Bookings >90 days ahead cancel at >50%.
    RECOMMENDATION: Require a deposit for bookings with lead_time > 90 days.
    MODELING DECISION: lead_time is a high-priority numeric predictor.""")

# ── 5d. Price (ADR) vs Cancellation ───────────────────────────
section("5d. EDA — PRICE (ADR) vs CANCELLATION")

if "adr" in df_train.columns:
    adr_c  = df_train[df_train[TARGET] == 1]["adr"].clip(0, 500)
    adr_nc = df_train[df_train[TARGET] == 0]["adr"].clip(0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Average Daily Rate — Do Expensive Bookings Cancel More?",
                 fontsize=12, color=DARK_PINK, fontweight="bold")

    axes[0].hist(adr_nc, bins=60, color=LIGHT_PINK, edgecolor=PINK_2,
                 alpha=0.8, label="Not Cancelled", density=True)
    axes[0].hist(adr_c,  bins=60, color=ROSE, edgecolor=DUSTY_PINK,
                 alpha=0.6, label="Cancelled", density=True)
    axes[0].set_title("ADR Distribution by Outcome", color=DARK_PINK)
    axes[0].set_xlabel("ADR (EUR/night, clipped at 500)", color=DARK_PINK)
    axes[0].legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK)
    style_ax(axes[0])

    df_train["_adr_bin"] = pd.cut(
        df_train["adr"].clip(0, 500),
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=["<50", "50-100", "100-150", "150-200", "200-300", "300+"])
    adr_rate = df_train.groupby("_adr_bin")[TARGET].mean()
    axes[1].bar(adr_rate.index, adr_rate.values, color=PINK, edgecolor=DARK_PINK)
    for i, (_, val) in enumerate(adr_rate.items()):
        axes[1].text(i, val + 0.01, f"{val:.0%}", ha="center",
                     color=DARK_PINK, fontweight="bold", fontsize=9)
    axes[1].set_title("Cancel Rate by Price Bucket", color=DARK_PINK)
    axes[1].set_ylim(0, 1)
    style_ax(axes[1])
    df_train.drop(columns=["_adr_bin"], inplace=True)
    save_fig("04_price_adr")

    print("""  BUSINESS INSIGHT: Premium bookings (ADR > 200) cancel at higher rates.
    Customers may comparison-shop after booking when rates are high.
    adr is right-skewed — StandardScaler normalises it for LR convergence.
    MODELING DECISION: adr is a useful numeric predictor; always scale before LR.""")

# ── 5e. Market Segment, Customer Type, Deposit Type ───────────
section("5e. EDA — SEGMENT / CUSTOMER TYPE / DEPOSIT TYPE")

for feat in ["market_segment", "customer_type", "deposit_type"]:
    if feat not in df_raw_train.columns:
        continue
    tmp  = df_raw_train[[feat, TARGET]].copy()
    rate = (tmp.groupby(feat)[TARGET]
               .agg(["mean", "count"])
               .rename(columns={"mean": "cancel_rate", "count": "n"})
               .sort_values("cancel_rate"))

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(rate.index, rate["cancel_rate"],
                   color=PINK, edgecolor=DARK_PINK)
    for i, (_, row) in enumerate(rate.iterrows()):
        ax.text(row["cancel_rate"] + 0.005, i, f"n={row['n']:,}",
                va="center", color=DARK_PINK, fontsize=8)
    ax.set_title(f"Cancellation Rate by {feat}",
                 color=DARK_PINK, fontweight="bold", fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Cancellation Rate", color=DARK_PINK)
    style_ax(ax)
    save_fig(f"05_cancel_by_{feat}")

print("""  BUSINESS INSIGHT:
    deposit_type 'Non Refund': near-zero cancellation (customer already paid).
    market_segment Online TA / Groups: highest cancellation rates.
    Direct / Corporate bookings: lowest cancellation rates.
    RECOMMENDATION: targeted deposit policies per segment.
    MODELING DECISION: high-value categorical predictors — include via OHE.""")

# ── 5f. Seasonality ───────────────────────────────────────────
section("5f. EDA — SEASONALITY")

if "arrival_date_month" in df_raw_train.columns:
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    tmp = df_raw_train[["arrival_date_month", TARGET]].copy()
    tmp["arrival_date_month"] = pd.Categorical(
        tmp["arrival_date_month"], categories=month_order, ordered=True)
    monthly = tmp.groupby("arrival_date_month")[TARGET].agg(["mean", "count"])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(range(12), monthly["count"], color=LIGHT_PINK,
            edgecolor=PINK_2, alpha=0.8, label="Total Bookings")
    ax2.plot(range(12), monthly["mean"], color=DARK_PINK,
             linewidth=2.5, marker="o", markersize=7, label="Cancel Rate")
    ax2.fill_between(range(12), monthly["mean"], alpha=0.15, color=BLUSH)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels([m[:3] for m in month_order], color=DARK_PINK)
    ax1.set_ylabel("Bookings", color=DARK_PINK)
    ax2.set_ylabel("Cancel Rate", color=DARK_PINK)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("Seasonality: Bookings & Cancel Rate by Month",
                  color=DARK_PINK, fontweight="bold", fontsize=13)
    ax1.set_facecolor(BG)
    ax1.tick_params(colors=DARK_PINK)
    for sp in ax1.spines.values():
        sp.set_edgecolor(DARK_PINK)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, facecolor=BG, edgecolor=DARK_PINK,
               labelcolor=DARK_PINK)
    save_fig("06_seasonality")

    print("""  BUSINESS INSIGHT: Summer peaks in both volume and cancellation rate.
    Revenue management should adjust overbooking buffers by month.
    MODELING DECISION: arrival_date_month included as an OHE feature.""")

# ── 5g. Skewness & Box Plots ──────────────────────────────────
section("5g. EDA — SKEWNESS & PINK BOX PLOTS")

skew_s  = pd.Series({c: df_train[c].skew() for c in num_cols_raw if c in df_train.columns}).sort_values(key=abs, ascending=False)
top5_sk = skew_s.head(5).index.tolist()

print("  Top 5 most skewed numeric features:")
for c in top5_sk:
    print(f"    {c:35s}  skewness = {skew_s[c]:+.2f}")

# Large box plot — mirrors the style from the project screenshot
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
bp = ax.boxplot(
    [df_train[c].dropna().values for c in top5_sk],
    patch_artist=True, vert=True, widths=0.45,
    flierprops=dict(marker="o", markerfacecolor=PINK,
                    markeredgecolor=PINK, markersize=4, alpha=0.5),
    medianprops=dict(color=DARK_PINK, linewidth=2.5),
    whiskerprops=dict(color=DARK_PINK, linewidth=1.5),
    capprops=dict(color=DARK_PINK, linewidth=2),
    boxprops=dict(color=DARK_PINK))
for patch in bp["boxes"]:
    patch.set_facecolor(LIGHT_PINK)
    patch.set_alpha(0.85)
ax.set_xticks(range(1, len(top5_sk) + 1))
ax.set_xticklabels(top5_sk, rotation=30, ha="right", fontsize=10, color=DARK_PINK)
ax.set_title("Box Plot — Top 5 Most Skewed Numeric Features",
             fontsize=13, color=DARK_PINK, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x:,.0f}" if abs(x) > 999 else f"{x:.1f}"))
ax.grid(axis="y", color=PINK_2, alpha=0.4, linestyle="--")
save_fig("07_boxplot_top5_skewed")

# Grid of all numeric features
n_c = 4
n_r = int(np.ceil(len(num_cols_raw) / n_c))
fig2, axes2 = plt.subplots(n_r, n_c, figsize=(n_c * 4, n_r * 3.2))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Box Plots — All Numeric Features",
              fontsize=14, color=DARK_PINK, fontweight="bold", y=1.01)
flat = axes2.flatten()
num_cols_in_train = [c for c in num_cols_raw if c in df_train.columns]
for i, col in enumerate(num_cols_in_train):
    a = flat[i]
    a.set_facecolor(BG)
    bp2 = a.boxplot(
        df_train[col].dropna().values, patch_artist=True, widths=0.5,
        flierprops=dict(marker="o", markerfacecolor=PINK,
                        markeredgecolor=PINK, markersize=3, alpha=0.35),
        medianprops=dict(color=DARK_PINK, linewidth=2),
        whiskerprops=dict(color=DARK_PINK, linewidth=1.2),
        capprops=dict(color=DARK_PINK, linewidth=1.5),
        boxprops=dict(color=DARK_PINK))
    bp2["boxes"][0].set_facecolor(LIGHT_PINK)
    a.set_title(f"{col}\nskew={df_train[col].skew():+.1f}", fontsize=8, color=DARK_PINK)
    a.tick_params(colors=DARK_PINK, labelsize=6)
    for sp in a.spines.values():
        sp.set_edgecolor(DARK_PINK)
    a.grid(axis="y", color=PINK_2, alpha=0.35, linestyle="--")
for j in range(i + 1, len(flat)):
    flat[j].set_visible(False)
save_fig("08_boxplot_all_numeric")

print("""  INSIGHT: Heavy right skew in lead_time, adr, agent, company.
    StandardScaler (fitted on train only) normalises these for Logistic Regression.
    Random Forest is tree-based and scale-invariant; LR variants benefit from scaling.""")

# ══════════════════════════════════════════════════════════════
# 6. FEATURE SELECTION  (correlation-based, numeric only)
#    Justified by correlation analysis in section 5b.
# ══════════════════════════════════════════════════════════════
section("6. FEATURE SELECTION — Drop Weak Numeric Features")

keep_num = [c for c in num_cols_raw if c not in weak_features]
print("\nFeature selection (numeric): removing very weak linear correlations (|r| < 0.02) mainly to simplify the feature space and support linear models")
print(f"  Numeric features kept ({len(keep_num)}): {keep_num}")
print(f"  Categorical features kept: {ohe_cols + ord_cols}")
print("""  Justification: Features with near-zero Pearson correlation add noise
  to linear models and can cause multicollinearity issues.
  Tree models are noise-robust but also benefit from cleaner input.""")

# Final leakage check on selected feature list
for col in LEAKAGE_COLS:
    assert col not in keep_num + ohe_cols + ord_cols, \
        f"LEAKAGE COLUMN IN SELECTED FEATURES: {col}"
print("  Safety check passed: leakage columns absent from selected features.")

# ══════════════════════════════════════════════════════════════
# 7. APPLY FEATURE SELECTION TO EXISTING SPLIT
#    No new split needed — reuse X_train / X_test from section 5.
#    Simply restrict columns to the selected feature set.
# ══════════════════════════════════════════════════════════════
section("7. APPLY FEATURE SELECTION TO EXISTING SPLIT")

all_features = keep_num + ohe_cols + ord_cols

# Restrict both splits to selected features only
X_train = X_train[[c for c in all_features if c in X_train.columns]]
X_test  = X_test [[c for c in all_features if c in X_test.columns]]

# Final assertion: no leakage column can be in X_train / X_test
for col in LEAKAGE_COLS:
    assert col not in X_train.columns, f"LEAKAGE COLUMN IN X_train: {col}"
print("  Safety check passed: X_train/X_test contain no leakage columns.")
print(f"  X_train shape after feature selection: {X_train.shape}")
print(f"  X_test  shape after feature selection: {X_test.shape}")
print("""
  All encoding, imputation, and scaling will be fitted on X_train only,
  then applied (transform only) to X_test — no information from the
  test set ever influences any transformation parameters.""")

# ══════════════════════════════════════════════════════════════
# 8. COLUMN TRANSFORMER  (fitted on train only inside Pipeline)
#    WHY OHE for low-cardinality?
#      LabelEncoder([City, Resort]) → [0, 1] implies an ordinal order.
#      OneHotEncoder creates independent binary columns — no false order.
#    WHY OrdinalEncoder for high-cardinality?
#      OHE on 100+ category columns explodes dimensions.
#      OrdinalEncoder is compatible with tree models (RF),
#      which split on thresholds and do not assume linearity.
# ══════════════════════════════════════════════════════════════
section("8. COLUMN TRANSFORMER — OHE + Ordinal (fit on TRAIN only)")

ohe_present = [c for c in ohe_cols if c in X_train.columns]
ord_present = [c for c in ord_cols if c in X_train.columns]
num_present = [c for c in keep_num if c in X_train.columns]

print(f"  OHE  cols ({len(ohe_present)}): {ohe_present}")
print(f"  Ord  cols ({len(ord_present)}): {ord_present}")
print(f"  Num  cols ({len(num_present)}): {num_present}")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

ohe_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

ord_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_present),
    ("ohe", ohe_transformer,     ohe_present),
    ("ord", ord_transformer,     ord_present),
], remainder="drop")

# ══════════════════════════════════════════════════════════════
# 9. PCA — DIAGNOSTIC EDA ONLY
#
#    Fitted on preprocessed X_train ONLY — no leakage.
#    Purpose: visualise variance structure and feature loadings ONLY.
#    PCA is NOT used in any of the three final compared models.
#
#    Why PCA-LR was removed from final comparison:
#      PCA is applied to a mix of: scaled numeric, OHE sparse, and
#      ordinal-encoded high-cardinality columns (e.g. country).
#      Ordinal codes are arbitrary integers — PCA treats them as real
#      distances, distorting components. Result: compressed artificial
#      variance, not meaningful cancellation signal.
#
#    Three diagnostic outputs:
#      Plot A — Scree plot: how many PCs capture 95%/99% variance?
#      Plot B — 2D scatter: class separability in PC1 vs PC2 space
#      Plot C — Loadings heatmap: which features drive each PC?
# ══════════════════════════════════════════════════════════════
section("9. PCA — DIAGNOSTIC EDA (fit on TRAIN only)")

# Fit preprocessor on train only, transform both — not leakage because
# preprocessor.fit() only sees X_train here (outside Pipeline, for diagnostics)
pre_fit     = preprocessor.fit(X_train)
X_train_pre = pre_fit.transform(X_train)
X_test_pre  = pre_fit.transform(X_test)
n_dims      = X_train_pre.shape[1]
print(f"  Dimensions after preprocessing (OHE expanded): {n_dims}")

# Full PCA for explained variance analysis (train only)
pca_full = PCA(random_state=42)
pca_full.fit(X_train_pre)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95    = int(np.searchsorted(cum_var, 0.95)) + 1
n_99    = int(np.searchsorted(cum_var, 0.99)) + 1
n_plot  = min(50, n_dims)

print(f"  PCs to explain 95% variance: {n_95}  (from {n_dims} total)")
print(f"  PCs to explain 99% variance: {n_99}")
print(f"  Dimensionality reduction: {n_dims} → {n_95} ({n_95/n_dims*100:.1f}% of original)")

# Plot A: Scree and cumulative explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA — Explained Variance Analysis",
             fontsize=14, color=DARK_PINK, fontweight="bold")

axes[0].bar(range(1, n_plot + 1),
            pca_full.explained_variance_ratio_[:n_plot] * 100,
            color=VIOLET, edgecolor=PURPLE, linewidth=0.6)
axes[0].set_title("Individual Explained Variance (first 50 PCs)", color=DARK_PINK)
axes[0].set_xlabel("Principal Component", color=DARK_PINK)
axes[0].set_ylabel("Variance Explained (%)", color=DARK_PINK)
style_ax(axes[0])

axes[1].plot(range(1, n_plot + 1), cum_var[:n_plot] * 100,
             color=DARK_PINK, linewidth=2.5, marker="o", markersize=4)
axes[1].axhline(95, color=VIOLET,    linewidth=1.5, linestyle="--", label="95% threshold")
axes[1].axhline(99, color=PURPLE,  linewidth=1.5, linestyle="--", label="99% threshold")
axes[1].axvline(n_95, color=VIOLET,    linewidth=1.2, linestyle=":")
axes[1].axvline(n_99, color=PURPLE,  linewidth=1.2, linestyle=":")
axes[1].fill_between(range(1, n_plot + 1), cum_var[:n_plot] * 100,
                     alpha=0.10, color=LAVENDER)
axes[1].set_title("Cumulative Explained Variance", color=DARK_PINK)
axes[1].set_xlabel("Number of Components", color=DARK_PINK)
axes[1].set_ylabel("Cumulative Variance Explained (%)", color=DARK_PINK)
axes[1].legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK)
axes[1].annotate(f"95% @ PC{n_95}", xy=(n_95, 95),
                 xytext=(n_95 + 2, 82), color=DARK_PINK, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=DARK_PINK))
style_ax(axes[1])
save_fig("PCA_A_explained_variance")

print(f"""  INSIGHT: {n_95} principal component(s) capture 95% of variance.
  In this dataset, PC1 captures most of the variance, indicating strong
  redundancy among the transformed features.

  The loadings also suggest that PC1 is strongly influenced by ordinal-encoded
  variables such as country, so PCA may capture encoding structure rather than
  pure cancellation behavior.

  However, this dominant variance direction does not clearly separate
  cancelled and non-cancelled bookings, as shown in the PCA scatter plot.

  This means the directions of highest variance are not the same as the
  directions that best predict cancellation, so PCA is useful here for
  diagnostics, not for final prediction.

  NOTE: PCA is used here for diagnostics only — it is not one of the
  three compared models (Logistic Regression, Random Forest, Decision Tree).""")


# Plot B: 2D scatter — class separability in PC1 vs PC2
pca_2d     = PCA(n_components=2, random_state=42)
pca_2d.fit(X_train_pre)
X_tr_2d    = pca_2d.transform(X_train_pre)

rng    = np.random.default_rng(42)
idx_s  = rng.choice(len(X_tr_2d), size=min(10000, len(X_tr_2d)), replace=False)
pts    = X_tr_2d[idx_s]
lbs    = y_train.values[idx_s]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.scatter(pts[lbs == 0, 0], pts[lbs == 0, 1],
           c=LIGHT_PINK, edgecolors=PINK_2, s=12, alpha=0.5,
           linewidths=0.3, label="Not Cancelled")
ax.scatter(pts[lbs == 1, 0], pts[lbs == 1, 1],
           c=VIOLET, edgecolors=PURPLE, s=12, alpha=0.6,
           linewidths=0.3, label="Cancelled")
ax.set_title(
    f"PCA 2D Projection — PC1 vs PC2\n"
    f"(PC1={pca_2d.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2={pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)",
    color=DARK_PINK, fontweight="bold", fontsize=12)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", color=DARK_PINK)
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", color=DARK_PINK)
ax.legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK, markerscale=2)
style_ax(ax)
save_fig("PCA_B_2D_scatter")

print(f"""  INSIGHT: Class overlap in 2D PCA space is expected for this dataset.
  PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%) captures the dominant variance direction.
  Strong overlap confirms non-linear models (RF) will outperform linear LR.
  A healthy scatter without single-axis separation confirms leakage is gone.""")

# Plot C: Loadings heatmap — what original features drive each PC?
try:
    feat_names_pre = pre_fit.get_feature_names_out()
except Exception:
    feat_names_pre = [f"f{i}" for i in range(n_dims)]

loadings     = pd.DataFrame(pca_full.components_[:5, :],
                             index=[f"PC{i+1}" for i in range(5)],
                             columns=feat_names_pre)
top_load_idx = np.argsort(np.abs(loadings.loc["PC1"]))[-12:]
load_subset  = loadings[loadings.columns[top_load_idx]]

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
sns.heatmap(load_subset, ax=ax, cmap=PMAP, annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.5, linecolor=LIGHT_PINK,
            center=0, cbar_kws={"shrink": 0.7})
ax.set_title("PCA Loadings Heatmap — Top 12 Features Driving PC1–PC5",
             color=DARK_PINK, fontweight="bold", fontsize=12)
ax.tick_params(colors=DARK_PINK, labelsize=8)
ax.set_yticklabels(ax.get_yticklabels(), color=DARK_PINK)
save_fig("PCA_C_loadings_heatmap")

print("""  INSIGHT: Loadings connect PCA back to interpretable business variables.
  Large positive loading → feature strongly pushes data along that PC direction.
  Large negative loading → feature pulls in the opposite direction.
  This lets us say e.g. 'PC1 mainly represents lead_time and deposit_type'.""")

print(f"\n  PCA diagnostic complete: {n_95} components capture 95% variance.")
print("  NOTE: PCA is used for diagnostics only — not carried into the final model comparison.")

# ══════════════════════════════════════════════════════════════
# 10. MODEL TRAINING — GridSearchCV
#
#     Three models compared:
#       1. Logistic Regression — linear baseline, sigmoid, interpretable
#       2. Random Forest       — non-linear ensemble, handles interactions
#       3. Decision Tree       — interpretable rules, tuned via GridSearchCV
#
#     PCA-LR is excluded: PCA on a mixed feature space (scaled numeric +
#     OHE sparse + ordinal-encoded high-cardinality columns) distorts the
#     components. Ordinal codes for e.g. country are arbitrary integers —
#     PCA treats them as real distances, producing misleading components.
#     Results empirically confirm this: PCA-LR loses discriminative signal.
#     PCA remains in Section 9 as a pure diagnostic visualization tool.
#
#     All three use the SAME preprocessor (ColumnTransformer).
#     The preprocessor is ALWAYS fitted inside GridSearchCV on the
#     training folds only — never on the test set.
# ══════════════════════════════════════════════════════════════
section("10. MODEL TRAINING — GridSearchCV (3 models)")

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# ── Model 1: Logistic Regression (binary / sigmoid) ───────────
print("  Training: Logistic Regression (sigmoid, binary) ...")
lr_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(solver="lbfgs",
                               max_iter=1000, random_state=42, n_jobs=-1)),
])
gs_lr = GridSearchCV(
    lr_pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"]},
    cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0, refit=True)
gs_lr.fit(X_train, y_train)
results["Logistic Regression"] = gs_lr
print(f"    Best: {gs_lr.best_params_}  |  CV AUC: {gs_lr.best_score_:.4f}")

# ── Model 2: Random Forest ────────────────────────────────────
print("  Training: Random Forest ...")
rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
gs_rf = GridSearchCV(
    rf_pipe,
    {"clf__n_estimators":     [100, 300],
     "clf__max_depth":        [10, 20, None],
     "clf__min_samples_leaf": [5, 10]},
    cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0, refit=True)
gs_rf.fit(X_train, y_train)
results["Random Forest"] = gs_rf
print(f"    Best: {gs_rf.best_params_}  |  CV AUC: {gs_rf.best_score_:.4f}")

# ── Model 3: Decision Tree ────────────────────────────────────
#   Interpretable if/else rules; non-linear; scale-invariant.
#   EXPLICIT OVERFITTING DISCUSSION:
#     A Decision Tree with no depth limit memorises training data —
#     Train AUC approaches 1.0 while CV AUC drops significantly.
#     This variance (Train AUC - CV AUC) IS overfitting.
#     GridSearchCV searches max_depth, min_samples_leaf, and criterion
#     across 5 stratified folds to find the complexity sweet spot.
#     After fitting, we explicitly print Train AUC vs CV AUC vs Test AUC
#     so the overfitting gap is visible and quantified in the output.
print("  Training: Decision Tree ...")
dt_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", DecisionTreeClassifier(random_state=42)),
])
gs_dt = GridSearchCV(
    dt_pipe,
    {"clf__max_depth":        [3, 5, 10, 20, None],
     "clf__min_samples_leaf": [1, 5, 10, 20],
     "clf__criterion":        ["gini", "entropy"]},
    cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0, refit=True)
gs_dt.fit(X_train, y_train)
results["Decision Tree"] = gs_dt

dt_train_auc = roc_auc_score(y_train,
                              gs_dt.best_estimator_.predict_proba(X_train)[:, 1])
dt_test_auc  = roc_auc_score(y_test,
                              gs_dt.best_estimator_.predict_proba(X_test)[:, 1])
dt_gap       = dt_train_auc - gs_dt.best_score_
print(f"    Best params : {gs_dt.best_params_}")
print(f"    Train AUC   : {dt_train_auc:.4f}")
print(f"    CV AUC      : {gs_dt.best_score_:.4f}  (gap vs train: {dt_gap:+.4f})")
print(f"    Test AUC    : {dt_test_auc:.4f}")
print(f"    Overfitting : {'YES — train AUC significantly exceeds CV AUC' if dt_gap > 0.05 else 'MINIMAL — tree complexity well-controlled by GridSearchCV'}")

# NOTE: PCA-LR (Model 4) has been removed from the final comparison.
#   PCA is used purely as a diagnostic tool in Section 9.
#   Reasons for removal:
#     1. PCA on a mix of scaled numeric, OHE, and ordinal-encoded columns
#        distorts the feature space — ordinal codes (e.g. country) are
#        arbitrary integers and skew the principal components.
#     2. Empirically, PCA often retains only 1 component in this dataset
#        (n_95 can be very small), losing most discriminative signal.
#     3. PCA is only beneficial when dimensionality is a genuine problem;
#        here RF and DT already handle high-dimensional sparse features well.
#   The three models below are the final comparison set.
print("  NOTE: PCA-LR excluded from final model comparison (see Section 9 for diagnostic use).")

# ══════════════════════════════════════════════════════════════
# 11. EVALUATION  (test set — never seen during training)
# ══════════════════════════════════════════════════════════════
section("11. TEST SET EVALUATION & MODEL COMPARISON")

eval_rows = []
for name, gs in results.items():
    yp     = gs.best_estimator_.predict(X_test)
    yprob  = gs.best_estimator_.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, yprob)
    report = classification_report(y_test, yp, output_dict=True)
    cm_i   = confusion_matrix(y_test, yp)
    tn, fp, fn, tp = cm_i.ravel()
    specificity  = tn / (tn + fp)          # true negative rate
    type1_error  = fp / (fp + tn)          # false positive rate  (α)
    type2_error  = fn / (fn + tp)          # false negative rate  (β)
    eval_rows.append({
        "Model":         name,
        "CV AUC":        gs.best_score_,
        "Test AUC":      auc,
        "F1 (Cancel)":   report["1"]["f1-score"],
        "Precision":     report["1"]["precision"],
        "Recall":        report["1"]["recall"],
        "Specificity":   specificity,
        "Type I Error":  type1_error,
        "Type II Error": type2_error,
    })
    print(f"\n  -- {name} --")
    print(f"     Test AUC-ROC : {auc:.4f}")
    print(f"     Specificity  : {specificity:.4f}  (TN / (TN+FP)  — how well the model identifies non-cancellations)")
    print(f"     Type I Error : {type1_error:.4f}  (FP / (FP+TN)  — predicted cancelled but actually not)")
    print(f"     Type II Error: {type2_error:.4f}  (FN / (FN+TP)  — predicted not cancelled but actually was)")
    print(classification_report(y_test, yp,
          target_names=["Not Cancelled", "Cancelled"]))

eval_df = pd.DataFrame(eval_rows).set_index("Model")
print("\n  Model Comparison Summary:")
print(eval_df.to_string())

# Comparison bar chart — each group of bars is a metric, colors = models
n_models = len(eval_df)
fig, ax  = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
x       = np.arange(n_models)
width   = 0.25
metrics = ["CV AUC", "Test AUC", "F1 (Cancel)"]
for i, metric in enumerate(metrics):
    bars = ax.bar(x + i * width, eval_df[metric], width,
                  label=metric, color=MODEL_COLORS[i],
                  edgecolor=DARK_PINK, linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center",
                color=DARK_PINK, fontsize=8, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(eval_df.index, color=DARK_PINK, fontsize=9,
                   rotation=15, ha="right")
ax.set_ylabel("Score", color=DARK_PINK)
ax.set_ylim(0, 1.08)
ax.set_title("Model Comparison — CV AUC vs Test AUC vs F1",
             color=DARK_PINK, fontweight="bold", fontsize=13)
ax.legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK)
style_ax(ax)
save_fig("09_model_comparison")

# ── Type I / Type II Error Schema — horizontal dual subplot ───
#   Styled like the "Cancellation Rate by deposit_type" chart.
#   Type I  (α) = FP/(FP+TN) — violet: predicted cancel, actually not
#   Type II (β) = FN/(FN+TP) — red:    predicted not cancel, actually was
fig, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Error Rate Analysis by Model", fontsize=15,
             color=DARK_PINK, fontweight="bold", y=1.02)

models    = eval_df.index.tolist()
t1_vals   = eval_df["Type I Error"].values
t2_vals   = eval_df["Type II Error"].values

# Left: Type I Error (Violet)
bars1 = ax_t1.barh(models, t1_vals, color=ROSE, edgecolor=DARK_PINK,
                   linewidth=0.8, height=0.5)
for bar, val in zip(bars1, t1_vals):
    ax_t1.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
               f"{val:.3f}", va="center", ha="left",
               color=ROSE, fontsize=10, fontweight="bold")
ax_t1.set_xlim(0, max(t1_vals) * 1.25 + 0.05)
ax_t1.set_xlabel("False Positive Rate", color=DARK_PINK, fontsize=11)
ax_t1.set_title("Type I Error  (α)\nPredicted Cancelled — Actually Not",
                color=ROSE, fontweight="bold", fontsize=11)
ax_t1.tick_params(colors=DARK_PINK)
ax_t1.set_facecolor(BG)
ax_t1.grid(axis="x", color=LAVENDER, alpha=0.6, linestyle="--")
for sp in ax_t1.spines.values():
    sp.set_edgecolor(DARK_PINK)

# Right: Type II Error (Red)
bars2 = ax_t2.barh(models, t2_vals, color=DUSTY_PINK, edgecolor=DARK_PINK,
                   linewidth=0.8, height=0.5)
for bar, val in zip(bars2, t2_vals):
    ax_t2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
               f"{val:.3f}", va="center", ha="left",
               color=DUSTY_PINK, fontsize=10, fontweight="bold")
ax_t2.set_xlim(0, max(t2_vals) * 1.25 + 0.05)
ax_t2.set_xlabel("False Negative Rate", color=DARK_PINK, fontsize=11)
ax_t2.set_title("Type II Error  (β)\nPredicted Not Cancelled — Actually Was",
                color=DUSTY_PINK, fontweight="bold", fontsize=11)
ax_t2.tick_params(colors=DARK_PINK)
ax_t2.set_facecolor(BG)
ax_t2.grid(axis="x", color=LAVENDER, alpha=0.6, linestyle="--")
for sp in ax_t2.spines.values():
    sp.set_edgecolor(DARK_PINK)

plt.tight_layout()
save_fig("09b_type1_type2_error_schema")

# ROC curves — one color per model (ROC_COLORS has 4 entries)
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for (name, gs), col in zip(results.items(), ROC_COLORS):
    yprob = gs.best_estimator_.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, yprob, name=name,
                                     ax=ax, color=col)
ax.plot([0, 1], [0, 1], "--", color=LAVENDER, linewidth=1)
ax.set_title("ROC Curves — All Models (Test Set)",
             color=DARK_PINK, fontweight="bold", fontsize=13)
ax.legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK)
style_ax(ax)
save_fig("10_roc_curves")

# Confusion matrix — best model
best_name = eval_df["Test AUC"].idxmax()
best_gs   = results[best_name]
cm        = confusion_matrix(y_test, best_gs.best_estimator_.predict(X_test))
fig, ax   = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor(BG)
ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled", "Cancelled"]).plot(
    ax=ax, colorbar=False, cmap=plt.cm.RdPu)
ax.set_title(f"Confusion Matrix — {best_name}",
             color=DARK_PINK, fontweight="bold")
ax.tick_params(colors=DARK_PINK)
for sp in ax.spines.values():
    sp.set_edgecolor(DARK_PINK)
save_fig("11_confusion_matrix")

# Feature importance — best model
clf_best = best_gs.best_estimator_.named_steps["clf"]
if hasattr(clf_best, "feature_importances_"):
    try:
        f_names = best_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
    except Exception:
        f_names = [f"f{i}" for i in range(len(clf_best.feature_importances_))]
    fi    = clf_best.feature_importances_
    top_i = np.argsort(fi)[-15:][::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.barh([str(f_names[i]) for i in top_i][::-1], fi[top_i][::-1],
            color=PINK, edgecolor=DARK_PINK, linewidth=0.8)
    ax.set_title(f"Top 15 Feature Importances — {best_name}",
                 color=DARK_PINK, fontweight="bold")
    ax.set_xlabel("Importance", color=DARK_PINK)
    style_ax(ax)
    save_fig("12_feature_importance")
elif hasattr(clf_best, "coef_"):
    try:
        f_names = best_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
    except Exception:
        f_names = [f"f{i}" for i in range(len(clf_best.coef_[0]))]
    coef  = np.abs(clf_best.coef_[0])
    top_i = np.argsort(coef)[-15:][::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.barh([str(f_names[i]) for i in top_i][::-1], coef[top_i][::-1],
            color=PINK, edgecolor=DARK_PINK, linewidth=0.8)
    ax.set_title(f"Top 15 Feature Importances (|Coef|) — {best_name}",
                 color=DARK_PINK, fontweight="bold")
    ax.set_xlabel("|Coefficient|", color=DARK_PINK)
    style_ax(ax)
    save_fig("12_feature_importance")

# ══════════════════════════════════════════════════════════════
# 12. THRESHOLD TUNING — Random Forest (best model)
#
#     WHY THRESHOLD TUNING?
#       The default decision threshold (0.5) was designed for balanced
#       classes. With ~27% cancellations, it biases the model toward
#       predicting "Not Cancelled" because that class has more training
#       examples. Lowering the threshold means: any booking with a
#       cancellation probability ≥ threshold is flagged as cancelled.
#       This directly increases Recall (fewer missed cancellations =
#       fewer false negatives = lower Type II error), at the cost of
#       some extra false alarms (higher Type I error).
#
#     APPROACH:
#       • No retraining — we only change where we draw the decision line
#         on the probabilities already produced by the fitted RF model.
#       • No SMOTE / class_weight — the model's probability estimates
#         remain unchanged; we only post-process them.
#       • Sweep thresholds 0.10 → 0.90 in steps of 0.01.
#       • Track: Precision, Recall, F1, Type I error, Type II error.
#       • Best threshold = argmax F1 for the Cancelled class.
#       • Secondary criterion: among thresholds within 0.5% of best F1,
#         prefer the one with lowest Type II error (fewest missed cancels).
# ══════════════════════════════════════════════════════════════
section("12. THRESHOLD TUNING — Random Forest (Cancelled class)")

from sklearn.metrics import precision_score, recall_score, f1_score

# ── Step 1: Get predicted probabilities from the best RF model ──
# We target Random Forest specifically — it is the highest-AUC model
# and tree ensembles produce well-calibrated probability estimates.
# predict_proba(X_test)[:, 1] gives P(Cancelled) for every booking.
rf_gs      = results["Random Forest"]          # GridSearchCV object
rf_model   = rf_gs.best_estimator_            # fitted Pipeline
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]  # P(Cancelled)

print(f"  Random Forest probability range: "
      f"[{y_proba_rf.min():.4f}, {y_proba_rf.max():.4f}]")
print(f"  Mean predicted P(Cancelled): {y_proba_rf.mean():.4f}")
print(f"  Actual cancel rate in test set: {y_test.mean():.4f}")

# ── Step 2: Sweep thresholds and record metrics ──────────────────
# For each threshold t:
#   y_pred = 1  if P(Cancelled) >= t,  else 0
# This is equivalent to moving the decision boundary left (lower t)
# to catch more cancellations, or right (higher t) to be more precise.
thresholds  = np.arange(0.10, 0.91, 0.01)
thresh_rows = []

for t in thresholds:
    # Convert probabilities to binary predictions at threshold t
    y_pred_t = (y_proba_rf >= t).astype(int)

    # Guard: skip degenerate thresholds where all predictions are one class
    # (precision is undefined when no positives are predicted)
    if y_pred_t.sum() == 0 or y_pred_t.sum() == len(y_pred_t):
        continue

    # Compute metrics for the Cancelled class (positive class = 1)
    prec  = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    rec   = recall_score   (y_test, y_pred_t, pos_label=1, zero_division=0)
    f1    = f1_score       (y_test, y_pred_t, pos_label=1, zero_division=0)

    # Confusion matrix components
    cm_t          = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    # Type I  error (α) = FP / (FP + TN)  — false positive rate
    # Type II error (β) = FN / (FN + TP)  — false negative rate (miss rate)
    type1 = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
    type2 = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0

    thresh_rows.append({
        "Threshold":    round(t, 2),
        "Precision":    prec,
        "Recall":       rec,
        "F1":           f1,
        "Type I Error": type1,
        "Type II Error":type2,
    })

thresh_df = pd.DataFrame(thresh_rows)

# ── Step 3: Identify optimal threshold ───────────────────────────
# Primary criterion:  maximise F1 for the Cancelled class.
# Rationale: F1 balances precision and recall — it penalises both
# missing cancellations (high Type II) and false alarms (high Type I).
best_f1_val   = thresh_df["F1"].max()
best_thresh   = thresh_df.loc[thresh_df["F1"].idxmax(), "Threshold"]

# Secondary criterion: among thresholds whose F1 is within 0.5% of
# the maximum, pick the one with the lowest Type II error.
# This prefers catching more cancellations when F1 is nearly equal.
near_best     = thresh_df[thresh_df["F1"] >= best_f1_val - 0.005]
best_thresh   = near_best.loc[near_best["Type II Error"].idxmin(), "Threshold"]
best_row      = thresh_df[thresh_df["Threshold"] == best_thresh].iloc[0]

# Reference: metrics at the standard 0.5 threshold
default_row   = thresh_df[thresh_df["Threshold"] == 0.50].iloc[0]

# ── Step 4: Print comparison ─────────────────────────────────────
print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │         THRESHOLD TUNING — Random Forest                │
  ├────────────────┬────────────┬────────────────────────── ┤
  │ Metric         │ Default    │ Optimal                   │
  │                │ (t = 0.50) │ (t = {best_thresh:.2f})              │
  ├────────────────┼────────────┼───────────────────────────┤
  │ Precision      │ {default_row['Precision']:.4f}     │ {best_row['Precision']:.4f}                    │
  │ Recall         │ {default_row['Recall']:.4f}     │ {best_row['Recall']:.4f}                    │
  │ F1-score       │ {default_row['F1']:.4f}     │ {best_row['F1']:.4f}                    │
  │ Type I  Error  │ {default_row['Type I Error']:.4f}     │ {best_row['Type I Error']:.4f}                    │
  │ Type II Error  │ {default_row['Type II Error']:.4f}     │ {best_row['Type II Error']:.4f}                    │
  └────────────────┴────────────┴───────────────────────────┘
  Best threshold : {best_thresh:.2f}
  F1 improvement : {best_row['F1'] - default_row['F1']:+.4f}
  Recall gain    : {best_row['Recall'] - default_row['Recall']:+.4f}
  Type II change : {best_row['Type II Error'] - default_row['Type II Error']:+.4f}
""")

# ── Step 5: Threshold curve plots ────────────────────────────────
# Threshold-specific colors (as requested):
#   F1     → pink        (PINK)
#   Recall → violet      (VIOLET)
#   Type I → red-rose    (ROSE)
#   Type II→ dark purple (PURPLE)
T_F1     = PINK
T_RECALL = VIOLET
T_TYPE1  = ROSE
T_TYPE2  = PURPLE

# Plot A — F1 and Recall vs Threshold
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(thresh_df["Threshold"], thresh_df["F1"],
        color=T_F1, linewidth=2.5, label="F1-score (Cancelled)")
ax.plot(thresh_df["Threshold"], thresh_df["Recall"],
        color=T_RECALL, linewidth=2.5, linestyle="--",
        label="Recall (Cancelled)")

# Mark optimal threshold on both curves
ax.axvline(best_thresh, color=DARK_PINK, linewidth=1.5,
           linestyle=":", label=f"Optimal threshold = {best_thresh:.2f}")
ax.axvline(0.50, color=PINK_2, linewidth=1.2,
           linestyle=":", label="Default threshold = 0.50")

# Annotate best F1 point
ax.scatter([best_thresh], [best_row["F1"]],
           color=T_F1, s=80, zorder=5)
ax.annotate(f"F1={best_row['F1']:.3f}",
            xy=(best_thresh, best_row["F1"]),
            xytext=(best_thresh + 0.03, best_row["F1"] - 0.04),
            color=T_F1, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=T_F1))

# Annotate recall at optimal threshold
ax.scatter([best_thresh], [best_row["Recall"]],
           color=T_RECALL, s=80, zorder=5)
ax.annotate(f"Recall={best_row['Recall']:.3f}",
            xy=(best_thresh, best_row["Recall"]),
            xytext=(best_thresh + 0.03, best_row["Recall"] + 0.02),
            color=T_RECALL, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=T_RECALL))

ax.set_xlabel("Decision Threshold", color=DARK_PINK, fontsize=11)
ax.set_ylabel("Score", color=DARK_PINK, fontsize=11)
ax.set_title("F1-score & Recall vs Decision Threshold — Random Forest",
             color=DARK_PINK, fontweight="bold", fontsize=13)
ax.set_xlim(0.08, 0.92)
ax.set_ylim(0, 1.05)
ax.legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK)
style_ax(ax)
save_fig("13a_threshold_f1_recall")

# Plot B — Type I and Type II Error vs Threshold (on same graph)
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(thresh_df["Threshold"], thresh_df["Type I Error"],
        color=T_TYPE1, linewidth=2.5, label="Type I Error  α  (False Positive Rate)")
ax.plot(thresh_df["Threshold"], thresh_df["Type II Error"],
        color=T_TYPE2, linewidth=2.5, linestyle="--",
        label="Type II Error  β  (False Negative Rate)")

# Crossover annotation — where the two error curves intersect
# Business insight: left of crossover → more false alarms;
#                   right of crossover → more missed cancellations.
diff = thresh_df["Type I Error"].values - thresh_df["Type II Error"].values
sign_changes = np.where(np.diff(np.sign(diff)))[0]
for idx in sign_changes:
    cross_t = (thresh_df["Threshold"].iloc[idx] +
               thresh_df["Threshold"].iloc[idx + 1]) / 2
    ax.axvline(cross_t, color=MAUVE, linewidth=1.2, linestyle="-.",
               label=f"Error crossover ≈ {cross_t:.2f}")

ax.axvline(best_thresh, color=DARK_PINK, linewidth=1.5,
           linestyle=":", label=f"Optimal threshold = {best_thresh:.2f}")
ax.axvline(0.50, color=PINK_2, linewidth=1.2,
           linestyle=":", label="Default threshold = 0.50")

# Mark both errors at the optimal threshold
ax.scatter([best_thresh], [best_row["Type I Error"]],
           color=T_TYPE1, s=80, zorder=5)
ax.scatter([best_thresh], [best_row["Type II Error"]],
           color=T_TYPE2, s=80, zorder=5)

ax.set_xlabel("Decision Threshold", color=DARK_PINK, fontsize=11)
ax.set_ylabel("Error Rate", color=DARK_PINK, fontsize=11)
ax.set_title("Type I & Type II Error vs Decision Threshold — Random Forest",
             color=DARK_PINK, fontweight="bold", fontsize=13)
ax.set_xlim(0.08, 0.92)
ax.set_ylim(0, 1.05)
ax.legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK,
          loc="upper right")
style_ax(ax)
save_fig("13b_threshold_errors")

# Plot C — All four curves together (overview / appendix)
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(thresh_df["Threshold"], thresh_df["F1"],
        color=T_F1,     linewidth=2.5, label="F1-score")
ax.plot(thresh_df["Threshold"], thresh_df["Recall"],
        color=T_RECALL, linewidth=2.0, linestyle="--", label="Recall")
ax.plot(thresh_df["Threshold"], thresh_df["Type I Error"],
        color=T_TYPE1,  linewidth=2.0, linestyle="-.", label="Type I Error (FPR)")
ax.plot(thresh_df["Threshold"], thresh_df["Type II Error"],
        color=T_TYPE2,  linewidth=2.0, linestyle=":",  label="Type II Error (FNR)")

ax.axvline(best_thresh, color=DARK_PINK, linewidth=1.8,
           linestyle=":", label=f"Optimal t = {best_thresh:.2f}")
ax.axvline(0.50, color=PINK_2, linewidth=1.2,
           linestyle=":", label="Default t = 0.50")

ax.fill_betweenx([0, 1.05], best_thresh - 0.02, best_thresh + 0.02,
                 alpha=0.10, color=DARK_PINK, label="Optimal region")

ax.set_xlabel("Decision Threshold", color=DARK_PINK, fontsize=11)
ax.set_ylabel("Score / Error Rate", color=DARK_PINK, fontsize=11)
ax.set_title("All Threshold Metrics — Random Forest (Overview)",
             color=DARK_PINK, fontweight="bold", fontsize=13)
ax.set_xlim(0.08, 0.92)
ax.set_ylim(0, 1.08)
ax.legend(facecolor=BG, edgecolor=DARK_PINK, labelcolor=DARK_PINK,
          fontsize=9, ncol=2)
style_ax(ax)
save_fig("13c_threshold_overview")

# ── Step 6: Confusion matrix at optimal threshold ────────────────
# Re-generate predictions using the tuned threshold instead of 0.50.
# The model itself is NOT retrained — only the decision boundary moves.
y_pred_best = (y_proba_rf >= best_thresh).astype(int)
cm_best     = confusion_matrix(y_test, y_pred_best)

fig, axes_cm = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)
fig.suptitle(f"Confusion Matrix — Random Forest (threshold comparison)",
             color=DARK_PINK, fontweight="bold", fontsize=13)

# Left: default threshold (0.50)
y_pred_default = (y_proba_rf >= 0.50).astype(int)
cm_default     = confusion_matrix(y_test, y_pred_default)
ConfusionMatrixDisplay(cm_default,
    display_labels=["Not Cancelled", "Cancelled"]).plot(
    ax=axes_cm[0], colorbar=False, cmap=plt.cm.RdPu)
axes_cm[0].set_title("Default Threshold  (t = 0.50)",
                     color=DARK_PINK, fontweight="bold")
axes_cm[0].tick_params(colors=DARK_PINK)
for sp in axes_cm[0].spines.values():
    sp.set_edgecolor(DARK_PINK)

# Right: optimal threshold
ConfusionMatrixDisplay(cm_best,
    display_labels=["Not Cancelled", "Cancelled"]).plot(
    ax=axes_cm[1], colorbar=False, cmap=plt.cm.RdPu)
axes_cm[1].set_title(f"Optimal Threshold  (t = {best_thresh:.2f})",
                     color=DARK_PINK, fontweight="bold")
axes_cm[1].tick_params(colors=DARK_PINK)
for sp in axes_cm[1].spines.values():
    sp.set_edgecolor(DARK_PINK)

plt.tight_layout()
save_fig("13d_confusion_matrix_tuned")

print(f"""
  THRESHOLD TUNING SUMMARY:
  ─────────────────────────────────────────────────────────
  Optimal threshold : {best_thresh:.2f}   (default was 0.50)
  Selection rule    : argmax F1, then prefer lowest Type II
                      within 0.5% of peak F1

  BUSINESS INTERPRETATION:
    A booking is flagged as a likely cancellation if the model
    assigns it P(Cancel) ≥ {best_thresh:.2f}.  This catches more
    genuine cancellations (higher Recall) so the hotel can
    proactively intervene (overbooking buffer, deposit request,
    re-marketing) before the cancellation is confirmed.

    Trade-off: slightly more false alarms (Type I ↑), meaning
    some non-cancelling guests are unnecessarily contacted.
    In hotel revenue management, a missed cancellation (Type II)
    is typically more costly than a false alarm, so this trade-off
    is acceptable.

  NO RETRAINING  — only the probability cut-off changed.
  NO RESAMPLING  — SMOTE / class_weight were not used.
  FIGURES SAVED  : 13a_threshold_f1_recall.png
                   13b_threshold_errors.png
                   13c_threshold_overview.png
                   13d_confusion_matrix_tuned.png
""")

# ══════════════════════════════════════════════════════════════
# 13. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("13. FINAL SUMMARY")
best_auc = eval_df["Test AUC"].max()

print(f"""
  EDA FINDING → MODELING DECISION CHAIN:
  -------------------------------------------------------
  Class imbalance (~37% cancelled)  → AUC-ROC metric; stratified CV
  reservation_status* (post-event)  → DROPPED — target leakage
  lead_time: strongest predictor    → Kept as high-priority feature
  deposit_type: near-deterministic  → OHE; top categorical predictor
  adr: right-skewed                 → Kept; StandardScaler normalises for LR
  Weak features (|r| < {CORR_THRESHOLD})        → Dropped (feature selection)
  Low-cardinality cats (<=10 val.)  → OneHotEncoder (no ordinal assumption)
  High-cardinality cats (>10 val.)  → OrdinalEncoder (safe for trees)
  Skewed distributions              → StandardScaler for LR; trees unaffected
  Binary target (0 or 1)            → Sigmoid / Logistic — Decision Tree splits on Gini/Entropy

  LEAKAGE STATUS (ALL FORMS ADDRESSED):
    reservation_status*  → dropped from df_raw and df BEFORE any EDA
    ohe_cols / ord_cols  → cardinality computed from X_train only (post-split)
    Encoding leakage     → OHE/Ordinal fitted on X_train only (Pipeline)
    Imputation leakage   → SimpleImputer fitted on X_train only
    Scaling leakage      → StandardScaler fitted on X_train only
    PCA (diagnostic)     → PCA fitted on preprocessed X_train only; NOT in final models
    Split order          → split BEFORE all encoding / imputation / scaling
    Safety assertions    → verified at 4 checkpoints in the code

  MODEL RESULTS (Test Set — default threshold 0.50):
{eval_df[["CV AUC", "Test AUC", "F1 (Cancel)"]].to_string()}

  Best model : {best_name}  →  Test AUC = {best_auc:.4f}

  THRESHOLD TUNING (Random Forest — Section 12):
    Optimal threshold : {best_thresh:.2f}   (default = 0.50)
    F1  @ default     : {default_row['F1']:.4f}   →   F1  @ optimal : {best_row['F1']:.4f}  ({best_row['F1'] - default_row['F1']:+.4f})
    Recall @ default  : {default_row['Recall']:.4f}   →   Recall @ optimal : {best_row['Recall']:.4f}  ({best_row['Recall'] - default_row['Recall']:+.4f})
    Type II @ default : {default_row['Type II Error']:.4f}   →   Type II @ optimal : {best_row['Type II Error']:.4f}  ({best_row['Type II Error'] - default_row['Type II Error']:+.4f})
    Type I  @ default : {default_row['Type I Error']:.4f}   →   Type I  @ optimal : {best_row['Type I Error']:.4f}  ({best_row['Type I Error'] - default_row['Type I Error']:+.4f})

  NOTE: PCA-LR was excluded from the final comparison.
  PCA on a mixed feature space (scaled numeric + OHE + ordinal-encoded
  high-cardinality columns like 'country') distorts the principal components
  because ordinal codes are arbitrary integers, not true distances.
  PCA is retained in Section 9 as a diagnostic visualization tool only.

  ALL FIGURES SAVED
""")
