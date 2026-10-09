from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def print_terminal_scatter(x_values, y_values, title, width=70, height=20):
    """Print a compact scatter plot using plain terminal characters."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x_value, y_value in zip(x_values, y_values):
        col = int((x_value - x_min) / x_span * (width - 1))
        row = height - 1 - int((y_value - y_min) / y_span * (height - 1))
        grid[row][col] = "*"

    print(f"\n{title}")
    print(f"PC value range: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")

# Load the dataset
csv_path = Path(__file__).resolve().parent / "covid19.csv"
df = pd.read_csv(csv_path)

print("Dataset preview:")
print(df.head())
print("\nShape:", df.shape)

# Keep only the numeric columns for PCA
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if "Country/Region" in numeric_cols:
    numeric_cols.remove("Country/Region")

if not numeric_cols:
    raise ValueError("No numeric columns found for PCA.")

X = df[numeric_cols].fillna(0).astype(float)

# Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA on the standardized data
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Choose the smallest number of components that keeps at least 95% of variance
target_variance = 0.95
n_components = int(np.argmax(cumulative_variance >= target_variance) + 1)

# Refit PCA with the chosen number of components
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

print("\nOriginal number of dimensions:", X_scaled.shape[1])
print("Number of components selected:", n_components)
print("Cumulative variance retained:", round(cumulative_variance[n_components - 1], 4))
print("\nExplained variance ratio per component:")
for i, ratio in enumerate(explained_variance[:n_components], start=1):
    print(f"PC{i}: {ratio:.4f}")

# Create a reduced dataframe for easier analysis
reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i}" for i in range(1, n_components + 1)],
    index=df.index,
)

print("\nReduced dataset preview:")
print(reduced_df.head())

# Plot cumulative explained variance to visualize the PCA effect
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(y=target_variance, color="r", linestyle="--", label=f"{target_variance * 100:.0f}% variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Variance Retained vs Number of Components")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(csv_path.parent / "pca_variance.png", dpi=150)
print(f"Saved variance plot: {csv_path.parent / 'pca_variance.png'}")

# Optional: show the first 2 principal components as a 2D projection
if n_components >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
    for i, country in enumerate(df["Country/Region"]):
        plt.annotate(country, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=7, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Countries projected onto the first two principal components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(csv_path.parent / "pca_2d.png", dpi=150)
    print(f"Saved 2D PCA plot: {csv_path.parent / 'pca_2d.png'}")
    print_terminal_scatter(
        X_reduced[:, 0], X_reduced[:, 1], "2D PCA graph (PC1 vs PC2)"
    )

# Plot the first three principal components as a 3D projection
if n_components >= 3:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.7)
    for i, country in enumerate(df["Country/Region"]):
        ax.text(X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], country, fontsize=6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Countries projected onto the first three principal components")
    fig.tight_layout()
    fig.savefig(csv_path.parent / "pca_3d.png", dpi=150)
    print(f"Saved 3D PCA plot: {csv_path.parent / 'pca_3d.png'}")

    # A terminal cannot draw a true 3D image, so print an isometric 2D projection.
    x_3d = (X_reduced[:, 0] - X_reduced[:, 1]) * 0.866
    y_3d = (X_reduced[:, 0] + X_reduced[:, 1]) * 0.5 - X_reduced[:, 2]
    print_terminal_scatter(x_3d, y_3d, "3D PCA graph (isometric PC1/PC2/PC3 projection)")

