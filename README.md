# COVID-19 PCA Project

This project applies principal component analysis (PCA) to country-level COVID-19 data.

## Run

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\covid19.py
```

The script reads `covid19.csv`, prints the PCA results and terminal scatter plots, and saves `pca_variance.png`, `pca_2d.png`, and `pca_3d.png` in the project directory.
