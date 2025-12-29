# Customer Segmentation & Clustering (K-means)

**Tech:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn  
**Goal:** Segment customers using clustering and generate business-oriented insights.

## What this project does
- Exploratory Data Analysis (EDA): distributions + correlation heatmap
- Preprocessing: log-transform + standardization
- Choose the best K using Silhouette Score
- Train K-means and profile segments
- Generate business recommendations for each segment
- Visualize clusters using PCA 2D

## Run
```bash
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py -m src.main
