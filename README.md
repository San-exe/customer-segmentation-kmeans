# 🛍️ Customer Segmentation using K-Means Clustering

This project uses K-Means clustering to segment customers based on their annual income and spending score.

## 📊 Features Used
- Annual Income
- Spending Score

## ⚙️ Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 🔍 Approach

- Data preprocessing and feature scaling using StandardScaler
- Used Elbow Method to determine optimal clusters
- Applied K-Means clustering (k = 5)
- Visualized clusters and centroids

## 📈 Customer Segments

- High Income - High Spending (Premium customers)
- High Income - Low Spending (Careful spenders)
- Low Income - High Spending (Target customers)
- Low Income - Low Spending (Low value)
- Average Income - Average Spending (Regular customers)

## ▶️ How to Run

```bash
pip install -r requirements.txt
python kmeans_segmentation.py