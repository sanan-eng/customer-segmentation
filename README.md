# 🛒 Customer Segmentation using K-Means and DBSCAN

## 📌 Description
This project performs *customer segmentation* on the Mall Customers dataset using clustering algorithms.  
The goal is to group customers based on their *Annual Income* and *Spending Score* for better business insights.

We implemented:
- *K-Means Clustering* (with Elbow Method for optimal k)
- *DBSCAN Clustering* (with parameter tuning using k-distance graph)

---

## 📂 Dataset
- *File:* Mall_Customers.csv  
- *Source:* Kaggle (Mall Customers Dataset)  

---

## 🛠 Technologies & Libraries
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 🚀 Steps Performed
1. *Data Cleaning*  
   - Checked for null and duplicate values  
2. *Data Scaling*  
   - StandardScaler applied on Income & Spending Score  
3. *K-Means Clustering*  
   - Elbow method used → Optimal k = 5  
   - Visualized clusters with centroids  
4. *DBSCAN Clustering*  
   - Tuned eps and min_samples  
   - Formed 3 clusters with 8 noise points  
5. *Bonus Analysis*  
   - Calculated *average spending score per cluster*  

---

## 📊 Results
- *K-Means* → 5 customer clusters identified  
- *DBSCAN* → 3 clusters + 8 noise points  
- *Average Spending per Cluster* calculated  

---

## 🖥 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/customer-segmentation.git# customer-segmentation
Clustering customers using KMeans and DBSCAN (Unsupervised ML Project)
