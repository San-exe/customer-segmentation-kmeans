import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#no dependent variable is used
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

#scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
#list of wcss
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    #init tells KMeans how to choose the initial cluster centroids.
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    #The total sum of squared distances of all data points to their nearest centroid.

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#training the kmeans  model on dataset
kmeans = KMeans(n_clusters=5,init="k-means++",random_state=0)
#creating the dependent variable
y_kmeans = kmeans.fit_predict(x)
print(kmeans.cluster_centers_)

#visualising the clustors
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],s=100,color = "lime",label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],s=100,color = "blue",label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],s=100,color = "pink",label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1],s=100,color = "yellow",label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1],s=100,color = "brown",label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300 ,color = "cyan",label = 'Centroids')
plt.title("Clusters of Customers")
plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.legend()
plt.show()


'''
High income - High spending (Premium customers)
High income - Low spending (Careful / conservative spenders)
Low income - High spending (Target for loyalty programs)
Low income - Low spending (Low-value segment)
Average income - Average spending (Regular customers)
'''


"""
Project Explanation:

In this project, I applied the K-Means clustering algorithm to segment customers of a retail store based on their annual income and spending score. 
I used the Elbow Method to determine the optimal number of clusters, which was found to be 5. 
Since K-Means is a distance-based algorithm, I scaled the features using StandardScaler to ensure both variables contributed equally. 
The model grouped customers into meaningful segments such as high-income high-spenders and low-income low-spenders. 
This segmentation can help businesses better understand customer behavior and design targeted marketing strategies.

"""