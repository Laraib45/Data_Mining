""" Customer Segmentation based on the Nutrients """

from matplotlib import pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

df = pd.read_csv('F:\\pythonProject\\Data_Mining\\Clustering\\Nutrient_Composition\\Nutrient Composition Dataset.csv')
print(df.shape)

print(df.iloc[:, 1:5].describe())
data_1 = df.iloc[:, 1:5]
print(data_1.head(10))
data_1.describe()
data_scaled = normalize(data_1)
data_scaled = pd.DataFrame(data_scaled, columns=data_1.columns)
print(data_scaled.head(10))
w_link = linkage(data_scaled , method = 'ward')
dend = dendrogram(w_link)
dend = dendrogram(w_link,
                 truncate_mode='lastp',
                 p = 10,
                 )
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
plt.figure(figsize=(10, 7))
print(cluster.fit_predict(data_scaled))
plt.title("Customer_Segmentation")
plt.scatter(data_scaled['vitaminC'], data_scaled['Fibre'], c=cluster.labels_)
plt.draw()
plt.savefig("scatter.jpg") #Saving the scatter plot as image
plt.show()
clusters = fcluster(w_link, 3, criterion='maxclust')
clusters
df['clusters'] = clusters # Creating new column as cluster and adding the value
print(df.head(10))
df.to_csv('solved_cluster_nutrient.csv') #Saving the analysed file