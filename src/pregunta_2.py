import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Punto 1
tfidf_data: pd.DataFrame = pd.read_csv("../res/tfidf_matrix_and_cluster.csv")

# Punto 2
tfidf_data.drop(["Grupo"], axis=1, inplace=True)
print(f"El dataframe es\n {tfidf_data}", end="\n\n")

# Punto 3
tfidf_data.drop(tfidf_data.columns[0], axis=1, inplace=True)
print(f"El nuevo dataframe es\n {tfidf_data}", end="\n\n")

# Punto 4
pca = PCA(625)
pca.fit(tfidf_data)
pca_projection = pca.transform(tfidf_data)

# Punto 5
print(f"El número de filas y columnas del dataframe original es, respectivamenete, {tfidf_data.shape[0]} y {tfidf_data.shape[1]}", end="\n\n")

# Punto 6
print(f"El número de filas y columnas de la matriz de componentes principales es, respectivamente, {pca_projection.shape[0]} y {pca_projection.shape[1]}", end="\n\n")

# Punto 7
pca_dataframe = pd.DataFrame(pca_projection, columns=[f"PCA {str(i)}" for i in range(1, len(pca.components_) + 1)])
print(pca_dataframe)

# Punto 8
km = KMeans(n_clusters=57, n_init=100)
cluster = km.fit_predict(pca_dataframe)
pca_dataframe["Grupo"] = cluster

# Punto 9
pca_dataframe.to_csv("../res/pca_matrix_and_cluster.csv")
print("El archivo pca_matrix_and_cluster.csv se ha creado con éxito")