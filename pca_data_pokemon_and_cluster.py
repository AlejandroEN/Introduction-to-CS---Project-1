import pandas as pd

pca_data = pd.read_csv("pca_matrix_and_cluster.csv")
pca_data.drop(pca_data.columns[0:-1], axis=1, inplace=True)

pokemon_data = pd.read_csv("smogon.csv")
pca_data["Pokemon"] = pokemon_data["Pokemon"]

pca_data.to_csv("pca_data_pokemon_and_cluster.csv")
print("El archivo pca_data_pokemon_and_cluster.csv se ha creado con éxito")