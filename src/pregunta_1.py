import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

data: pd.DataFrame = pd.read_csv("../res/smogon.csv")
stop_words: list[str] = stopwords.words("english")

# Punto 1
tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(data["moves"])

# Punto 2
print(f"El número total de tokens (vocabulario) es {len(tfidf_vectorizer.vocabulary_)}", end="\n\n")

# Punto 3
print(f"Los tokens son\n {sorted(tfidf_vectorizer.vocabulary_)}", end="\n\n")

# Punto 4
tfidf_dataframe = pd.DataFrame(data=tfidf_matrix.toarray(), columns=sorted(tfidf_vectorizer.vocabulary_))
print(f"El dataframe es\n {tfidf_dataframe}", end="\n\n")

# Punto 5
km = KMeans(n_clusters=57, n_init=100)
cluster = km.fit_predict(tfidf_dataframe)
tfidf_dataframe["Grupo"] = cluster

# Punto 6
tfidf_dataframe.to_csv("../res/tfidf_matrix_and_cluster.csv")
print("El archivo tfidf_matrix_and_cluster.csv se ha creado con éxito")