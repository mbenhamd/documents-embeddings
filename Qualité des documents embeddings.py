
# coding: utf-8

# # Projet Text Mining

# ## Sujet 1. Qualité des document embeddings

# Membres : Mohamed BEN HAMDOUNE | Louis BOURQUARD | Lucas ISCOVICI

# In[1]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import *
from sklearn.decomposition import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import *
from sklearn.cluster import MiniBatchKMeans
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm,tqdm_pandas,tqdm_notebook
import os
from progress.bar import Bar
from ipywidgets import FloatProgress
import time
import os
from sklearn.cluster import KMeans
from tqdm import tqdm,tqdm_pandas,tqdm_notebook
tqdm.pandas(desc="progress-bar")
from sklearn.decomposition import NMF
from collections import Counter
from sklearn import preprocessing
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize 
import nltk
import multiprocessing
import scikitplot as skplt
from mpl_toolkits.mplot3d import Axes3D
from spherecluster import SphericalKMeans
from sklearn.feature_extraction import text 
#from word_cloud.word_cloud_generator import WordCloud 
#from IPython.core.display import HTML
import pandas as pd


# In[2]:


nltk.download('punkt')


# In[3]:


pathDonnees="donnees/"


# In[4]:


readTable = lambda filename,sep="\n": pd.read_table(pathDonnees+"/"+filename,delimiter=sep,header=None,decimal=".")


# In[5]:


def get_top_n_words(bag_of_words,vec, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    """
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[182]:


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc[1], model.infer_vector(doc[0], steps=20, alpha=0.025)) for doc in sents])
    return targets, regressors


# In[41]:


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


# In[42]:


def scoring_cluster(methode,encoded_label,y_pred):
    ari = adjusted_rand_score(encoded_label,y_pred)
    nmi = normalized_mutual_info_score(encoded_label,y_pred,average_method='arithmetic')
    accuracy = acc(encoded_label,y_pred)
    print("ARI:",round(ari*100,4),"NMI:",round(nmi*100,4),"Accuracy:",round(accuracy*100,4))


# In[325]:


def analyse(methode,preproc,true_label,nb_clusters=3,normalizer=True,scikit=True):        
        if scikit:
            data = methode.fit_transform(preproc)
        else:
            data = preproc
        if normalizer:
            data = Normalizer(norm='l2',copy=False).fit_transform(data)
        skplt.cluster.plot_elbow_curve(SphericalKMeans(random_state=42,n_jobs=-1),data,title = "Elbow Curve avec Spherical K-means" ,cluster_ranges=range(1, 15))
        skplt.cluster.plot_elbow_curve(KMeans(random_state=42,n_jobs=-1,precompute_distances=True),data,title="Elbow Curve avec K-means" ,cluster_ranges=range(1, 15))
        ("Fitting For Spherical K-means for ",nb_clusters,"...")
        skmeans = SphericalKMeans(n_clusters=nb_clusters,random_state=42,n_jobs=-1).fit(data)
        ("Fitting For Spherical K-means for ",nb_clusters,"...")
        kmeans = KMeans(n_clusters=nb_clusters,random_state=42,n_jobs=-1,precompute_distances=True).fit(data)
        y_pred_skmeans = skmeans.predict(data)
        y_pred_kmeans = kmeans.predict(data)
        print("Results from Spherical K-means")
        scoring_cluster(skmeans,true_label,y_pred_skmeans)
        print("Results from K-means")
        scoring_cluster(kmeans,true_label,y_pred_kmeans)
        return methode,skmeans,kmeans,data


# In[44]:


classic_raw=readTable("classic3_raw.txt")
classic_label=readTable("classic3_labels.txt")
classic_doc_50_dims = readTable("classic3_doc_50_dims.txt")
reuters_raw = readTable("reuters8_raw.txt")
reuers_labels = readTable("reuters8_labels.txt")
reuters_embeddings = np.loadtxt("./donnees/reuters8_embeddings.txt")
classic_embeddings_200 = np.loadtxt("./donnees/classic3_doc_200.txt")
classic_embeddings_25 = np.loadtxt("./donnees/classic3_doc_25_dims.txt")
classic_embeddings_50 = np.loadtxt("./donnees/classic3_doc_50_dims.txt")
classic_embeddings_150 = np.loadtxt("./donnees/classic3_doc_150_dims.txt")


# #### CountVectorizer et TD-IDF

# In[45]:


my_additional_stop_words = ['said']
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


# In[46]:


vectorizer = CountVectorizer(stop_words = stop_words )
tf_idf = TfidfVectorizer(stop_words = stop_words )


# ### Analyse sur Reuers 8

# In[47]:


reuers_labelU = np.unique(reuers_labels.values)
reuers_labelU


# In[48]:


len(reuters_raw.values)


# In[49]:


X_countVectorizer_Reuers8 = vectorizer.fit_transform(reuters_raw.iloc[:,0])


# In[50]:


X_tfidf_Reuers8 = tf_idf.fit_transform(reuters_raw.iloc[:,0])


# In[17]:


X_countVectorizer_Reuers8.shape


# In[18]:


non_zero = np.count_nonzero(X_countVectorizer_Reuers8.toarray())
total_val = np.product(X_countVectorizer_Reuers8.shape)
sparsity = (total_val - non_zero) / total_val


# In[19]:


print("La matrice est sparse à ",round(sparsity*100.,4),"%")


# In[20]:


list_num_word = get_top_n_words(X_tfidf_Reuers8,tf_idf,20)


# In[22]:


for i in range(8):    
    X = tf_idf.fit_transform(reuters_raw[reuers_labels==reuers_labelU[i]].dropna().iloc[:,0])
    list_num_word = get_top_n_words(X,tf_idf,5)
    x,y = zip(*list_num_word)
    plt.figure(figsize=(30, 6))
    plt.title(reuers_labelU[i])
    pd.Series(y,index=x).plot.bar()
    plt.xticks(rotation=0,size=18)
plt.show()


# In[23]:


x,y = zip(*list_num_word)
plt.figure(figsize=(24, 6))
plt.bar(x,y)
plt.xticks(range(len(x)),x)
plt.xticks(rotation=0,size=20)
plt.show()


# In[24]:


val_perc = Counter(list(reuers_labels.values.ravel()))
x_lab = [(i, val_perc[i] / len(reuers_labels.values.ravel()) * 100.0) for i in val_perc]
pd.Series(Counter(list(reuers_labels.values.ravel()))).sort_values(ascending=False).plot(kind='bar')
plt.xticks(rotation=0)
print(x_lab)


# ### Analyse sur Classic3

# Les noms de classes disponible.

# In[25]:


classic_labelUC = np.unique(classic_label.values)
classic_labelUC


# On dénombre 3891 lignes dans le document brut.

# In[26]:


len(classic_raw.values)


# In[27]:


X_countVectorizer_Classic3 = vectorizer.fit_transform(classic_raw.iloc[:,0])


# In[28]:


X_tfidf_Classic3 = tf_idf.fit_transform(classic_raw.iloc[:,0])


# CountVectorizer représentant sur les lignes un document et en colonnes chaque mot  

# In[29]:


X_tfidf_Classic3.shape


# In[30]:


non_zero = np.count_nonzero(X_tfidf_Classic3.toarray())
total_val = np.product(X_tfidf_Classic3.shape)
sparsity = (total_val - non_zero) / total_val


# In[31]:


print("La matrice est sparse à ",round(sparsity*100.,4),"%")


# In[32]:


list_num_word_classic = get_top_n_words(X_tfidf_Classic3,tf_idf,5)


# In[33]:


for i in range(3):    
    X = tf_idf.fit_transform(classic_raw[classic_label==classic_labelUC[i]].dropna().iloc[:,0])
    list_num_word = get_top_n_words(X,tf_idf,5)
    x,y = zip(*list_num_word)
    plt.figure(figsize=(24, 6))
    plt.title(classic_labelUC[i])
    pd.Series(y,index=x).plot.bar()
    plt.xticks(rotation=0,size=20)

plt.show()


# In[34]:


x,y = zip(*list_num_word_classic)
plt.figure(figsize=(36, 6))
plt.bar(x,y)
plt.xticks(size=25)
plt.show()


# In[35]:


val_perc = Counter(list(classic_label.values.ravel()))
x_lab = [(i, val_perc[i] / len(classic_label.values.ravel()) * 100.0) for i in val_perc]
pd.Series(Counter(list(classic_label.values.ravel()))).plot(kind='bar')
plt.xticks(rotation=0)
print(x_lab)


# ### Encodage des labels sur Classic3 et Reurs8

# In[36]:


le = preprocessing.LabelEncoder()
#### Encodage sur le jeux de données classic3
encoded_label_classic = le.fit_transform(classic_label.values.ravel())
label_encoded_classic = le.inverse_transform(encoded_label_classic)
#### Encodage sur le jeux de données Reuers
encoded_label_reuers = le.fit_transform(reuers_labels.values.ravel())
label_encoded_reuers = le.inverse_transform(encoded_label_reuers)


# In[37]:


skmeans = SphericalKMeans(random_state=0)
kmeans = KMeans(random_state=0)


# In[37]:


#wc=WordCloud(use_tfidf=False,stopwords=ENGLISH_STOP_WORDS)


# In[38]:


# World Cloud sur Classic Raw
#embed_code=wc.get_embed_code(text=classic_raw[0].values,random_color=True,topn=100)
#HTML(embed_code)


# In[39]:


# World Cloud sur Reuters Raw
#embed_code=wc.get_embed_code(text=reuters_raw[0].values,random_color=True,topn=100)
#HTML(embed_code)


# #### Analyse sur la matrice embeddings Classic3 en utilisant K-means et Spherical K-means

# In[40]:


# Matrice embeddings de taille 25
analyse(methode=None,preproc=classic_embeddings_25,true_label=encoded_label_classic,nb_clusters=3,scikit=False,normalizer=False)


# In[41]:


# Matrice embeddings de taille 50
analyse(methode=None,preproc=classic_embeddings_50,true_label=encoded_label_classic,nb_clusters=3,scikit=False,normalizer=False)


# In[42]:


# Matrice embeddings de taille 150
analyse(methode=None,preproc=classic_embeddings_150,true_label=encoded_label_classic,nb_clusters=3,scikit=False,normalizer=False)


# In[43]:


# Matrice embeddings de taille 200
analyse(methode=None,preproc=classic_embeddings_200,true_label=encoded_label_classic,nb_clusters=3,scikit=False,normalizer=False)


# ### Analyse Sur Classic3 en utilisant la LSA avec K-means et Sperical K-means 

# In[44]:


lsa = TruncatedSVD(20, random_state=42,n_iter = 20)


# In[45]:


analyse(lsa,X_countVectorizer_Classic3,encoded_label_classic,normalizer=False)


# In[46]:


analyse(lsa,X_countVectorizer_Classic3,encoded_label_classic)


# In[47]:


analyse(lsa,X_tfidf_Classic3,encoded_label_classic,normalizer=False)


# In[48]:


analyse(lsa,X_tfidf_Classic3,encoded_label_classic,normalizer=True)


# ### NMF avec Classic3

# In[49]:


#initialization (better for sparseness)
# ‘nndsvd’: Nonnegative Double Singular Value Decomposition (NNDSVD)
model = NMF(n_components=20,init='nndsvd',max_iter=1000,random_state=42)


# In[50]:


analyse(model,X_tfidf_Classic3,encoded_label_classic,normalizer=False)


# In[51]:


analyse(model,X_tfidf_Classic3,encoded_label_classic)


# In[52]:


analyse(model,X_countVectorizer_Classic3,encoded_label_classic)


# In[53]:


analyse(model,X_countVectorizer_Classic3,encoded_label_classic,normalizer=False)


# #### DOC2VEC avec Classic3

# In[373]:


tagged_data = [TaggedDocument(words=word_tokenize(i.lower()), tags=[str(_d)]) for i, _d in zip(classic_raw.values.ravel(),encoded_label_classic)]


# In[374]:


len(vectorizer.get_feature_names())


# In[375]:


max_epochs = 100
alpha = 0.025

model = Doc2Vec(alpha=alpha,
                vector_size = 20,
                min_alpha=0.00025,
                min_count=1,
                dm=1,
                epoch=max_epochs,
                workers=multiprocessing.cpu_count())
  
model.build_vocab([x for x in tqdm_notebook(tagged_data)])

for epoch in tqdm_notebook(range(max_epochs)):
    model.train(tagged_data,
                total_examples=model.corpus_count,epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


# In[376]:


tagged_data = pd.DataFrame(TaggedDocument(words=word_tokenize(i.lower()), tags=[str(_d)]) for i, _d in zip(classic_raw.values.ravel(),encoded_label_classic))


# In[377]:


y_train, X_train = vec_for_learning(model, tagged_data)


# In[378]:


analyse(model,X_train,encoded_label_classic,normalizer=True,scikit=False)


# In[379]:


analyse(model,X_train,encoded_label_classic,normalizer=False,scikit=False)


# #### Analyse sur Reurs8-Embeddings avec Kmeans et Skmeans

# In[ ]:


analyse(methode=None,preproc=reuters_embeddings,true_label=encoded_label_reuers,nb_clusters=8,scikit=False,normalizer=True)


# In[ ]:


analyse(methode=None,preproc=reuters_embeddings,true_label=encoded_label_reuers,nb_clusters=8,scikit=False,normalizer=False)


# ### Analyse Sur Reuers8 en utilisant la LSA avec K-means et Sperical K-means 

# In[67]:


lsa_reuers = TruncatedSVD(20,random_state=42)


# In[68]:


analyse(lsa_reuers,X_countVectorizer_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=True)


# In[69]:


analyse(lsa_reuers,X_countVectorizer_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=False)


# In[70]:


analyse(lsa_reuers,X_tfidf_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=False)


# In[71]:


analyse(lsa_reuers,X_tfidf_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=True)


# ### NMF sur Reuers8

# In[62]:


model = NMF(n_components=20, init='nndsvd',max_iter=1000,random_state=42)


# In[63]:


analyse(model,X_countVectorizer_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=True)


# In[64]:


analyse(model,X_countVectorizer_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=False)


# In[65]:


analyse(model,X_tfidf_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=True)


# In[66]:


analyse(model,X_tfidf_Reuers8,encoded_label_reuers,nb_clusters=8,normalizer=False)


# ### DOC2VEC

# In[366]:


tagged_data = [TaggedDocument(words=word_tokenize(i.lower()), tags=[str(_d)]) for i, _d in zip(reuters_raw.values.ravel(),encoded_label_reuers)]


# In[367]:


len(vectorizer.get_feature_names())


# In[368]:


max_epochs = 100
alpha = 0.025

model = Doc2Vec(alpha=alpha,
                vector_size = 20,
                min_alpha=0.00025,
                min_count=1,
                dm=0,
                workers=multiprocessing.cpu_count())
  
model.build_vocab([x for x in tqdm_notebook(tagged_data)])

for epoch in tqdm_notebook(range(max_epochs)):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


# In[369]:


tagged_data = pd.DataFrame(TaggedDocument(words=word_tokenize(i.lower()), tags=[str(_d)]) for i, _d in zip(reuters_raw.values.ravel(),encoded_label_reuers))


# In[370]:


y_train, X_train = vec_for_learning(model, tagged_data)


# In[371]:


analyse(model,X_train,encoded_label_reuers,nb_clusters=8,normalizer=True,scikit=False)


# In[372]:


analyse(model,X_train,encoded_label_reuers,nb_clusters=8,normalizer=False,scikit=False)

