#!/usr/bin/env python
# coding: utf-8

# ## MNB & RL Classification 

# In[2]:


#########################
##Import des librairies #
########################

from os import listdir
from os.path import isfile, join,isdir
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import re,json
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
stemmer = SnowballStemmer('english')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import seaborn as sns; sns.set()


# In[4]:


##########################################
## Définition du path pour la data ######
########################################

path = 'bbc/'


# In[5]:


def list_classes(path) : 
    classes = [d for d in listdir(path) if isdir(join(path,d))]
    return classes


# In[6]:


classes = list_classes(path)
#print("les differentes classes sont : ")
#print(classes)


# In[7]:


# Recupere le contenu de chaque dossier correspendant à une classe 
# retourne le DataFrame regroupant les classes et leurs contenus 
####################################################################

def to_df_with_content(liste_classes) :
    data = pd.DataFrame({"fic":[],"classe":[]})
    for c in classes : 
        print(str(c))
        liste = [f for f in listdir(join(path,c)) if isfile(join(path+c,f))]
        df_tmp = pd.DataFrame(liste,columns=["fic"])
        df_tmp["classe"] = c
    
        data = data.append(df_tmp).reset_index(drop=True)
        
    return data


# In[8]:


data = to_df_with_content(classes)


# In[9]:


data.head()


# In[10]:


#################################
#count nmbr document per classe #
################################
#print("============================================================")
#print(data.groupby("classe").count().rename(columns={"classe":"classe","fic":"count"}).T)


# In[11]:


def fic_to_string(fichier):
    file = open(fichier, "r",encoding="utf-8") 
    return file.read() 


# In[12]:


data.loc[data.fic.str.contains("262.txt")].index.values.astype(int)


# In[13]:


data.index.values.astype(int)


# In[14]:


#df.apply(lambda x : ) ? pas possible ! 

def add_content_to_df(data) : 
    data["contenu"] = "/"
    for i in data.index.values.astype(int) :
        try : 
            #print(join(path+data["classe"][i],data["fic"][i]))
            data["contenu"][i] = fic_to_string(join(path+data["classe"][i],data["fic"][i]))
        except :
            data["contenu"][i] = None
    print("-------------------------------------------------------")
    print("Contenu récuperé ")


# In[15]:


print("Recupération du contenu ")
add_content_to_df(data)


# In[16]:


data[data.contenu.isna()].count()[0]


# In[17]:


data.head()


# In[18]:


data.tail()


# In[19]:


data = data[['fic', 'contenu', 'classe']]
data.head()


# In[20]:


data.shape


# In[21]:


data = data[~data.contenu.isna()]
data.shape


# In[22]:


print("Sauvegarde des données ...")
data.to_csv('data_news.csv',index =False)


# In[23]:


###########################
#stem & remove stopwords ##
###########################
print("stem & remove stopwords")
data["contenu_clean"] = data["contenu"].apply(lambda x : ' '.join([stemmer.stem(w) for w in str(x).split() if w not in set(stopwords.words('english'))]))


# In[24]:


data.head()


# In[25]:


"""import operator
n = 1 
vectorizer = CountVectorizer(binary=True, ngram_range=(n, n))#, stop_words=stop_stemmed)
X_df = vectorizer.fit_transform(data['contenu_clean'])

# CONSOLIDATION
a = np.asarray(X_df.sum(axis=0)).ravel().tolist()
b = vectorizer.vocabulary_.items()
b = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
b1 = [item[0] for item in b]
df_page = pd.DataFrame([b1,a]).transpose()
df_page.columns = ['mot','nb_page']
df_page = df_page.sort_values('nb_page', ascending=False)
df_page.reset_index(inplace=True, drop = True)
df_page"""


# In[26]:


print("count values per classes : ")
print(data.classe.value_counts())


# In[27]:


X = data.iloc[:,3].values
y = data.iloc[:,2].values


# In[28]:


set(y)


# In[29]:


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("Encode classes : Done ")


# In[30]:


set(y)


# In[31]:


list(labelencoder_y.classes_)


# In[32]:


#########################
#### Train & test #######
########################
print("Split train set & test set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[33]:


##########################################
#### MNB -     TF BINAIRE  ###############
###########################################


# In[34]:


n=1

vectorizer = CountVectorizer(binary=True,ngram_range=(n, n) )
print("CountVectorizer : Done ...")


# In[35]:


vectorizer_TFIDF = TfidfVectorizer()
print("TfIdf Vectorizer : Done ...")


# In[36]:


print("MNB_Tf model ")
model_MNB = make_pipeline(vectorizer, MultinomialNB())


# In[38]:


model_MNB.fit(X_train, y_train)
labels = model_MNB.predict(X_test)


# In[39]:


### Score MNB TF binaire ########
#################################

print()
print("Testing Accuracy for MNB:  {:.4f}".format(model_MNB.score(X_test,y_test)))
print("===========================================================================")


# In[84]:




mat = confusion_matrix(y_test, labels)
print('MNB confusion matrix')
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('MNB confusion matrix')
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[41]:


##########################################
#### MNB -     TF IDF  ##################
###########################################


# In[42]:


model_MNB_TFIDF = make_pipeline(vectorizer_TFIDF, MultinomialNB())
print("MNB_TfIDF model ")


# In[43]:


model_MNB_TFIDF.fit(X_train, y_train)
labels_MNB_TFIDF = model_MNB_TFIDF.predict(X_test)


# In[46]:


### Score MNB TF IDF ########
#################################

print()
print("Testing Accuracy for MNB_TfIdf:  {:.4f}".format(model_MNB_TFIDF.score(X_test,y_test)))
print("===========================================================================")


# In[83]:





mat = confusion_matrix(y_test, labels_MNB_TFIDF)
print('MNB TfIdf confusion matrix')
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True)
plt.title('MNB TfIdf confusion matrix')
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[48]:


########################################
#### LOGISTICREGRESSION - TFBinaire  ###
#######################################


# In[49]:


model_LRegression = make_pipeline(vectorizer, LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'))
print("LR  model ")


# In[50]:


model_LRegression.fit(X_train, y_train)
labels_LRegression = model_LRegression.predict(X_test)


# In[52]:


### Score Regression TF Binaire ########
#################################


print()
print("Testing Accuracy for Logisitc Regression:  {:.4f}".format(model_LRegression.score(X_test,y_test)))
print("===========================================================================")
      


# In[82]:




mat = confusion_matrix(y_test, labels_LRegression)
print('Regression logistic confusion matrix')
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('Regression logistic confusion matrix')
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[54]:


########################################
#### LOGISTICREGRESSION - TF IDF ###
#######################################


# In[55]:


model_LRegression_TFIDF = make_pipeline(vectorizer_TFIDF,LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'))
print("LR_TFIDF model ")


# In[56]:


model_LRegression_TFIDF.fit(X_train, y_train)
labels_LRegression_TFIDF = model_LRegression_TFIDF.predict(X_test)


# In[58]:


###SCORE REGRESSION - TFIDF ###############
#############################################

print()
print("Testing Accuracy for Logisitc Regression TfIdr:  {:.4f}".format(model_LRegression.score(X_test,y_test)))
print("===========================================================================")


# In[81]:


###########################
## Matrice de confustion ##
###########################



mat = confusion_matrix(y_test, labels_LRegression_TFIDF)
print('Regression logistic TFidf confusion matrix')
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.title('Regression logistic TFidf confusion matrix')
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[60]:


##############################################
#### Sauvegarde des models pour réutilsation ##
############################################


# In[61]:


#Mnb_Tf
joblib.dump(model_MNB, "models/mnb_tf.joblib")
#Mnb_tf_Idf
joblib.dump(model_MNB_TFIDF, "models/mnb_tf_Idf.joblib")


# In[62]:


print("Sauvegarde des models")
#Reg_tf
joblib.dump(model_LRegression, "models/reg_tf.joblib")
#Reg_tf_Idf
joblib.dump(model_LRegression_TFIDF, "models/reg_tf_Idf.joblib")


# In[63]:


############################################
### Chargment des models si réutilisation ##
############################################


# In[64]:


#test avec mnb tf_binaire 

mnb_tf = joblib.load("models/mnb_tf.joblib")


# In[65]:


#tet sur le test déja utilisé 
y_pred = mnb_tf.predict(X_test)


# In[66]:


#################################################
## Verification du score pour le model chargé ##
###############################################
mnb_tf.score(X_test,y_test)


# In[93]:


##############################################
### Verification de la matrice de confusion ## 
#############################################
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[68]:


df_test = pd.DataFrame({"contenu_clean":X_test,"type":y_test})


# In[69]:


df_test.head()


# In[70]:


df_test.shape


# In[71]:


df_test["prediction"] = y_pred


# In[72]:


df_test.drop_duplicates("contenu_clean",inplace=True)


# In[73]:


df_test.shape


# In[74]:


df_test = pd.merge(data,df_test,on="contenu_clean",how='inner').drop_duplicates("contenu_clean")


# In[75]:


df_test.shape


# In[88]:


df_test.head()


# In[77]:


df_test["prediction_classe"] = df_test["prediction"].apply(lambda x : labelencoder_y.classes_[x])


# In[92]:


print("Apérçu de la prédiction ")
print(df_test.head(15).to_string())


# In[ ]:





# In[ ]:




