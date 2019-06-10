#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix


# In[2]:


####################################
### Récupération data deja traité #
##################################
print("CNN TEXT CLASSIFIER")


# In[3]:


data = pd.read_csv('data_news.csv')
print("Load Data : Done")


# In[4]:


###########################
#stem & remove stopwords ##
###########################
print("stem & remove stopwords")
data["contenu_clean"] = data["contenu"].apply(lambda x : ' '.join([stemmer.stem(w) for w in str(x).split() if w not in set(stopwords.words('english'))]))


# In[ ]:


data.head()


# In[ ]:


print("count values per classes : ")
print(data.classe.value_counts())


# In[ ]:


#####################################
## Séparation des données    ########
#####################################
text = data.iloc[:,3].values
y = data.iloc[:,2].values


# In[ ]:


set(y)


# In[ ]:


#############################
###  Transformation des classe 
##############################

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[ ]:


set(y)


# In[ ]:


from keras.utils import to_categorical
y = to_categorical(y, num_classes = 5)
print("Classes to cetegorical : Done ")


# In[ ]:


list(labelencoder_y.classes_)


# In[ ]:


max_words = 5000


# In[ ]:


#############################################
### Tokennizer 
###########################################
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_words) # Setup tokenizer
tokenizer.fit_on_texts(text)

X = tokenizer.texts_to_sequences(text)

vocab_size = len(tokenizer.word_index) + 1
print("Tokenizer : Done ")


# In[ ]:


#################################
### Quelques tests et affichage #
######################## #######


# In[ ]:


print("max sequence length:", max(len(s) for s in X))
print("min sequence length:", min(len(s) for s in X))


# In[ ]:


word_index = tokenizer.word_index
print('Found {:,} unique words.'.format(len(word_index)))


# In[ ]:


for word in ['the', 'all', 'happy', 'sad']: 
    print('{}: {}'.format(word, tokenizer.word_index[word]))


# In[ ]:





# In[ ]:


#########################
#### Train & test #######
########################
print("Split train set & test set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, padding='post', maxlen=max_words)
X_test = pad_sequences(X_test, padding='post', maxlen=max_words)


# In[ ]:


print(X_train.shape)


# In[ ]:


import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, LSTM , Flatten,Embedding,Masking,Conv1D,GlobalMaxPooling1D,Input,MaxPooling1D,GlobalMaxPooling1D


# In[ ]:


print("Bulding the CNN ")


# In[ ]:


print("====================================================================")


# In[ ]:


#Initialisation 
Classifier = Sequential()


# In[ ]:


# Ajouter layer Embedding
Classifier.add(Embedding(vocab_size, 50))


# In[ ]:


# Ajout de la premiere couche de convulution 
Classifier.add(Conv1D(64, 3, activation='relu'))


# In[ ]:


# MaxPooling pour éviter le surapprentisage
Classifier.add(MaxPooling1D(3))


# In[ ]:


Classifier.add(GlobalMaxPooling1D())


# In[ ]:


#ajout des layers 
Classifier.add(Dense(64, activation='relu'))


# In[ ]:


#Eviter le surapprentissage
Classifier.add(Dropout(0.3))


# In[ ]:


#Couche de sortie 
Classifier.add(Dense(5, activation='softmax'))


# In[ ]:


print(Classifier.summary())


# In[ ]:


#############################
# Compilation du model ######
#############################

#Classifier.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
Classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


print()


# In[ ]:


print("TRAIN : ")


# In[ ]:


######################
##### TRAIN ##########
##################


Classifier.fit(X_train, y_train,epochs=10,batch_size = 100,validation_data=(X_test, y_test))


# In[ ]:


print("====================================================================")


# In[ ]:


###########################
###  Affichage des scores #

###########################
loss, accuracy = Classifier.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[ ]:


#############################
## Prediction sur le test ###
#############################""
y_pred = Classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


# In[ ]:


print("CNN confusion matrix ")
print(matrix)
sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)
plt.title("CNN confusion matrix")
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[ ]:





# In[ ]:


##############################################
#### Sauvegarde des models pour réutilsation ##
############################################
print("Sauvegarde des models")


# In[ ]:


joblib.dump(Classifier, "models/CNN_txt_Classification.joblib")


# In[ ]:


joblib.dump(tokenizer,"models/tokenizer_CNN_text.joblib")
joblib.dump(labelencoder_y,"models/labelencoder_CNN_text.joblib")


# In[ ]:


#####################################
### Chargement du model souvegardé ##
#####################################


# In[ ]:


Cnn_classifier = joblib.load("models/CNN_txt_Classification.joblib")


# In[ ]:




