#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train['toxic'][6]


# In[4]:




lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()


# In[5]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()


# In[6]:




COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


# In[7]:


import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[8]:


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


# In[9]:


def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[10]:




x = trn_term_doc
test_x = test_term_doc


# In[11]:


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# In[12]:




preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


# In[13]:


subm = pd.read_csv('sample_submission.csv')


# In[14]:


submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)


# In[16]:


list_sentences_train = train["comment_text"]


# In[17]:


print(list_sentences_train)


# In[18]:


from keras.preprocessing.text import Tokenizer


# In[19]:


max_features = 200
tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(list(list_sentences_train[1]))


# In[20]:


list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)


# In[21]:


list_tokenized_train


# In[22]:


from keras.preprocessing.sequence import pad_sequences
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)


# In[23]:


for i in X_t:
    print(i)


# In[24]:


totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]


# In[25]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[26]:


plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()


# In[27]:




inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier


# In[28]:


embed_size = 128
x = Embedding(max_features, embed_size)(inp)


# In[29]:


x = LSTM(60, return_sequences=True,name='lstm_layer')(x)


# In[30]:


x = GlobalMaxPool1D()(x)


# In[31]:


x = Dropout(0.1)(x)


# In[32]:


x = Dense(50, activation="relu")(x)


# In[33]:


x = Dropout(0.1)(x)


# In[34]:


x = Dense(6, activation="sigmoid")(x)


# In[35]:


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# In[36]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values


# In[37]:


batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[38]:


model.summary()


# In[39]:


from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([X_t[:1]])[0]
layer_output.shape
#print layer_output to see the actual data


# In[40]:


model.save('my_model_01.hdf5')


# In[41]:


from keras.models import load_model


# In[42]:


model = load_model('my_model_01.hdf5')


# In[51]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()

index_list = tk.texts_to_sequences("kill all niggas in the city")
x_train = pad_sequences(index_list, maxlen=200)


# In[52]:


y_pred = model.predict(x_train)


# In[53]:


print(y_pred)


# In[46]:


from sklearn.metrics import accuracy_score


# In[47]:


print(model.summary())

