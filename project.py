from flask import Flask
from flask import request, jsonify
from flask import render_template
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import MaxPooling1D
from keras.layers import Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, BatchNormalization, Input, SpatialDropout1D
from keras.layers import Dense, Bidirectional, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
import keras.backend as K
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input
from numpy import asarray
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

x=[]
f = open('tok.txt', 'r')
x = f.readlines()
f.close()


config = tf.ConfigProto( intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(config=config)
graph = tf.get_default_graph()
t  = Tokenizer(num_words=5000)
t.fit_on_texts(x)
import nltk
from nltk.tokenize import word_tokenize


app = Flask(__name__)

set_session(sess)
model_dwn = load_model('my_model_01.hdf5')
#model_dwn = load_model('unbiased_model.hdf5',custom_objects={'SeqSelfAttention': SeqSelfAttention})
model_dwn._make_predict_function()



@app.route('/')
def project():
    return render_template('index.html')
    
    #return "bla.."
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    stop_words = ['a', 'and']
    input_string = request.form['comment']
    text_tokens = word_tokenize(input_string)
    #int_features = [int(x) for x in request.form.values()]
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    # final_features = [np.array(int_features)]
    filtered_sentence = (" ").join(tokens_without_sw)
    encoded_sample = t.texts_to_sequences([filtered_sentence])
# defining a max size for padding.
    max_len = 200
# padding the vectors of each datapoint to fixed length of 600.
    pad_sample = pad_sequences(encoded_sample,maxlen = max_len,padding='post')

    global sess
    global graph
    with graph.as_default():
    	set_session(sess)
    	results = model_dwn.predict(pad_sample)
    	results = sum(sum(results))
    	# return str(results)

    return render_template('index.html', prediction_text='Toxic {}'.format(results), p='Toxic {}'.format(filtered_sentence))


