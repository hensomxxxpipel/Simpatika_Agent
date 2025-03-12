import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_tokenizer(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def load_max_length(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_cnn_model(path):
    return load_model(path)

tokenizer_aspek = load_tokenizer(r'D:/Kerjaan/Magang/Simpatika_Sentimen/CNN_Model_Analisis/front-end/tokenizer_aspek.pkl')
tokenizer_sentimen = load_tokenizer(r'D:/Kerjaan/Magang/Simpatika_Sentimen/CNN_Model_Analisis/front-end/tokenizer_sentimen.pkl')

max_length_aspek = load_max_length(r'D:/Kerjaan/Magang/Simpatika_Sentimen/CNN_Model_Analisis/front-end/max_length_aspek.pkl')
max_length_sentimen = load_max_length(r'D:/Kerjaan/Magang/Simpatika_Sentimen/CNN_Model_Analisis/front-end/max_length_sentimen.pkl')

model_aspek = load_cnn_model(r'D:/Kerjaan/Magang/Simpatika_Sentimen/CNN_Model_Analisis/front-end/cnn_aspek_model.h5')
model_sentimen = load_cnn_model(r'D:/Kerjaan/Magang/Simpatika_Sentimen/CNN_Model_Analisis/front-end/cnn_sentiment_model.h5')
