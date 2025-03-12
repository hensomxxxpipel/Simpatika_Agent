import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from load_model import tokenizer_aspek, tokenizer_sentimen, max_length_aspek, max_length_sentimen, model_aspek, model_sentimen
from preprocessing import normalisasi, clean_text
import requests

# Function to predict sentiment
def predict_aspek(text):
    text = normalisasi(text)
    text = clean_text(text)
    sequence = tokenizer_aspek.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length_aspek, padding='post')
    prediction = model_aspek.predict(padded_sequence)
    label_map = {0: 'Lainnya', 1: 'Kepraktisan', 2: 'Fitur', 3: 'Autentikasi'}
    return label_map[np.argmax(prediction)]

# Function to predict sentiment
def predict_sentiment(text):
    text = normalisasi(text)
    text = clean_text(text)
    sequence = tokenizer_sentimen.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length_sentimen, padding='post')
    prediction = model_sentimen.predict(padded_sequence)
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label_map[np.argmax(prediction)]

API_URL = "http://127.0.0.1:8000"

df = pd.read_csv("D:\Kerjaan\Magang\Simpatika_Sentimen\datates_simpatika.csv")
df_ori = df['content'].copy()
# df['sentimen'] = df['content'].apply(predict_sentiment)
# df['aspek'] = df['content'].apply(predict_aspek)
# df_result = pd.concat([df_ori, df['aspek'], df['sentimen']], axis=1)
# print(df_result)

try:

    if 'content' in df.columns:
        with st.spinner("Sedang menganalisis dataset..."):
            df["sentimen"] = df['content'].apply(predict_sentiment)
            df["aspek"] = df['content'].apply(predict_aspek)
            df_result = pd.concat([df_ori, df['aspek'], df['sentimen']], axis=1)
            
            combined_text = " ".join(df_result["content"].astype(str))
             
            # # Kirim ke API untuk ringkasan
            # df_result["Ringkasan"] = df_result['content'].apply(
            #     lambda x: requests.post(f"{API_URL}/resume/", json={"text": x}).json().get("summary", "Ringkasan tidak tersedia")
            # )
            
            response = requests.post(f"{API_URL}/resume/", json={"text": combined_text})

            if response.status_code == 200:
                summary = response.json().get("summary", "Ringkasan tidak tersedia")
            else:
                summary = "Gagal mendapatkan ringkasan"
        
        st.subheader("Hasil Prediksi Dataset")        
        st.dataframe(df_result)  # Menampilkan DataFrame dalam Streamlit
        
        st.subheader("Ringkasan Narasi Keseluruhan")
        st.write(summary)
    else:
        st.error("Kolom 'content' tidak ditemukan dalam dataset!")

except Exception as e:
    st.error(f"Gagal memuat file: {str(e)}")