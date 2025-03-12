import re
from deep_translator import GoogleTranslator
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import streamlit as st
import requests
import pandas as pd
import time
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu


# Load tokenizer aspek
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\CNN_Model_Analisis\front-end\tokenizer_aspek.pkl', 'rb') as handle:
    tokenizer_aspek = pickle.load(handle)
    
# Load tokenizer sentimen
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\CNN_Model_Analisis\front-end\tokenizer_sentimen.pkl', 'rb') as handle:
    tokenizer_sentimen = pickle.load(handle)

# Load max_length aspek
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\CNN_Model_Analisis\front-end\max_length_aspek.pkl', 'rb') as f:
    max_length_aspek = pickle.load(f)
        
# Load max_length sentimen
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\CNN_Model_Analisis\front-end\max_length_sentimen.pkl', 'rb') as f:
    max_length_sentimen = pickle.load(f)

# Load model aspek
model_aspek = load_model(r'D:\Kerjaan\Magang\Simpatika_Sentimen\CNN_Model_Analisis\front-end\cnn_aspek_model.h5')

# Load model sentimen
model_sentimen = load_model(r'D:\Kerjaan\Magang\Simpatika_Sentimen\CNN_Model_Analisis\front-end\cnn_sentiment_model.h5')

# Normalisasi
norm = {
    " yg " : " yang ",
    " bgt " : " banget ",
    " bgt" : " banget",
    " bangat " : " banget ",
    " pisan " : " banget ",
    " trimakasih " : " terima kasih ",
    "terimakasih" : " terima kasih",
    " terimakasih" : " terima kasih",
    " terimakasih " : " terima kasih ",
    " kereen " : " keren ",
    "alhamdulillaah " : " Alhamdulillah ",
    " setinggitingginya " : " tinggi ",
    " app " : " aplikasi ",
    " apk " : " aplikasi ",
    " apl " : " aplikasi ",
    " nusantara" : " Indonesia",
    " he " : " ",
    " full " : " penuh ",
    "mudahmudahan " : " semoga ",
    " dgn " : " dengan ",
    "dg " : "dengan ",
    " tertakit " : " terkait ",
    " pengapdian " : " pengabdian ",
    "mantab " : "mantap ",
    " mantab " : " mantap ",
    " mantab" : " mantap",
    " mantaaf" : " mantap",
    " mantaaaaap" : " mantap ",
    "mantul" : "mantap",
    "mantaaab" : "mantap",
    "mantap jiwa" : "mantap",
    " ppg" : " ppg ",
    " tdk " : " tidak ",
    " sy " : " saya ",
    "tolong " : " tolong ",
    " berbeling belit " : " berbelit ",
    " guruguru " : " guru ",
    " ijasah " : " ijazah ",
    " terhusus " : " terkhusus ",
    " dn " : " dan ",
    "sngat " : "sangat ",
    " fahammmm" : " paham",
    " oke" : " baik",
    " blm " : " belum ",
    " lg" : " lagi",
    " sip " : " bagus ",
    " asyik " : " seru ",
    " fun " : " seru ",
    " best " : " terbaik ",
    "good" : "bagus",
    "bagussss" : "bagus",
    "bagussss " : "bagus ",
    " bagussss" : " bagus",
    " bagussss " : " bagus ",
    "jos" : "bagus",
    "like" : "suka",
    "informatig" : "informatif",
    "siiiiip" : "bagus",
    "good job" : "bagus",
    "nice" : "bagus",
    "jozz" : "bagus",
    "msh " : "masih ",
    "easy " : "mudah ",
    " parmudahkan " : " mudahkan ",
    " jg " : " juga ",
    " km " : " kami ",
    "smg " : "semoga ",
    " kl " : " kalau ",
    " lg " : " lagi ",
    " jd " : " jadi ",
    " dpt " : " dapat ",
    "sngt " : "sangat ",
    " sngt " : " sangat ",
    " se x" : " sekali",
    " mmbntu " : " membantu ",
    " memudhkn " : " memudahkan ",
    "praktis" : " praktis",
    " yg": " yang",
    " ok": " baik",
    " oke": " baik",
    " banget": " sangat",
    " aplikasinya": " aplikasi",
    " sip": " baik",
    " tdk": " tidak",
    " gak": " tidak",
    " bgt": " sangat",
    " trs": " terus",
    " dr": " dari",
    " krn": " karena",
    " sdh": " sudah",
    " blm": " belum",
    " tp": " tapi",
    " sy": " saya",
    " sbg": " sebagai",
    " utk": " untuk",
    " dlm": " dalam",
    " sj": " saja",
    " sm": " sama",
    " pd": " pada",
    " jd": " jadi",
    " mnrt": " menurut",
    " trmksh": " terima kasih",
    " mksh": " terima kasih",
    " kereen": " keren",
    "huebat": " hebat",
    "siippp": "sip",
    " trimakasih": " terima kasih",
    " aplkasi ": " aplikasi ",
    " apalikasi ": " aplikasi ",
    " apliasi ": " aplikasi ",
    " baguss ": " bagus ",
    " baguuss ": " bagus ",
    " bgs ": " bagus ",
    " mantabb ": " mantap ",
    " mantul ": " mantap ",
    " mantep ": " mantap ",
    " sgtt ": " sangat ",
    " sngt ": " sangat ",
    " sangaat ": " sangat ",

    " tdk ": " tidak ",
    " gpp ": " tidak apa-apa ",
    " trs ": " terus ",
    " blm ": " belum ",
    " udh ": " sudah ",
    " bgt ": " banget ",
    " bgt": " banget",
    " gak ": " tidak ",
    " ga ": " tidak ",
    " gabisa ": " tidak bisa ",
    " gaboleh ": " tidak boleh ",

    " log ": " login ",
    " server down ": " server tidak tersedia ",
    " nyesel ": " menyesal ",

    " best ": " terbaik ",
    " vibes ": " suasana ",
    " stylish ": " bergaya ",
    " worth ": " layak ",
    " simple ": " mudah ",
    " early ": " awal ",

    " kudu ": " harus ",
    " hrs ": " harus ",
    " jk ": " jika ",
    " dg ": " dengan ",
    " jd ": " jadi ",
    " problem ": " masalah ",
    " iru ": " itu ",
    " kereen": " keren",
    "huebat": " hebat",
    "siippp": "sip",
    " mendanlut": " mendownload",
    " logen": " login",
    " lht": " lihat",
    " yng": " yang",
    " laah": " lah",
    " tidakmpang": " gampang",
    "siip": "sip",
    " donlod": " download",
    " bs ": " bisa ",
    "gak ": " tidak ",
    "gkmana ": " bagaimana ",
    " betmanfaat ": " bermanfaat ",
    " dlm ": " dalam ",
    " aplikasix ": " aplikasinya ",
    " hax ": " hanya ",
    " gk ": " tidak ",
    " log in": " login ",
    " sllu ": " selalu ",
    " ggal ": " gagal ",
    " daribrowser ": " dari browser ",
    "error": "error ",
    "entah ": "tidak tau ",
    " lelet ": " lambat ",
    " dn ": " dan ",
    "kaga ": "tidak ",
    "eror ": "error ",
    " erorre ": " error ",
    " nggak ": " tidak ",
    " donlot ": " download ",
    " males ": " malas ",
    " tdk ": " tidak ",
    " gak ": " tidak ",
    " pasword": " password ",
    " dk ": " tidak ",
    " lgsung ": " langsung ",
    "knpa ": " kenapa ",
    " bisaaa ": " bisa ",
    "updet": "update",
    "login": "login ",
    "trims ": "terima ",
    " blm ": " belum ",
    " sklh ": " sekolah ",
    " tidk ": " tidak ",
    "singkron": "sinkron",
    "thankss": "terima kasih",
    "ngak": "tidak",
    "danlut": "download",
    "thanks": " terima kasih",
    " kpn ": " kapan ",
    "trimksh": "terima kasih",
    "ptkterimakasih": "ptk terima kasih",
    "mantaaap": "mantap",
    "baguuss": "bagus",
    "trima ksh": "terima kasih",
    "thanks": "terima kasih",
    "sipppppp": "sip",
    "mantaaaap": "mantap",
    "siiiippppp": "sip",
    "hebet": "hebat",
    "sipzzzz": "sip",
    "mantappp": "mantap",
    "mantaaap": "mantap",
    "kereeeeen": "keren",
    "mantaaabbbb": "mantab",
    "siiiip": "sip",
    "siiipppp": "sip",
    "bagusssss": "bagus",
}

def normalisasi(text):
  for i in norm:
    text = text.replace(i, norm[i])
  return text

# Translate to English
def translate_text(text):
    return GoogleTranslator(source='id', target='en').translate(text)

# Function to preprocess input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function to predict sentiment
def predict_aspek(text):
    text = normalisasi(text)
    text = clean_text(text)
    text = translate_text(text)
    sequence = tokenizer_aspek.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length_aspek, padding='post')
    prediction = model_aspek.predict(padded_sequence)
    label_map = {0: 'Lainnya', 1: 'Kepraktisan', 2: 'Fitur', 3: 'Autentikasi'}
    return label_map[np.argmax(prediction)]

# Function to predict sentiment
def predict_sentiment(text):
    text = normalisasi(text)
    text = clean_text(text)
    text = translate_text(text)
    sequence = tokenizer_sentimen.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length_sentimen, padding='post')
    prediction = model_sentimen.predict(padded_sequence)
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label_map[np.argmax(prediction)]

# # Example usage
# sample_text = "Aplikasi susah login!"
# print("Predicted Sentiment:", predict_sentiment(sample_text))
# print("Predicted Aspek:", predict_aspek(sample_text))

# # Streamlit UI
# st.title("Sentiment and Aspect Prediction App")

# # URL Metabase yang ingin ditampilkan
# metabase_url = "https://www.youtube.com/"

# # Menampilkan Metabase dalam iframe
# st.components.v1.iframe(metabase_url, width=800, height=600, scrolling=True)

# st.write("Masukkan teks untuk menganalisis sentimen dan aspek.")

# # # Input text box for user
# input_text = st.text_area("Masukkan Teks", "Aplikasi susah login!")

# # Button to get predictions
# if st.button('Prediksi'):
#     # Sentiment Prediction
#     sentiment = predict_sentiment(input_text)
#     # Aspect Prediction
#     aspect = predict_aspek(input_text)
    
#     # Display results
#     st.write(f"**Prediksi Sentimen**: {sentiment}")
#     st.write(f"**Prediksi Aspek**: {aspect}")


# Button to get predictions
# API URL
# API_URL = "http://127.0.0.1:8000"

# # Button to get predictions
# if st.button('Prediksi'):
#     # Sentiment Prediction
#     sentiment = predict_sentiment(input_text)
#     # Aspect Prediction
#     aspect = predict_aspek(input_text)

#     # Send request to FastAPI for summarization
#     response = requests.post(f"{API_URL}/resume/", json={"text": input_text})

#     if response.status_code == 200:
#         summary = response.json().get("summary", "Ringkasan tidak tersedia")
#     else:
#         summary = "Gagal mendapatkan ringkasan"

#     # Display results
#     st.write(f"**Prediksi Sentimen**: {sentiment}")
#     st.write(f"**Prediksi Aspek**: {aspect}")
#     st.write(f"**Ringkasan Narasi**: {summary}")

# df = pd.read_csv("D:\Kerjaan\Magang\Simpatika_Sentimen\datates_simpatika.csv")
# df_ori = df['content'].copy()
# # df['sentimen'] = df['content'].apply(predict_sentiment)
# # df['aspek'] = df['content'].apply(predict_aspek)
# # df_result = pd.concat([df_ori, df['aspek'], df['sentimen']], axis=1)
# # print(df_result)

# try:

#     if 'content' in df.columns:
#         with st.spinner("Sedang menganalisis dataset..."):
#             df["sentimen"] = df['content'].apply(predict_sentiment)
#             df["aspek"] = df['content'].apply(predict_aspek)
#             df_result = pd.concat([df_ori, df['aspek'], df['sentimen']], axis=1)
            
#             combined_text = " ".join(df_result["content"].astype(str))
             
#             # # Kirim ke API untuk ringkasan
#             # df_result["Ringkasan"] = df_result['content'].apply(
#             #     lambda x: requests.post(f"{API_URL}/resume/", json={"text": x}).json().get("summary", "Ringkasan tidak tersedia")
#             # )
            
#             response = requests.post(f"{API_URL}/resume/", json={"text": combined_text})

#             if response.status_code == 200:
#                 summary = response.json().get("summary", "Ringkasan tidak tersedia")
#             else:
#                 summary = "Gagal mendapatkan ringkasan"
        
#         st.subheader("Hasil Prediksi Dataset")        
#         st.dataframe(df_result)  # Menampilkan DataFrame dalam Streamlit
        
#         st.subheader("Ringkasan Narasi Keseluruhan")
#         st.write(summary)
#     else:
#         st.error("Kolom 'content' tidak ditemukan dalam dataset!")

# except Exception as e:
#     st.error(f"Gagal memuat file: {str(e)}")

# # Pertanyaan biasa
# st.write("Masukkan pertanyaan.")    
# user_input = st.text_area("Masukkan pertanyaan:")
# if st.button("Kirim"):
#     if user_input:
#         response = requests.post("http://127.0.0.1:8000/chat", json={"prompt": user_input})
#         st.write("Model:", response.json()["response"])
#     else:
#         st.warning("Masukkan teks sebelum mengirim.")


# API URL
API_URL = "http://127.0.0.1:8000"


# Inisialisasi session state jika belum ada
if "sentiment" not in st.session_state:
    st.session_state["sentiment"] = None
if "aspect" not in st.session_state:
    st.session_state["aspect"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "df_result" not in st.session_state:
    st.session_state["df_result"] = None
if "dataset_summary" not in st.session_state:
    st.session_state["dataset_summary"] = None
    
    
# # Tampilan 1
# st.set_page_config(page_title="Sentiment & Aspect Analysis", layout="wide")
# st.markdown("""
#     <style>
#         .main-title {
#             text-align: center;
#             font-size: 32px;
#             font-weight: bold;
#             animation: fadeIn 1s;
#         }
#         .sub-title {
#             text-align: center;
#             font-size: 24px;
#             color: #4CAF50;
#             animation: slideIn 1s;
#         }
#         .stButton>button {
#             background-color: #4CAF50;
#             color: white;
#             width: 100%;
#             height: 50px;
#             font-size: 18px;
#         }
#         @keyframes fadeIn {
#             from { opacity: 0; }
#             to { opacity: 1; }
#         }
#         @keyframes slideIn {
#             from { transform: translateY(-10px); opacity: 0; }
#             to { transform: translateY(0); opacity: 1; }
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Navigation Bar
# st.sidebar.title("📌 Menu")
# page = st.sidebar.radio("Pilih Halaman", ["📊 Dashboard", "🔍 Analisis Sentimen & Aspek"])

# # Transisi Halaman
# time.sleep(0.5)

# if page == "📊 Dashboard":
#     st.markdown("<p class='main-title'>📊 Dashboard Sentiment and Aspect Prediction</p>", unsafe_allow_html=True)
#     metabase_url = "https://www.youtube.com/"  # Ganti dengan URL Metabase asli
#     components.iframe(metabase_url, width=1000, height=600, scrolling=True)

# elif page == "🔍 Analisis Sentimen & Aspek":
#     st.markdown("<p class='main-title'>🔍 Sentiment and Aspect Prediction App</p>", unsafe_allow_html=True)
#     st.write("Masukkan teks untuk menganalisis sentimen dan aspek.")
    
#     input_text = st.text_area("Masukkan Teks", "Aplikasi susah login!")
    
#     if st.button('🚀 Prediksi'):
#         with st.spinner("🔄 Menganalisis teks..."):
#             time.sleep(1)
#             st.session_state["sentiment"] = predict_sentiment(input_text)
#             st.session_state["aspect"] = predict_aspek(input_text)
#             response = requests.post(f"{API_URL}/resume/", json={"text": input_text})
#             st.session_state["summary"] = response.json().get("summary", "Ringkasan tidak tersedia") if response.status_code == 200 else "Gagal mendapatkan ringkasan"
    
#     if st.session_state["sentiment"] is not None:
#         st.markdown(f"<p class='sub-title'>🎭 Prediksi Sentimen: `{st.session_state['sentiment']}`</p>", unsafe_allow_html=True)
#         st.markdown(f"<p class='sub-title'>📌 Prediksi Aspek: `{st.session_state['aspect']}`</p>", unsafe_allow_html=True)
#         st.markdown(f"<p class='sub-title'>📜 Ringkasan Narasi: `{st.session_state['summary']}`</p>", unsafe_allow_html=True)
    
#     # Load Dataset
#     try:
#         df = pd.read_csv("D:/Kerjaan/Magang/Simpatika_Sentimen/datates_simpatika.csv")
#         df_ori = df['content'].copy()
        
#         if 'content' in df.columns and st.session_state["df_result"] is None:
#             with st.spinner("⏳ Sedang menganalisis dataset..."):
#                 time.sleep(1)
#                 df["sentimen"] = df['content'].apply(predict_sentiment)
#                 df["aspek"] = df['content'].apply(predict_aspek)
#                 st.session_state["df_result"] = pd.concat([df_ori, df['aspek'], df['sentimen']], axis=1)
#                 combined_text = " ".join(st.session_state["df_result"]["content"].astype(str))
#                 response = requests.post(f"{API_URL}/resume/", json={"text": combined_text})
#                 st.session_state["dataset_summary"] = response.json().get("summary", "Ringkasan tidak tersedia") if response.status_code == 200 else "Gagal mendapatkan ringkasan"
        
#         if st.session_state["df_result"] is not None:
#             st.subheader("📑 Hasil Prediksi Dataset")        
#             st.dataframe(st.session_state["df_result"], height=400)
            
#             st.subheader("📌 Ringkasan Narasi Keseluruhan")
#             st.write(st.session_state["dataset_summary"])
#         else:
#             st.error("❌ Kolom 'content' tidak ditemukan dalam dataset!")
#     except Exception as e:
#         st.error(f"⚠️ Gagal memuat file: {str(e)}")


## Tampilan 2
st.set_page_config(page_title="Sentiment and Aspect Prediction", layout="wide")

# Navbar
selected = option_menu(
    menu_title=None,
    options=["🏠 Dashboard", "🔍 Analisis Sentimen"],
    icons=["house", "bar-chart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "5px", "background-color": "#01230a"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin": "5px"},
        "nav-link-selected": {"background-color": "#2E7630", "color": "white"},
    },
)

def transition_effect():
    with st.spinner("⏳ Memuat halaman..."):
        time.sleep(0.5)

if selected == "🏠 Dashboard":
    transition_effect()
    st.markdown("<h2 style='text-align: center;'>📊 Dashboard Sentiment & Aspect Analysis</h2>", unsafe_allow_html=True)
    metabase_url = "https://www.youtube.com/"  # Ganti dengan URL Metabase asli
    st.markdown(
        f'<iframe src="{metabase_url}" style="width:100%; height:600px; border:none;"></iframe>',
        unsafe_allow_html=True
    )

elif selected == "🔍 Analisis Sentimen":
    transition_effect()
    st.markdown("<h2 style='text-align: center;'>🔍 Sentiment and Aspect Prediction</h2>", unsafe_allow_html=True)
    st.write("Masukkan teks untuk menganalisis sentimen dan aspek.")

    input_text = st.text_area("Masukkan Teks", "Aplikasi susah login!")
    if st.button('🚀 Prediksi'):
        with st.spinner("🔄 Menganalisis teks..."):
            time.sleep(1)
            st.session_state["sentiment"] = predict_sentiment(input_text)
            st.session_state["aspect"] = predict_aspek(input_text)
            response = requests.post(f"{API_URL}/resume/", json={"text": input_text})
            st.session_state["summary"] = response.json().get("summary", "Ringkasan tidak tersedia") if response.status_code == 200 else "Gagal mendapatkan ringkasan"
            st.success("✅ Analisis berhasil!")
    
    if st.session_state["sentiment"] is not None:
        st.markdown(f"<h5 style='color:#4CAF50;'>🎭 Prediksi Sentimen: {st.session_state['sentiment']}</h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color:#2196F3;'>📌 Prediksi Aspek: {st.session_state['aspect']}</h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color:#FF9800;'>📜 Ringkasan Narasi: {st.session_state['summary']}</h5>", unsafe_allow_html=True)
    
    # # Load Dataset
    # try:
    #     df = pd.read_csv("D:/Kerjaan/Magang/Simpatika_Sentimen/datates_simpatika.csv")
    #     df_ori = df['content'].copy()
        
    #     if 'content' in df.columns and st.session_state["df_result"] is None:
    #         with st.spinner("⏳ Sedang menganalisis dataset..."):
    #             time.sleep(1)
    #             df["sentimen"] = df['content'].apply(predict_sentiment)
    #             df["aspek"] = df['content'].apply(predict_aspek)
    #             st.session_state["df_result"] = pd.concat([df_ori, df['aspek'], df['sentimen']], axis=1)
    #             combined_text = " ".join(st.session_state["df_result"]["content"].astype(str))
    #             response = requests.post(f"{API_URL}/resume/", json={"text": combined_text})
    #             st.session_state["dataset_summary"] = response.json().get("summary", "Ringkasan tidak tersedia") if response.status_code == 200 else "Gagal mendapatkan ringkasan"
        
    #     if st.session_state["df_result"] is not None:
    #         st.subheader("📑 Hasil Prediksi Dataset")        
    #         st.dataframe(st.session_state["df_result"], height=400)
    #         st.subheader("📌 Ringkasan Narasi Keseluruhan")
    #         st.write(st.session_state["dataset_summary"])
    #     else:
    #         st.error("❌ Kolom 'content' tidak ditemukan dalam dataset!")
    # except Exception as e:
    #     st.error(f"⚠️ Gagal memuat file: {str(e)}")