import pandas as pd
import re
import Sastrawi
import numpy as np
import pickle
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load sentimen model
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\svm_model_sentimen.pkl', 'rb') as f:
    svm_sentimen = pickle.load(f)

# Load aspek model
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\svm_model_aspek.pkl', 'rb') as f:
    svm_aspek = pickle.load(f)

# Load dataset
df = pd.read_csv("D:\Kerjaan\Magang\Simpatika_Sentimen\datates_simpatika.csv")

# Remove rows where 'content' is missing (NaN values)
df = df.dropna(subset=['content'])

# Normalization
# Dictionary for text normalization (handling slang and misspellings)
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
    " aplkasi ": " aplikasi ", " apalikasi ": " aplikasi ", " apliasi ": " aplikasi ",
    " baguss ": " bagus ", " baguuss ": " bagus ", " bgs ": " bagus ",
    " mantabb ": " mantap ", " mantul ": " mantap ", " mantep ": " mantap ",
    " sgtt ": " sangat ", " sngt ": " sangat ", " sangaat ": " sangat ",

    " tdk ": " tidak ", " gpp ": " tidak apa-apa ", " trs ": " terus ",
    " blm ": " belum ", " udh ": " sudah ", " bgt ": " banget ", " bgt": " banget",
    " gak ": " tidak ", " ga ": " tidak ", " gabisa ": " tidak bisa ", " gaboleh ": " tidak boleh ",

    " log ": " login ", " server down ": " server tidak tersedia ",
    " nyesel ": " menyesal ",

    " best ": " terbaik ", " vibes ": " suasana ", " stylish ": " bergaya ",
    " worth ": " layak ", " simple ": " mudah ", " early ": " awal ",

    " kudu ": " harus ", " hrs ": " harus ", " jk ": " jika ", " dg ": " dengan ",
    " jd ": " jadi ", " problem ": " masalah ", " iru ": " itu ",
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

def normalisasi(str_text):
  for i in norm:
    str_text = str_text.replace(i, norm[i])
  return str_text

df['content'] = df['content'].apply(lambda x: normalisasi(x))

# Initialize stopword remover from Sastrawi
stop_words = StopWordRemoverFactory().get_stop_words()
stopword_dict = ArrayDictionary(stop_words)
stopword_remover = StopWordRemover(stopword_dict)

# Function to remove stopwords from text
def remove_stopwords(text):
    return stopword_remover.remove(text)

# Apply stopword removal
df['content'] = df['content'].apply(remove_stopwords)

# Tokenization: Split text into words
tokenized = df['content'].apply(lambda x:x.split())

# Function for stemming Indonesian words using Sastrawi
def stemming(text_cleaning):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  do = []
  for w in text_cleaning:
    if w.lower() == "semoga":
      dt = w
    else:
      dt = stemmer.stem(w)
    do.append(dt)
  d_clean = []
  d_clean = " ".join(do)
  # print(d_clean)
  return d_clean

tokenized = tokenized.apply(stemming)

X_new = tokenized

# Load vectorizer aspek
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\tfidf_vectorizer_aspek.pkl', 'rb') as f:
    vectorizer_aspek = pickle.load(f)
    
X_aspek = vectorizer_aspek.transform(X_new)

X_aspek = X_aspek.toarray()

# Load vectorizer sentimen
with open(r'D:\Kerjaan\Magang\Simpatika_Sentimen\tfidf_vectorizer_new.pkl', 'rb') as f:
    vectorizer_sentimen = pickle.load(f)

X_sentimen = vectorizer_sentimen.transform(X_new)

X_sentimen = X_sentimen.toarray()

# Predict aspek  
y_pred_aspek= svm_aspek.predict(X_aspek)

# Predict sentimen
y_pred_sentimen = svm_sentimen.predict(X_sentimen)



# Decoding aspek
aspek_mapping = {
    'Autentikasi': 0,
    'Fitur': 1,
    'Iklan': 2,
    'Informasi': 3,
    'Kepraktisan': 4,
    'Performa': 5,
    'Umum': 6
}

inverse_aspek = {v: k for k, v in aspek_mapping.items()}

y_aspek_decoded = pd.DataFrame({'aspek': pd.Series(y_pred_aspek).map(inverse_aspek)})

# Decoding sentimen
sentimen_mapping = {
    'Negatif': 0,
    'Netral': 1,
    'Positif': 2
}

inverse_sentimen = {v: k for k, v in sentimen_mapping.items()}

y_sentimen_decoded = pd.DataFrame({'sentimen': pd.Series(y_pred_sentimen).map(inverse_sentimen)})

# Merge dataframe
df_result = pd.concat([df, y_aspek_decoded, y_sentimen_decoded], axis=1)

print(df_result)

# Save to csv
# df_result.to_csv('D:/Kerjaan/Magang/Simpatika_Sentimen/result_sentimen.csv', index=False, encoding='utf-8')
