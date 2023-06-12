import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

st.title("UAS PENAMBANGAN DATA - APLIKASI PREDIKSI BATU GINJAL BERDASARKAN ANALISIS URIN")

dataset, preprocessing, modeling, implementation, profile = st.tabs(
    ["Data ", "Preprocessing Data", "Modeling", "Implementation", "Profile"])

with profile:
    st.write("##### Nama  : Jennatul Macwe ")
    st.write("##### Nim   : 210411100151 ")
    st.write("##### Kelas : Penambangan Data B ")
    st.write("##### E-mail : jennatmc@gmail.com ")

# with description:


with dataset:
    st.write("""# Dataset """)
    df = pd.read_csv(
        'https://raw.githubusercontent.com/jennamacwe/UAS-PENDATA/main/Dataset%20-%20Kidney%20Stone%20Prediction.csv')
    st.dataframe(df)

    st.write("""# Deskripsi Dataset """)
    st.write("###### Dataset yang digunakan Adalah : ")
    st.write("###### Kidney Stone Prediction based on Urine Analysis (Prediksi Batu Ginjal Berdasarkan Analisis Urin) ")
    st.write("###### Sumber Dataset : https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis")
    st.write(" Batu ginjal adalah massa keras yang terbentuk di dalam ginjal atau saluran kemih. Batu ginjal terbentuk ketika zat-zat seperti kalsium, oksalat, asam urat, atau kalsium fosfat mengendap dan membentuk kristal di dalam ginjal. Kristal-kristal ini kemudian dapat bergabung dan membentuk batu yang lebih besar. Batu ginjal dapat terbentuk di salah satu atau kedua ginjal, dan kemudian dapat bergerak melalui saluran kemih menuju kandung kemih. Batu ginjal yang lebih kecil dapat keluar dari tubuh secara alami melalui urin tanpa menimbulkan gejala yang signifikan. Namun, batu ginjal yang lebih besar atau yang terjebak di dalam saluran kemih dapat menyebabkan gejala nyeri yang parah dan memerlukan pengobatan medis.  ")
    st.write("""# Deskripsi Data""")
    st.write("Ttipe data: Numerik ")
    st.write(" Total Data dari dataset sebanyak 79 Data")
    st.write("Informasi Atribut")
    st.write("1) gravity : berat jenis atau densitas urin ")
    st.write("2) ph : logaritma negatif dari ion hidrogen ")
    st.write("3) osmo : osmolaritas (mOsm), satuan yang digunakan dalam biologi dan kedokteran tetapi tidak dalam kimia fisik. Osmolaritas sebanding dengan konsentrasi molekul dalam larutan ")
    st.write("4) cond : konduktivitas (mMho miliMho). Satu Mho adalah satu timbal balik Ohm. Konduktivitas sebanding dengan konsentrasi muatan ion dalam larutan ")
    st.write("5) urea : konsentrasi urea dalam milimol per liter ")
    st.write("6) calc : kalsium konsentrasi (CALC) dalam milimol-liter ")
    st.write("7) target : penentuan termasuk memiliki Batu Ginjal atau tidak ")
    st.write("""          0 = Tidak Adanya Batu Ginjal""")
    st.write("""          1 = Adanya Batu Ginjal""")


with preprocessing:

    st.subheader("""(Min Max Scalar)""")
    st.write("""Normalisasi Data :""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('rumus_normalisasi.png', use_column_width=False, width=250)

    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)

    # Mendefinisikan Varible X dan Y
    X = df.drop(columns=['target'])
    y = df['target'].values
    df
    X
    df_min = X.min()
    df_max = X.max()

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.target).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]]
    })

    st.write(labels)

with modeling:
    # Nilai X training dan Nilai X testing
    training, test = train_test_split(
        scaled_features, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(
        y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor (K-NN)')
        destree = st.checkbox('Decission Tree (Pohon Keputusan)')
        mlp_model = st.checkbox('MLP')

        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Naive Bayes Classification
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))

        # KNN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))

        # ANNBP
        # Menggunakan 2 layer tersembunyi dengan 100 neuron masing-masing
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_predict))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(
                    gaussian_akurasi))
            if k_nn:
                st.write(
                    "Model K-Nearest Neighboor (KNN) accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree:
                st.write(
                    "Model Decision Tree (Pohon Keputusan) accuracy score : {0:0.2f}" . format(dt_akurasi))
            if mlp_model:
                st.write(
                    'Model MLP accuracy score: {0:0.2f}'.format(mlp_accuracy))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [gaussian_akurasi, knn_akurasi, dt_akurasi, mlp_accuracy],
                'Model': ['Naive Bayes', 'K-NN', 'Decission Tree', 'ANNBP'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)


with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        gravity = st.number_input('Masukkan Berat Jenis Urin : ')
        ph = st.number_input('Masukkan nilai Ph : ')
        osmo = st.number_input('Masukkan nilai Enzim Osmolaritas Urine : ')
        cond = st.number_input('Masukkan nilai Konduktivitas Urine : ')
        urea = st.number_input(
            'Masukkan Nilai Konsentrasi Ureum dalam Urin : ')
        calc = st.number_input(
            'Masukkan nilai Konsentrasi Kalsium dalam Urin : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi dibawah ini:',
                             ('Naive Bayes', 'K-NN', 'Decision Tree', 'MLP'))

        prediksi = st.form_submit_button("Submit")

        if prediksi:
            inputs = np.array([
                gravity,
                ph,
                osmo,
                cond,
                urea,
                calc,
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn
            if model == 'Decision Tree':
                mod = dt
            if model == 'ANNBackpropaganation':
                mod = mlp

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
            ada = 1
            tidak_ada = 0
            if input_pred == ada:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'ditemukan bahwa adanya batu ginjal')
            else:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'ditemukan bahwa Tidak adanya batu ginjal')
