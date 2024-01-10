import streamlit as st
import pickle
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler

st.title("Понижение размерности")

file = st.file_uploader("Загрузите файл для понижение размерности в формате .csv")


def get_prepared(df):
    df["engine_capacity"].fillna(0, inplace=True)

    bin = BinaryEncoder()
    binarized_categorical = bin.fit_transform(df.select_dtypes(include="object")).astype("int8")
    data = df.select_dtypes(exclude="object")
    data = pd.concat([data, pd.DataFrame(binarized_categorical)], axis=1)
    
    temp = data.select_dtypes(include="bool").astype("int8")
    data = data.select_dtypes(exclude="bool")
    data = pd.concat([data, temp], axis=1)

    data = pd.DataFrame(MinMaxScaler().fit_transform(data.drop("price_usd", axis=1)))

    return data


if file:
    df = pd.read_csv(file).drop("Unnamed: 0", axis=1)
    st.write(df)
    st.write(get_prepared(df))

    btn = st.button("Понизить размерность до 2")

    if btn:
        with open(r"./models/5_stacking_regression.pickle", "rb") as f:
            pca = pickle.load(f)
            reduced = pca.transform(get_prepared(df))
            st.download_button("Загрузить файл", pd.DataFrame(reduced).to_csv(), "reducted.csv")
