import streamlit as st
import pandas as pd
import pickle
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras

# file = st.file_uploader("Выберите файл с расширением .csv")
df_cars = pd.read_csv(r"./data/cars.csv")

# if file:
#     df = pd.read_csv(file)
#     st.write(df)

st.header("Получение прогноза для конкретного экземпляра")

manufacturer = st.selectbox("Марка", df_cars["manufacturer_name"].unique())

model = st.selectbox("Модель", df_cars[df_cars["manufacturer_name"] == manufacturer]["model_name"].unique())

transmission = st.selectbox("Тип трансмиссии", df_cars["transmission"].unique())

color = st.selectbox("Цвет кузова", df_cars["color"].unique())

odometer = st.slider("Пробег", 0, df_cars["odometer_value"].max())

year = st.slider("Год производства", int(df_cars["year_produced"].min()), int(df_cars["year_produced"].max()))

engine_fuel = st.selectbox("Тип топлива", df_cars["engine_fuel"].unique())

gbo = st.toggle("Установлено ли газобаллонное оборудование")

engine_type = st.selectbox("Тип двигателя", df_cars["engine_type"].unique())

engine_capacity = st.slider("Объём двигателя в литрах", 0.0, float(df_cars["engine_capacity"].max()), step=0.1)

body_type = st.selectbox("Тип кузова", df_cars["body_type"].unique())

has_warranty = st.toggle("Есть ли гарантия")

state = st.selectbox("Текущее состояние", df_cars["state"].unique())

drivetrain = st.selectbox("Тип привода", df_cars["drivetrain"].unique())

price_usd = st.slider("Стоимость в долларах", 0, int(df_cars["price_usd"].max()))

exchangeable = st.toggle("Возможен ли обмен")

location_region = st.selectbox("Город/область продажи", df_cars["location_region"].unique())

number_of_photos = st.slider("Количество фотографий", 0, df_cars["number_of_photos"].max())

up_counter = st.slider("Количество посещений сервиса", 0, df_cars["up_counter"].max())

feature_0 = st.toggle("feature_0")
feature_1 = st.toggle("feature_1")
feature_2 = st.toggle("feature_2")
feature_3 = st.toggle("feature_3")
feature_4 = st.toggle("feature_4")
feature_5 = st.toggle("feature_5")
feature_6 = st.toggle("feature_6")
feature_7 = st.toggle("feature_7")
feature_8 = st.toggle("feature_8")
feature_9 = st.toggle("feature_9")

duration_listed = st.slider("Дней в продаже", 0, df_cars["duration_listed"].max())


new_df = pd.DataFrame(
    [[
        manufacturer,
        model,
        transmission,

        color,

        odometer,

        year,

        engine_fuel,

        gbo,

        engine_type,

        engine_capacity,

        body_type,

        has_warranty,

        state,

        drivetrain,

        price_usd,

        exchangeable,

        location_region,

        number_of_photos,
        up_counter,
        feature_0,
        feature_1,
        feature_2,
        feature_3,
        feature_4,
        feature_5,
        feature_6,
        feature_7,
        feature_8,
        feature_9,
        duration_listed,
    ]],
    columns=df_cars.columns,
)

st.subheader("Введённые данные")
st.write(new_df)


def get_prepared_row(df):
    df = pd.concat([new_df, df_cars], axis=0, ignore_index=True)

    df["engine_capacity"].fillna(0, inplace=True)

    bin = BinaryEncoder()
    binarized_categorical = bin.fit_transform(df.select_dtypes(include="object")).astype("int8")
    data = df.select_dtypes(exclude="object")
    data = pd.concat([data, pd.DataFrame(binarized_categorical)], axis=1)
    
    temp = data.select_dtypes(include="bool").astype("int8")
    data = data.select_dtypes(exclude="bool")
    data = pd.concat([data, temp], axis=1)

    return data.loc[:1].drop("price_usd", axis=1)


row = get_prepared_row(new_df)


predictions = []
btn = st.button("Рассчитать")
if btn:
    with open(r"./models/1_multi_dimensional_regression.pickle", "rb") as f:
        reg = pickle.load(f)
        pred = reg.predict(row)[0]
        predictions.append(pred)
        st.write("Множественная линейная регрессия: " + str(round(pred)))

    with open(r"./models/3_gradient_boosting_regression.pickle", "rb") as f:
        gbr = pickle.load(f)
        pred = gbr.predict(row)[0]
        predictions.append(pred)
        st.write("Градиентный бустинг: " + str(round(pred)))

    with open(r"./models/4_bagging_regression.pickle", "rb") as f:
        br = pickle.load(f)
        pred = br.predict(row)[0]
        predictions.append(pred)
        st.write("Бэггинг: " + str(round(pred)))

    with open(r"./models/5_stacking_regression.pickle", "rb") as f:
        sr = pickle.load(f)
        pred = sr.predict(row)[0]
        predictions.append(pred)
        st.write("Стекинг: " + str(round(pred)))

    nn = keras.models.load_model(r"./models/6_neural_network_regression.keras")
    pred = nn.predict(row)
    st.write("Полносвязная нейронная сеть:", str(round(int(pred[0]))))

    st.write("Усреднённое значение прогнозов:", str(round(np.array(predictions).mean())))
