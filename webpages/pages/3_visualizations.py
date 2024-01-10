import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Основные визуализации")


df = pd.read_csv(r"./data/cars.csv")

st.header("Тепловая карта корреляций числовых признаков")
fig = plt.figure()
fig.add_subplot(sns.heatmap(df.select_dtypes(exclude=["object", "bool"]).corr(), annot=True))
st.pyplot(fig)

st.header("График BoxPlot цены автомобиля")
fig = plt.figure()
plt.boxplot(df["price_usd"])
st.pyplot(plt)

st.header("Гистограмма распределения пробега автомобилей")
fig = plt.figure()
fig.add_subplot(sns.histplot(df["odometer_value"]))
st.pyplot(fig)

st.header("Круговая диаграмма для типа привода автомобиля")
fig = plt.figure()
size = df.groupby("drivetrain").size()
sns.set_palette("rocket")
plt.pie(size.values, labels=size.index, counterclock=True, autopct='%1.0f%%')
st.pyplot(fig)
