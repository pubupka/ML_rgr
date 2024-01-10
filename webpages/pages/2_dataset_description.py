import streamlit as st
import pandas as pd


st.title("О датасете")
st.write("Все модели были обучены на датасете, в котором необходимо прогнозировать стоимость автомобиля в долларах")
df = pd.read_csv(r"./data/cars.csv")
st.dataframe(df, hide_index=True)

description = pd.DataFrame(
    [
        ["manufacturer_name", "Марка автомобиля"],
        ["model_name", "Название модели"],
        ["transmission", "Тип трансмиссии"],
        ["color", "Цвет кузова"],
        ["odometer_value", "Пробег автомобиля"],
        ["year_produced", "Год производства"],
        ["engine_fuel", "Тип топлива"],
        ["engine_has_gas", "Установлено ли газобаллонное оборудование"],
        ["engine_type", "Тип двигателя"],
        ["engine_capacity", "Объём двигателя в литрах"],
        ["body_type", "Тип кузова"],
        ["has_warranty", "Действует ли гарантия на автомобиль"],
        ["state", "Текущее состояние (новая, в пользовании, сломана)"],
        ["drivetrain", "Тип привода"],
        ["price_usd", "Стоимость в долларах"],
        ["is_exchangeable", "Возможен ли обмен"],
        ["location_region", "Город/область продажи"],
        ["number_of_photos", "Количество фотографий"],
        ["up_counter", "Количество посещений сервиса"],
        ["feature_0-9", "Элементы комплектации (кондиционер, стеклоподъёмники, люк и т.д.)"],
        ["duration_listed", "Сколько дней стоит на продаже"],
    ],
    columns=["Название столбца", "Описание"]
).style.hide().hide(axis=1)

with st.expander("Описание датасета"):
    st.dataframe(description, hide_index=True, use_container_width=True)

with st.expander("Особенности предобработки"):
    st.write(
        "В датасете было несколько пропусков в столбце 'engine_capacity', все эти пропуски были для автомобилей\
         с электродвигателем. Они были заполнены нулями."
    )
    st.write(
        "Также перед обучением моделей столбцы были исследованы на нормальность распределения.\
         Нормальным было распределение лишь одного столбца, поэтому данные были масштабированы при помощи MinMaxScaler."
    )
    st.write(
        "Категориальные признаки были закодированы с помощью кодировщика BinaryEncoder, чтобы не создавать слишком много столбцов"
    )
