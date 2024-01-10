# *DashBoard для вывода моделей ML*
## https://khalimov-rgr.streamlit.app/models_predictions


# Веб-приложение включает в себя 5 страниц:
## Титульная страница.
ФИО и группа.

## Описание датасета
Описание каждого столбца в датасете и особенностей его предобработки.

## Визуализации
На этой страницы представлены такие графики, как:
1. Тепловая карта корреляций числовых признаков
2. BoxPlot стоимости автомобилей
3. Гистограмма распределения пробега автомобилей
4. Круговая диаграмма типов привода автомобилей

## Получение прогнозов
С помощью интерактивных элементов пользователь вводит параметры одного автомобиля, по которым может получить его приблизительную стоимость на основе прогнозов пяти моделей: множественная линейная регрессия,
градиентный бустинг, бэггинг, стекинг, полносвязная нейронная сеть.

## Понижение размерности
Данная страница предоставляет интерфейс для использования алгоритма понижения размерности PCA, который обучается на неразмеченных данных.
