import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
import rgb

# потом уберу
def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)  

# Настройки Seaborn и Matplotlib для темной темы
sns.set(style="darkgrid")
plt.style.use('dark_background')

# Функция для создания графика ЗБЧ
def create_plot(distribution_type, sample_size):
    if distribution_type == 'normal':
        data = np.random.normal(size=sample_size)
    elif distribution_type == 'uniform':
        data = np.random.uniform(size=sample_size)
    else:  # логнормальное распределение
        data = lognorm.rvs(3, size=sample_size)

    df = pd.DataFrame(data)
    df['cum'] = df[0].cumsum()
    df['sample_sizes'] = range(1, sample_size + 1)
    df['mean'] = df['cum'] / df['sample_sizes']

    plt.figure(figsize=(7, 4), facecolor=rgb(0,5,16))  # не пашет rgb
    plot = sns.lineplot(x='sample_sizes', y='mean', data=df, color='cyan')
    plot.set(title=f'Среднее значение {distribution_type} при размере выборки {sample_size}',
             xlabel='Размер выборки', ylabel='Среднее значение')
    plt.title(f'Среднее значение {distribution_type} при размере выборки {sample_size}', color='white')
    plt.xlabel('Размер выборки', color='white')
    plt.ylabel('Среднее значение', color='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    st.pyplot(plt)

# Cоздания гистограммы ЦПТ
def central_limit_theorem(distribution_type, sample_size, num_samples):
    if distribution_type == 'normal':
        data = np.random.normal(size=(num_samples, sample_size))
    elif distribution_type == 'uniform':
        data = np.random.uniform(size=(num_samples, sample_size))
    else:  
        data = np.random.lognormal(0, 1, size=(num_samples, sample_size))

    sample_means = np.mean(data, axis=1)
    population_mean = np.mean(sample_means)
    population_std = np.std(sample_means)

    plt.figure(figsize=(10, 6), facecolor=rgb(0,5,16)) 
    sns.histplot(sample_means, color='cyan', bins=30, kde=False)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, population_mean, population_std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Гистограмма выборочных средних ({distribution_type}) при размере выборки {sample_size}', color='white')
    plt.xlabel('Среднее значение', color='white')
    plt.ylabel('Частота', color='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    st.pyplot(plt)

st.title('Визуализация статистических закономерностей')

# Сайдбар 
page = st.sidebar.selectbox('Выберите страницу', ['Закон больших чисел', 'Центральная предельная теорема'])

if page == 'Закон больших чисел':
    st.header('Визуализация закона больших чисел')

    # Слайдеры
    distribution_type = st.sidebar.selectbox('Выберите тип распределения', ['Нормальное', 'Равномерное', 'Логнормальное'])
    sample_size = st.sidebar.slider('Выберите размер выборки', min_value=10, max_value=10000, value=1000, step=10)

    if distribution_type == 'Нормальное':
        create_plot('normal', sample_size)
    elif distribution_type == 'Равномерное':
        create_plot('uniform', sample_size)
    else:
        create_plot('lognormal', sample_size)

else:
    st.header('Визуализация Центральной предельной теоремы')

    # Параметров для ЦПТ
    distribution_type = st.sidebar.selectbox('Выберите тип распределения', ['Нормальное', 'Равномерное', 'Логнормальное'])
    sample_size = st.sidebar.slider('Выберите размер выборки', min_value=10, max_value=1000, value=100, step=10)
    num_samples = st.sidebar.slider('Выберите количество выборок', min_value=10, max_value=1000, value=100, step=10)

    # ЦПТ
    central_limit_theorem(distribution_type.lower(), sample_size, num_samples)


