from numpy import random
import numpy as np
from time import perf_counter
import csv
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter


def task_1():
    N_SIZE = 1000000
    RANDOM_SEED = 100000

    t1_start = perf_counter()
    x1 = random.randint(RANDOM_SEED, size=N_SIZE)
    y1 = random.randint(RANDOM_SEED, size=N_SIZE)
    z1 = np.multiply(x1, y1)
    t1_close = perf_counter()

    t2_start = perf_counter()
    x2 = [random.randint(RANDOM_SEED) for i in range(N_SIZE)]
    y2 = [random.randint(RANDOM_SEED) for i in range(N_SIZE)]
    z2 = []
    for i in range(len(x2)):
        z2.append(x2[i]*y2[i])
    t2_close = perf_counter()

    print(
        f"Время выполнения операции поэлементного перемножения массивов NumPy (numpy.array): {t1_close-t1_start}")
    print(
        f"Время выполнения операции поэлементного перемножения стандартных списков:          {t2_close - t2_start}")


def task_2():
    # Чтобы создать диаграмму результатов с двумя столбцами и соответствующими метками
    def scatter_graph(values, labels):
        fig, ax1 = plt.subplots()
        # Создать ax2, чтобы у меня было 2 оси y
        ax2 = ax1.twinx()
        ax1.set_xlabel(labels[0])

        # Первая ось дает синюю точку
        ax1.scatter(values[0], values[1], label=labels[1], color='dodgerblue')
        ax1.set_ylabel(labels[1], color='dodgerblue')

        # 2-я ось для оранжевой точки
        ax2.scatter(values[0], values[2], label=labels[2], color='orange')
        ax2.set_ylabel(labels[2], color='orange')

        # Показать комментарии
        fig.legend(loc="upper right")

        # Показать график вверх
        plt.title(f"График {labels[1]} и {labels[2]} от {labels[0]}")
        plt.xlim(0, 38)
        plt.show()

    def density_curve(values, labels):
        # Преобразуйте диапазон, чтобы вернуть его к однородности
        # Потому что пространство неоднородно

        min_val = min(values[2])
        max_val = max(values[2])

        dataset = DataFrame(
            {
                labels[0]: values[0],
                labels[1]: values[1],
                # [x, y] -> [u, v] ~ [0, y - x] -> [0, v - u]
                # i принадлежит [x, y] -> (i - x) / (y - x) * (v - u) + u
                labels[2]: [((i - min_val) * 100) /
                            (max_val - min_val) for i in values[2]]
            }
        )

        fig, ax1 = plt.subplots()

        # Построить гистограмму и кривую плотности для столбца 4
        sns.distplot(dataset[[labels[1]]], label=labels[1],
                     color='dodgerblue')

        # Построить гистограмму и кривую плотности для столбца 16
        sns.distplot(dataset[[labels[2]]],
                     label=labels[2], color='orange')

        # Примечание
        ax1.set_ylabel("Плотность")
        fig.legend(loc="upper right")

        # Показать график
        plt.title(f'График корреляции {labels[1]} и {labels[2]}')
        plt.show()

    file_name = 'data1.csv'
    columns = []
    list1 = []
    list4 = []
    list16 = []

    # Используя приложенный файл ```data1.csv```, подсчитать количество записей в нём
    with open(file_name, 'rt', newline='', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = 1
                columns = row
            else:
                line_count += 1
                list1.append(float(row[0]))
                list4.append(float(row[3]))
                list16.append(float(row[15]))

    scatter_graph([list1, list4, list16], [
                  columns[0], columns[3], columns[15]])
    density_curve([list1, list4, list16], [
                  columns[0], columns[3], columns[15]])


def task_3():
    # Задаем интервал x
    x = np.linspace(-np.pi, np.pi, 100)

    # Вычисляем значения y и z
    y = np.sin(x) * np.cos(x)
    z = np.sin(x)

    # Создаем объект для построения трехмерного графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # # Строим график
    ax.plot(x, y, z)

    # Добавляем названия осей
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    line, = ax.plot([], [], [], lw=2)

    # Рисуйте медленно, как анимация 0 -> i - 1
    # Функция, которая будет вызываться на каждом кадре анимации
    def animate(i):
        line.set_data(x[:i], y[:i])
        line.set_3d_properties(z[:i])
        return (line,)

    # Создание анимации
    anim = FuncAnimation(fig, animate, frames=len(x)+1, interval=50)

    # Сохранение анимации в файл
    writer = PillowWriter(fps=20)
    anim.save('task3.gif', writer=writer)

    # Показываем график
    plt.show()


def task_4():
    # Создание функции y=sin(x)
    x = np.arange(0, 2*np.pi, 0.1)
    y = np.sin(x)

    # Создание фигуры и осей графика
    fig, ax = plt.subplots()
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('График функции y=sin(x)')

    # Создание линии графика и добавление ее на график
    line, = ax.plot([], [], lw=2)

    # Функция, которая будет вызываться на каждом кадре анимации

    def animate(i):
        line.set_data(x[:i], y[:i])
        return (line,)

    # Создание анимации
    anim = FuncAnimation(fig, animate, frames=len(x)+1, interval=50)

    # Сохранение анимации в файл
    writer = PillowWriter(fps=20)
    anim.save('task4.gif', writer=writer)

    plt.show()


task_1()
task_2()
task_3()
task_4()
