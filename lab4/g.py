import random
import math

# Функция активации
def f(net):
    return (1 - math.exp(-net)) / (1 + math.exp(-net))

# Производная функции активации
def df(y):
    return 0.5 * (1 - y**2)

# Инициализация весов
def init_weights():
    w1 = [[random.uniform(-1, 1) for j in range(N)] for i in range(J)]
    w2 = [[random.uniform(-1, 1) for j in range(J)] for i in range(M)]
    return w1, w2

# Обучение нейронной сети
def train(x, y, n, eps):
    # Инициализация весов
    w1, w2 = init_weights()
    # Цикл обучения
    while True:
        # Прямой проход
        h = [f(sum([x[i]*w1[j][i] for i in range(N)])) for j in range(J)]
        o = [f(sum([h[j]*w2[k][j] for j in range(J)])) for k in range(M)]
        # Ошибка
        err = math.sqrt(sum([(y[i]-o[i])**2 for i in range(M)]))
        if err < eps:
            break
        # Обратный проход
        delta_o = [(y[i]-o[i])*df(o[i]) for i in range(M)]
        delta_h = [df(h[j])*sum([delta_o[k]*w2[k][j] for k in range(M)]) for j in range(J)]
        # Корректировка весов
        for k in range(M):
            for j in range(J):
                w2[k][j] += n*delta_o[k]*h[j]
        for j in range(J):
            for i in range(N):
                w1[j][i] += n*delta_h[j]*x[i]
    return w1, w2

# Тестирование нейронной сети
def test(x, w1, w2):
    h = [f(sum([x[i]*w1[j][i] for i in range(N)])) for j in range(J)]
    o = [f(sum([h[j]*w2[k][j] for j in range(J)])) for k in range(M)]
    return o

# Параметры
N = 1
J = 1
M = 3
x = [1,-2]
y = [0.2, 0.1, 0.3]
n = 1
eps = 0.001

# Обучение нейронной сети
w1, w2 = train(x, y, n, eps)

# Тестирование нейронной сети
o = test(x, w1, w2)
print(o)