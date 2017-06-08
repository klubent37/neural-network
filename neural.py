import numpy as np


def nonlin(x, deriv=False):
    if deriv is True:
        return x*(1-x)
    return 1/(1+np.exp(-x))


data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])

output = np.array([[0],
                   [1],
                   [1],
                   [0]])

np.random.seed(1)

# случайно инициализируем веса, в среднем - 0
synapse0 = 2*np.random.random((3, 4)) - 1
synapse1 = 2*np.random.random((4, 1)) - 1

while True:
    # проходим вперёд по слоям 0, 1 и 2
    input_data = data
    hidden_layer = nonlin(np.dot(input_data, synapse0))
    l2_output_data = nonlin(np.dot(hidden_layer, synapse1))

    # как сильно мы ошиблись относительно нужной величины?
    l2_error = output - l2_output_data

    print("Ошибка:" + str(np.mean(np.abs(l2_error))))

    # в какую сторону нужно двигаться?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l2_delta = l2_error*nonlin(l2_output_data, deriv=True)

    # Умножение весов на синапсы (по формуле)
    l1_error = l2_delta.dot(synapse1.T)

    # в каком направлении нужно двигаться, чтобы прийти к l1?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l1_delta = l1_error * nonlin(hidden_layer, deriv=True)

    synapse1 += hidden_layer.T.dot(l2_delta)
    synapse0 += input_data.T.dot(l1_delta)

    if np.mean(np.abs(l2_error)) < 0.001:
        break

print("Результат после тренировок \n", l2_output_data)
