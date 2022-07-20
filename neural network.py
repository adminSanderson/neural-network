from wehter import ani, anin

x_input = [ani, ani] #Вводные данные            # 1. умножаем первый столбик
w_weights = [anin, anin] #начальный набор весов   # 2, умножаем 2 столбик и плюсуем 1 столбик
threshold = 0.9 # порог

# активируем функцию

def step(weighted_sum):
    if weighted_sum > threshold: #если сумма весов > порога, то
        return 1
    else:
        return 0

def perceptron(): #перцептрон
    weighted_sum = 0 #взвешенная сумма на 0
    for x,w in zip(x_input, w_weights): #перебираем списки x_input и w_weights умножая (ВАЖНО!: 0/1 * 0.4!!!!!!!!!!)
        weighted_sum += x*w #(ВАЖНО!: 0/1 * 0.4!!!!!!!!!!) в x_input и w_weights
        print(weighted_sum)
    return step(weighted_sum)

output = perceptron()
print("Вывод: " + str(output))
if output >= 15:
    print("Закрыть")
else:
    print("Открыть")