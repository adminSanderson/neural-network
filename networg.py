import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        # раздача для генерации случайных чисел
        np.random.seed(1)
        
        # преобразование весов в матрицу 3 на 1 со значениями от -1 до 1 и средним значением 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #применение сигмовидной функции
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #вычисление производной сигмовидной функции
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #обучение модели делать точные прогнозы при постоянной корректировке весов
        for iteration in range(training_iterations):
            #отсасывать обучающие данные через нейрон
            output = self.think(training_inputs)

            # вычисление коэффициента ошибок для обратного распространения
            error = training_outputs - output
            
            #коррекция веса
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #передача входных данных через нейрон для получения выходных данных
        #преобразование значений в числа с плавающей запятой
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    #инициализация класса нейронов
    neural_network = NeuralNetwork()

    print("Начальные случайно сгенерированные веса: ")
    print(neural_network.synaptic_weights)

    # обучающие данные, состоящие из 4 примеров - 3 входных значения и 1 выход
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    #проходит тренировка
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Завершающие веса после тренировки: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))
    
    print("Учитывая новую ситуацию: ", user_input_one, user_input_two, user_input_three)
    print("Новые выходные данные: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("Мы сделали это.")