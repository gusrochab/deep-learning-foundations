from numpy import *
import matplotlib.pyplot as plt


def compute_error_for_given_ponits(b, m, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) **2
    return totalError / len(points)


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def graph(b_current, m_current, points):
    x_array = []
    y_array = []
    y_calc = []
    for i in range(0, len(points)):
        x_array.append(points[i, 0])
        y_array.append(points[i, 1])
        y_calc.append(m_current * x_array[i] + b_current)

    plt.figure(figsize=(5, 5))
    plt.scatter(x_array, y_array)
    plt.plot(x_array, y_calc)
    plt.show()


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        error = compute_error_for_given_ponits(b, m, points)
        #print('{} - b: {}, m: {}, ERROR: {}'.format(i, b, m, error))

    return [b, m]


def run():
    points = genfromtxt('data.txt', delimiter=',')
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('FINAL - b: {}, m: {}'.format(b, m))
    graph(b, m, array(points))


if __name__ == '__main__':
    run()