import numpy as np
import matplotlib.pyplot as plt


def gradient(x, y):
    return analytical_partial_derivative_of_x(y), analytical_partial_derivative_of_y(x, y)


def function(x, y):
    return (x / np.sin(y)) - 5 * x - np.log10(y ** 3)


def analytical_partial_derivative_of_x(y):
    return (1 / np.sin(y)) - 5


def analytical_partial_derivative_of_y(x, y):
    return x * (-np.cos(y) / np.sin(y) ** 2) - 3 / (y * np.log(10))


def numerical_derivative_derivative_of_x(x, y, step):
    return (function(x + step, y) - function(x - step, y)) / (2 * step)


def numerical_derivative_derivative_of_y(x, y, step):
    return (function(x, y + step) - function(x, y - step)) / (2 * step)


def draw_function():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    i = np.arange(0.0, 3.0, 0.01)
    x, y = np.meshgrid(i, i)
    f = function(x, y)

    ax.plot_surface(x, y, f)
    ax.set(title='x / np.sin(y)) - 5 * x - np.log10(y ** 3)')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    x = 2.5
    y = 3.0
    step = 1e-5

    gradient_function = gradient(x, y)

    draw_function()

    print("function value at x = {0} and y = {1} is {2}".format(x, y, function(x, y)))
    print()
    print("analytical partial derivative of x value at x = {0}, y = {1} is {2}".format(
        x, y, analytical_partial_derivative_of_x(y)))
    print("numerical partial derivative of x value at x = {0}, y = {1} with step = {2} is {3}".format(
        x, y, step, numerical_derivative_derivative_of_x(x, y, step)))
    print()
    print("analytical partial derivative of y value at x = {0}, y = {1} is {2}".format(
        x, y, analytical_partial_derivative_of_y(x, y)))
    print("numerical partial derivative of y value at x = {0}, y = {1} with step = {2} is {3}".format(
        x, y, step, numerical_derivative_derivative_of_y(x, y, step)))
    print()
    print("gradient x = {0}, y = {1}".format(gradient_function[0], gradient_function[1]))
