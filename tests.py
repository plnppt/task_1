import pytest
import main


@pytest.mark.parametrize("x, y, expected_result", [(0, 1, 0),
                                                   (0, 10, -3),
                                                   (0, 100, -6)])
def test_function(x, y, expected_result):
    assert main.function(x, y) == expected_result


@pytest.mark.parametrize("x, y", [(1, 2),
                                  (5, 10),
                                  (10, 3),
                                  (50, 60),
                                  (12, 175)])
def test_derivative_1(x, y):
    assert abs(main.analytical_partial_derivative_of_x(y) - main.numerical_derivative_derivative_of_x(x, y, 1e-5)) < 1e-4


@pytest.mark.parametrize("x, y", [(1, 2),
                                  (5, 10),
                                  (10, 3),
                                  (50, 60),
                                  (12, 175)])
def test_derivative_2(x, y):
    assert abs(main.analytical_partial_derivative_of_y(x, y) - main.numerical_derivative_derivative_of_y(x, y, 1e-5)) < 1e-4


@pytest.mark.parametrize("x, y, expected_x, expected_y", [(2, 5, -6.04284, -0.877545),
                                                          (4, 3, 2.08617, 198.41072),
                                                          (7, 9, -2.57351, 37.40734)])
def test_gradient(x, y, expected_x, expected_y):
    gradient = main.gradient(x, y)
    assert abs(gradient[0] - expected_x) < 0.00001 and abs(gradient[1] - expected_y) < 0.00001
