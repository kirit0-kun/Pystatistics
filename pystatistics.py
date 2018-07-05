import io
from functools import reduce
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt, log, exp, pow


class Math(object):

    @staticmethod
    def solve_eqt(*args) -> list:
        """
        :param args: list of coefficients of the equations to be solved
            eg:  x +  y +  z = 9
                2x + 3y +  z = 7
                3x + 4y + 5z = 5
                args are [[1,1,1,9],[2,3,1,7],[3,4,5,5]]
        :return: a list of the values of the unknowns respectively
        """
        n = len(args)
        a = np.array([arg[0:n] for arg in args[0:n]])
        b = np.array([arg[n] for arg in args[0:n]])
        ans = np.linalg.solve(a, b)
        return np.round(ans, 2).tolist()


# noinspection PyBroadException
class Statistics(object):

    @staticmethod
    def get_arithmetic_mean(x_data: list, f_data: list = None) -> float:
        """
        :param x_data: a list of the values the mean is calculated for
        :param f_data: a list of frequencies
        :return: the mean of the values
        """
        x_data = np.array(x_data)
        f_data = np.array(f_data) if f_data else 1
        mean = x_data * f_data
        mean = mean.mean() if type(f_data) is int else (mean.sum() / f_data.sum())
        return round(mean, 2)

    @staticmethod
    def geometric_mean(x_data: list, f_data: list = None) -> float:
        x_data = np.array(x_data)
        f_data = np.array(f_data) if f_data else 1
        mean = np.power(x_data, f_data)
        mean = reduce(lambda a, b: a * b, mean.tolist())
        freq = len(x_data) if type(f_data) is int else f_data.sum()
        mean = pow(mean, 1 / freq)
        return round(mean, 2)

    @staticmethod
    def harmonic_mean(x_data: list, f_data: list = None) -> float:
        f_data = np.array(f_data) if f_data else 1
        x_data = np.array(x_data)
        weights = f_data if type(f_data) is int else f_data.sum()
        mean = weights / (f_data / x_data).sum()
        return round(mean, 2)

    @staticmethod
    def get_mode(x_data: list, f_data: list = None):
        """
        :param x_data: a list of the values the mode is calculated for
        :param f_data: a list of frequencies
        :return: the mode of the values
        """
        if f_data is None:
            data, mode, count = x_data, [0, ], 0
            for elem in data:
                if elem != mode[0]:
                    if data.count(elem) > count:
                        mode.clear()
                        mode.append(elem)
                        count = data.count(elem)
                    elif data.count(elem) == count and elem != mode[len(mode) - 1]:
                        mode.append(elem)
            return mode
        else:
            data = dict(zip(x_data, f_data))
            b = max(data.values())
            temp = {data[t]: t for t in list(data.keys())}
            a = temp[b]
            temp = sorted(list(data.keys()))
            c = data[temp[temp.index(a) - 1]] if (temp.index(a) - 1) < len(temp) else 0
            d = data[temp[temp.index(a) + 1]] if (temp.index(a) + 1) < len(temp) else 0
            e = abs(temp[1] - temp[0])
            mode = a + ((b - c) / (2 * b - c - d)) * e
            return round(mode, 2)

    @staticmethod
    def get_median(x_data: list, f_data: list = None):
        """
        :param x_data: a list of the values the median is calculated for
        :param f_data: a list of frequencies
        :return: the median of the values
        """
        mean = Statistics.get_arithmetic_mean(x_data=x_data, f_data=f_data)
        x_data = np.array(x_data)
        median = np.median(x_data) if not f_data else mean - (
                mean - Statistics.get_mode(x_data=list(x_data.tolist()), f_data=f_data)) / 3
        return round(median, 2)

    @staticmethod
    def get_mean_deviation(x_data: list, f_data: list = None) -> float:
        """
        :param x_data: a list of the values the mean deviation is calculated for
        :param f_data: a list of frequencies
        :return: the mean deviation of the values
        """
        mean = Statistics.get_arithmetic_mean(x_data=x_data, f_data=f_data)
        x_data = np.array(x_data)
        f_data = np.array(f_data) if f_data else 1
        if type(f_data) is int:
            mean_deviation = np.abs(x_data - mean).mean()
            return mean_deviation
        else:
            mean_deviation = np.abs((x_data - mean) * f_data).sum() / f_data.sum()
            return round(mean_deviation, 2)

    @staticmethod
    def get_variance(x_data: list, f_data: list = None, is_sample: bool = True) -> float:
        """
        :param x_data: a list of the values the variance is calculated for
        :param f_data: a list of frequencies
        :param is_sample: are the values a sample or a population
        :return: the variance of the values
        """
        mean = Statistics.get_arithmetic_mean(x_data=x_data, f_data=f_data)
        x_data = np.array(x_data)
        f_data = np.array(f_data) if f_data else 1
        if type(f_data) is int:
            variance = (np.abs(x_data - mean) ** 2).sum() / (x_data.size - int(not is_sample))
        else:
            variance = ((np.abs(x_data - mean) ** 2) * f_data).sum() / (f_data.sum() - int(not is_sample))
        return round(variance, 2)

    @staticmethod
    def get_standard_deviation(x_data: list, f_data: list = None, is_sample: bool = True) -> float:
        """
        :param x_data: a list of the values the standard deviation is calculated for
        :param f_data: a list of frequencies
        :param is_sample: are the values a sample or a population
        :return: the standard deviation of the values
        """
        return round(
            complex(sqrt(Statistics.get_variance(x_data=x_data, f_data=f_data, is_sample=is_sample))).real,
            2)

    @staticmethod
    def get_coefficient_of_variation(x_data: list, f_data: list = None, is_sample: bool = True) -> str:
        """
        :param x_data: a list of the values the coefficient of variation is calculated for
        :param f_data: a list of frequencies
        :param is_sample: are the values a sample or a population
        :return: the coefficient of variation of the values
        """
        return str(round(abs(100 * (Statistics.get_standard_deviation(x_data=x_data, f_data=f_data,
                                                                      is_sample=is_sample) /
                                    Statistics.get_arithmetic_mean(
                                        x_data=x_data, f_data=f_data))), 2)) + '%'

    @staticmethod
    def polynomial_regression_model(list_x: list, list_y: list, n: int = 1) -> dict:
        """
        :param list_x: values of the first sample
        :param list_y: values of the second sample
        :param n: the degree of the polynomial or the highest power of x
        :return: a dictionary  {"coefs": coefficients of the polynomial starting with
                                        x to the power of zero till the highest power,
                                "formula": a string representation of the formula,
                                "coc": coefficient of correlation,
                                "ser": standard error of the correlation
                                }
        """
        ans = dict()
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        mat1 = dict()
        for i in range(n + 1):
            mat1[i] = list_x ** i
        mat1 = np.array([data for key, data in mat1.items()])
        mat1 = mat1.T
        coefs = np.round(np.linalg.inv((mat1.T.dot(mat1))).dot(mat1.T).dot(list_y), 2)
        ans['coefs'] = coefs
        formula = "Y = " + (str(coefs[0]) if coefs[0] != 0 else '')
        formula += (" +" if coefs[1] > 0 else " ") + ((str(coefs[1]) + "X") if coefs[1] != 0 else '')
        for i in range(2, len(coefs)):
            _1 = "+" if coefs[i] > 0 else ""
            formula += (" " + _1 + str(coefs[i]) + "X^" + str(i) + '') if coefs[i] != 0 else ''
        ans['formula'] = formula
        predicted_y = np.array(Statistics.apply_polynomial_formula(list(list_x.tolist()), list(coefs.tolist())))
        numerator = list_y - predicted_y
        numerator **= 2
        denominator = (list_y - list_y.mean()) ** 2
        coc = 0
        try:
            coc = round(sqrt(1 - (np.sum(numerator) / np.sum(denominator))), 2)
        except:
            coc = 0
        finally:
            ans['coc'] = coc
            standard_error = np.abs(np.divide(list_y - predicted_y, list_y, where=list_y != 0)).mean()
            ans['ser'] = round(standard_error * 100, 2)
            return ans

    @staticmethod
    def power_regression_model(list_x: list, list_y: list) -> dict:
        """
        :param list_x: values of the first sample
        :param list_y: values of the second sample
        :return: a dictionary  {"b": value of b,
                                "a": value of a,
                                "formula": a string representation of the formula Y=a.X^b,
                                "coc": coefficient of correlation,
                                "ser": standard error of the correlation
                                }
        """
        ans = dict()
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        n = list_x.size
        b = (n * (np.log(list_x) * np.log(list_y)).sum() - (np.log(list_x).sum() * np.log(list_y).sum())) / (
                n * (np.log(list_x) ** 2).sum() - ((np.log(list_x)).sum() ** 2))
        ans['b'] = round(b, 2)
        a = exp(((1 / n) * np.log(list_y).sum()) - ((b / n) * np.log(list_x).sum()))
        ans['a'] = round(a, 2)
        formula = 'Y = ' + str(ans['a']) + '.' + 'X^' + str(ans['b'])
        ans['formula'] = formula
        predicted_y = np.array(Statistics.apply_power_formula(list(list_x.tolist()), dict(a=a, b=b)))
        numerator = list_y - predicted_y
        numerator **= 2
        denominator = (list_y - list_y.mean()) ** 2
        coc = 0
        try:
            coc = round(sqrt(1 - (np.sum(numerator) / np.sum(denominator))), 2)
        except:
            coc = 0
        finally:
            ans['coc'] = coc
            standard_error = np.abs(np.divide(list_y - predicted_y, list_y, where=list_y != 0)).mean()
            ans['ser'] = round(standard_error * 100, 2)
            return ans

    @staticmethod
    def ab_exponential_regression_model(list_x: list, list_y: list) -> dict:
        """
        :param list_x: values of the first sample
        :param list_y: values of the second sample
        :return: a dictionary  {"b": value of b,
                                "a": value of a,
                                "formula": a string representation of the formula Y=a.b^X,
                                "coc": coefficient of correlation,
                                "ser": standard error of the correlation
                                }
        """
        ans = dict()
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        n = list_x.size
        b = exp((n * (list_x * np.log(list_y)).sum() - (list_x.sum() * np.log(list_y).sum())) / (
                n * (list_x ** 2).sum() - (list_x.sum() ** 2)))
        ans['b'] = round(b, 2)
        a = exp(((1 / n) * np.log(list_y).sum()) - ((log(b) / n) * list_x.sum()))
        ans['a'] = round(a, 2)
        formula = 'Y = ' + str(ans['a']) + '.' + str(ans['b']) + '^X'
        ans['formula'] = formula
        predicted_y = np.array(Statistics.apply_ab_exponential_formula(list(list_x.tolist()), dict(a=a, b=b)))
        numerator = list_y - predicted_y
        numerator **= 2
        denominator = (list_y - list_y.mean()) ** 2
        coc = 0
        try:
            coc = round(sqrt(1 - (np.sum(numerator) / np.sum(denominator))), 2)
        except:
            coc = 0
        finally:
            ans['coc'] = coc
            standard_error = np.abs(np.divide(list_y - predicted_y, list_y, where=list_y != 0)).mean()
            ans['ser'] = round(standard_error * 100, 2)
            return ans

    @staticmethod
    def hyperbolic_regression_model(list_x: list, list_y: list) -> dict:
        """
        :param list_x: values of the first sample
        :param list_y: values of the second sample
        :return: a dictionary  {"b": value of b,
                                "a": value of a,
                                "formula": a string representation of the formula Y=a+X/b,
                                "coc": coefficient of correlation,
                                "ser": standard error of the correlation
                                }
        """
        ans = dict()
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        n = list_x.size
        b = ((n * (list_y / list_x).sum()) - ((np.ones(n) / list_x).sum() * list_y.sum())) / (
                (n * (np.ones(n) / (list_x ** 2)).sum()) - ((np.ones(n) / list_x).sum()) ** 2)
        ans['b'] = round(b, 2)
        a = ((1 / n) * list_y.sum()) - ((b / n) * (np.ones(n) / list_x).sum())
        ans['a'] = round(a, 2)
        formula = "Y = " + str(ans['a']) + ('+' if ans['b'] > 0 else '') + str(ans['b']) + '/X'
        ans['formula'] = formula
        predicted_y = np.array(Statistics.apply_hyperbolic_formula(list(list_x.tolist()), dict(a=a, b=b)))
        numerator = list_y - predicted_y
        numerator **= 2
        denominator = (list_y - list_y.mean()) ** 2
        coc = 0
        try:
            coc = round(sqrt(1 - (np.sum(numerator) / np.sum(denominator))), 2)
        except:
            coc = 0
        finally:
            ans['coc'] = coc
            standard_error = np.abs(np.divide(list_y - predicted_y, list_y, where=list_y != 0)).mean()
            ans['ser'] = round(standard_error * 100, 2)
            return ans

    @staticmethod
    def logarithmic_regression_model(list_x: list, list_y: list) -> dict:
        """
        :param list_x: values of the first sample
        :param list_y: values of the second sample
        :return: a dictionary  {"b": value of b,
                                "a": value of a,
                                "formula": a string representation of the formula Y=a+blnX,
                                "coc": coefficient of correlation,
                                "ser": standard error of the correlation
                                }
        """
        ans = dict()
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        n = list_x.size
        b = ((n * (list_y * np.log(list_x)).sum()) - ((np.log(list_x)).sum() * list_y.sum())) / (
                (n * (np.log(list_x) ** 2).sum()) - (np.log(list_x).sum() ** 2))
        ans['b'] = round(b, 2)
        a = ((1 / n) * list_y.sum()) - ((b / n) * np.log(list_x).sum())
        ans['a'] = round(a, 2)
        formula = "Y = " + str(ans['a']) + ('+' if ans['b'] > 0 else '') + str(ans['b']) + 'lnX'
        ans['formula'] = formula
        predicted_y = np.array(Statistics.apply_logarithmic_formula(list(list_x.tolist()), dict(a=a, b=b)))
        numerator = list_y - predicted_y
        numerator **= 2
        denominator = (list_y - list_y.mean()) ** 2
        coc = 0
        try:
            coc = round(sqrt(1 - (np.sum(numerator) / np.sum(denominator))), 2)
        except:
            coc = 0
        finally:
            ans['coc'] = coc
            standard_error = np.abs(np.divide(list_y - predicted_y, list_y, where=list_y != 0)).mean()
            ans['ser'] = round(standard_error * 100, 2)
            return ans

    @staticmethod
    def exponential_regression_model(list_x: list, list_y: list) -> dict:
        """
        :param list_x: values of the first sample
        :param list_y: values of the second sample
        :return: a dictionary  {"b": value of b,
                                "a": value of a,
                                "formula": a string representation of the formula Y=exp(a+bX),
                                "coc": coefficient of correlation,
                                "ser": standard error of the correlation
                                }
        """
        ans = dict()
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        n = list_x.size
        b = ((n * (list_x * np.log(list_y)).sum()) - (list_x.sum() * np.log(list_y).sum())) / (
                (n * (list_x ** 2).sum()) - (list_x.sum() ** 2))
        ans['b'] = round(b, 2)
        a = ((1 / n) * np.log(list_y).sum()) - ((b / n) * list_x.sum())
        ans['a'] = round(a, 2)
        formula = "Y = exp(" + str(ans['a']) + ('+' if ans['b'] > 0 else '') + str(ans['b']) + 'X)'
        ans['formula'] = formula
        predicted_y = np.array(Statistics.apply_exponential_formula(list(list_x.tolist()), dict(a=a, b=b)))
        numerator = list_y - predicted_y
        numerator **= 2
        denominator = (list_y - list_y.mean()) ** 2
        coc = 0
        try:
            coc = round(sqrt(1 - (np.sum(numerator) / np.sum(denominator))), 2)
        except:
            coc = 0
        finally:
            ans['coc'] = coc
            standard_error = np.abs(np.divide(list_y - predicted_y, list_y, where=list_y != 0)).mean()
            ans['ser'] = round(standard_error * 100, 2)
            return ans

    @staticmethod
    def apply_polynomial_formula(x_list: list, coefs: list) -> iter:
        """
        :param x_list: list: values of x to apply the a polynomial formula on
        :param coefs: list: coefficients of the polynomial formula
                        eg: y = 2x^2 + 4x + 5, coefs are [2, 4, 5]
        :return: a list of the outcome of the application of the polynomial formula on each value of x
        """
        x_list = np.array(x_list, dtype=np.float64)
        y_list = np.zeros(x_list.size, dtype=np.float64)
        n = len(coefs)
        for i in range(n):
            y_list += ((x_list ** (n - i - 1)) * coefs[n - i - 1])
        return y_list.tolist()

    @staticmethod
    def apply_power_formula(x_list: list, coefs: dict) -> iter:
        """
        :param x_list: list: values of x to apply the a polynomial formula on
        :param coefs: dict: coefficients of the formula Y=a.X^b as {"a":a, "b":b}
        :return: a list of the outcome of the application of the formula on each value of x
        """
        x_list = np.array(x_list, dtype=np.float64)
        y_list = coefs['a'] * (x_list ** coefs['b'])
        return y_list.tolist()

    @staticmethod
    def apply_ab_exponential_formula(x_list: list, coefs: dict) -> iter:
        """
        :param x_list: list: values of x to apply the a polynomial formula on
        :param coefs: dict: coefficients of the formula Y=a.b^X as {"a":a, "b":b}
        :return: a list of the outcome of the application of the formula on each value of x
        """
        x_list = np.array(x_list, dtype=np.float64)
        b = np.ones(x_list.size, dtype=np.float64) * coefs['b']
        y_list = coefs['a'] * np.power(b, x_list)
        return y_list.tolist()

    @staticmethod
    def apply_hyperbolic_formula(x_list: list, coefs: dict) -> iter:
        """
        :param x_list:list: values of x to apply the a polynomial formula on
        :param coefs:dict: coefficients of the formula Y=a+b/X as {"a":a, "b":b}
        :return: a list of the outcome of the application of the formula on each value of x
        """
        x_list = np.array(x_list, dtype=np.float64)
        b = np.ones(x_list.size, dtype=np.float64) * coefs['b']
        y_list = coefs['a'] + np.divide(b, x_list, where=x_list != 0)
        return y_list.tolist()

    @staticmethod
    def apply_logarithmic_formula(x_list: list, coefs: dict) -> iter:
        """
        :param x_list: list: values of x to apply the a polynomial formula on
        :param coefs: dict: coefficients of the formula Y=a+blnX as {"a":a, "b":b}
        :return: a list of the outcome of the application of the formula on each value of x
        """
        x_list = np.array(x_list, dtype=np.float64)
        y_list = coefs['a'] + np.log(x_list) * coefs['b']
        return y_list.tolist()

    @staticmethod
    def apply_exponential_formula(x_list: list, coefs: dict) -> iter:
        """
        :param x_list: list: values of x to apply the a polynomial formula on
        :param coefs: dict: coefficients of the formula Y=exp(a+bX) as {"a":a, "b":b}
        :return: a list of the outcome of the application of the formula on each value of x
        """
        x_list = np.array(x_list, dtype=np.float64)
        y_list = np.exp(coefs['a'] + x_list * coefs['b'])
        return y_list.tolist()

    @staticmethod
    def clear_plot() -> None:
        plt.figure()

    @staticmethod
    def add_to_plot(predicted_x: iter, predicted_y: iter, xlist: iter, ylist: iter, title: object,
                    color: str = 'blue') -> None:
        plt.scatter(xlist, ylist)
        plt.plot(predicted_x, predicted_y, color=color)
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")

    @staticmethod
    def plot(title: str = '') -> None:
        buf = io.BytesIO()
        plt.title(title) if title else None
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im.show()
        buf.close()
        plt.show()

    @staticmethod
    def polynomial_plot(x_list: list, y_list: list, n: int = 1, color: str = "red") -> None:
        titles = dict()
        titles[1] = "Linear Regression"
        titles[2] = "Quadratic Regression"
        titles[3] = "Cubic Regression"
        titles[0] = "Linear Regression"
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.polynomial_regression_model(x_list, y_list, n)['coefs']
        predicted_y = Statistics.apply_polynomial_formula(list(predicted_x), coefs)
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title=titles[n if n < 4 else 0], color=color)

    @staticmethod
    def power_plot(x_list: list, y_list: list = list()) -> None:
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.power_regression_model(x_list, y_list)
        predicted_y = Statistics.apply_power_formula(list(predicted_x), dict(a=coefs['a'], b=coefs['b']))
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title="Power Regression", color='green')

    @staticmethod
    def ab_exponential_plot(x_list: list, y_list: list) -> None:
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.ab_exponential_regression_model(x_list, y_list)
        predicted_y = Statistics.apply_ab_exponential_formula(list(predicted_x), dict(a=coefs['a'], b=coefs['b']))
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title="ab-Exponential Regression",
                               color='yellow')

    @staticmethod
    def hyperbolic_plot(x_list: list, y_list: list) -> None:
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.hyperbolic_regression_model(x_list, y_list)
        predicted_y = Statistics.apply_hyperbolic_formula(list(predicted_x), dict(a=coefs['a'], b=coefs['b']))
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title="Hyperbolic Regression", color='brown')

    @staticmethod
    def logarithmic_plot(x_list: list, y_list: list) -> None:
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.logarithmic_regression_model(x_list, y_list)
        predicted_y = Statistics.apply_logarithmic_formula(list(predicted_x), dict(a=coefs['a'], b=coefs['b']))
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title="Logarithmic Regression",
                               color='magenta')

    @staticmethod
    def exponential_plot(x_list: list, y_list: list) -> None:
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.exponential_regression_model(x_list, y_list)
        predicted_y = Statistics.apply_exponential_formula(list(predicted_x), dict(a=coefs['a'], b=coefs['b']))
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title="Exponential Regression", color='black')

    @staticmethod
    def plot_all(x_list: list, y_list: list) -> None:
        predicted_x = np.linspace(min(x_list), max(x_list), 100).tolist()
        coefs = Statistics.exponential_regression_model(x_list, y_list)
        predicted_y = Statistics.apply_exponential_formula(list(predicted_x), dict(a=coefs['a'], b=coefs['b']))
        Statistics.add_to_plot(predicted_x, predicted_y, x_list, y_list, title="Exponential Regression", color='black')


def to_str(x_list: list, y_list: list) -> str:
    sep = '\n' + '-' * 50
    string = 'Statistics Analyser Instance'
    string += sep
    string += '\n|arithmetic mean is ' + str(Statistics.get_arithmetic_mean(x_list, y_list))
    string += sep
    string += '\n|geometric mean is ' + str(Statistics.geometric_mean(x_list, y_list))
    string += sep
    string += '\n|harmonic mean is ' + str(Statistics.harmonic_mean(x_list, y_list))
    string += sep
    string += '\n|mode is ' + str(Statistics.get_mode(x_list, y_list))
    string += sep
    string += '\n|median is ' + str(Statistics.get_median(x_list, y_list))
    string += sep
    string += '\n|mean deviation is ' + str(Statistics.get_mean_deviation(x_list, y_list))
    string += sep
    string += '\n|variance is ' + str(Statistics.get_variance(x_list, y_list))
    string += sep
    string += '\n|standard deviation is ' + str(Statistics.get_standard_deviation(x_list, y_list))
    string += sep
    string += '\n|coefficient of variation is ' + str(Statistics.get_coefficient_of_variation(x_list, y_list))
    string += sep
    linear = Statistics.polynomial_regression_model(x_list, y_list, 1)
    string += '\n|coefficient of correlation is ' + str(linear['coc'])
    string += sep
    string += '\n|linear regression model formula is ' + str(linear['formula'])
    string += sep
    quadratic = Statistics.polynomial_regression_model(x_list, y_list, 2)
    string += '\n|quadratic coefficient of correlation is ' + str(quadratic['coc'])
    string += sep
    string += '\n|quadratic regression model formula is ' + str(quadratic['formula'])
    string += sep
    cubic = Statistics.polynomial_regression_model(x_list, y_list, 3)
    string += '\n|cubic coefficient of correlation is ' + str(cubic['coc'])
    string += sep
    string += '\n|cubic regression model formula is ' + str(cubic['formula'])
    string += sep
    power = Statistics.power_regression_model(x_list, y_list)
    string += '\n|power coefficient of correlation is ' + str(power['coc'])
    string += sep
    string += '\n|power regression model formula is ' + str(power['formula'])
    string += sep
    expo = Statistics.exponential_regression_model(x_list, y_list)
    string += '\n|exponential coefficient of correlation is ' + str(expo['coc'])
    string += sep
    string += '\n|exponential regression model formula is ' + str(expo['formula'])
    string += sep
    ab_expo = Statistics.ab_exponential_regression_model(x_list, y_list)
    string += '\n|ab-exponential coefficient of correlation is ' + str(ab_expo['coc'])
    string += sep
    string += '\n|ab-exponential regression model formula is ' + str(ab_expo['formula'])
    string += sep
    hyper = Statistics.hyperbolic_regression_model(x_list, y_list)
    string += '\n|hyperbolic coefficient of correlation is ' + str(hyper['coc'])
    string += sep
    string += '\n|hyperbolic regression model formula is ' + str(hyper['formula'])
    string += sep
    loga = Statistics.logarithmic_regression_model(x_list, y_list)
    string += '\n|logarithmic coefficient of correlation is ' + str(loga['coc'])
    string += sep
    string += '\n|logarithmic regression model formula is ' + str(loga['formula'])
    string += sep
    return string


if __name__ == '__main__':
    x = [11, 5, 18, 11, 30, 12, 65, 15, 46]
    y = [16, 16, 9, 13, 10, 15, 22, 18, 20]
    print(to_str(x, y))
    Statistics.polynomial_plot(x, y, 1)
    Statistics.polynomial_plot(x, y, 2, color='grey')
    Statistics.polynomial_plot(x, y, 3, color='purple')
    Statistics.polynomial_plot(x, y, 4, color='yellow')
    Statistics.ab_exponential_plot(x, y)
    Statistics.exponential_plot(x, y)
    Statistics.hyperbolic_plot(x, y)
    Statistics.logarithmic_plot(x, y)
    Statistics.power_plot(x, y)
    Statistics.plot("Regression")
