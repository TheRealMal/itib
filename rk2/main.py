import numpy as np
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class MultilayerNNetwork:
    def __init__(self, N, J, M, X, T, learning_rate, epsilon_cap) -> None:
        self.__N              = N
        self.__J              = J
        self.__M              = M
        self.__x_arr          = X
        self.__t_arr          = T
        self.__learning_rate  = learning_rate
        self.__eps_cap        = epsilon_cap
        self.__k              = 0
        
        self.__Y              = [0.0] * M

        self.__hidden_w       = [[0.5, -0.2]]
        self.__output_w       = [[0.5, 0], [0.5, -0.3], [0.5, -0.4]]

        self.__hidden_net     = [0.0] * J
        self.__output_net     = [0.0] * M

        self.__hidden_e       = [0.0] * J
        self.__output_e       = [0.0] * M

        self.__hidden_x       = [0] * (N + 1)
        self.__output_x       = [0] * J

    def activation_func(self, net) -> float:
        return 1 / (1 + np.exp(-net))
    def d_activation_func(self, net) -> float:
        return np.exp(net) / (np.exp(net) + 1) ** 2
    
    def solve_y(self) -> None:
        # I.1
        for _ in range(self.__N + 1):
            self.__hidden_x[_] = self.__x_arr[_]
        # I.2
        tmp_ = []
        for _ in range(self.__J):
            self.__hidden_net[_] = self.__hidden_w[_][0]
            tmp_.append("w1[0,{}]*1".format(_+1))
            for __ in range(self.__N):
                self.__hidden_net[_] += self.__hidden_w[_][__ + 1] * self.__hidden_x[__ + 1]
                tmp_.append("w1[{},{}]*x{}".format(__+1, _+1,__))
            print("net{}(1) = {} = {}".format(_+1, " + ".join(tmp_), np.round(self.__hidden_net[_], 3)))
        # I.3
        for _ in range(self.__J):
            self.__output_x[_] = self.activation_func(self.__hidden_net[_])
            print("x{} = f(net{}) = {}".format(_, _+1, np.round(self.__output_x[_], 3)))

        # I.4
        tmp_ = []
        for _ in range(self.__M):
            self.__output_net[_] = self.__output_w[_][0]
            tmp_.append("w0{}*1".format(_+1))
            for __ in range(self.__J):
                self.__output_net[_] += self.__output_w[_][__ + 1] * self.__output_x[__]
                tmp_.append("w2[{},{}]*x{}(2)".format(__+1, _+1,__))
            print("net{}(2) = {} = {}".format(_+1, " + ".join(tmp_), np.round(self.__output_net[_], 3)))

        # I.5
        for _ in range(self.__M):
            self.__Y[_] = self.activation_func(self.__output_net[_])
            print("y{} = f(net{}) = {}".format(_+1 , _+1, np.round(self.__Y[_], 3)))

    def rate_errors(self) -> None:
        # II.1
        for _ in range(self.__M):
            self.__output_e[_] = self.d_activation_func(self.__output_net[_]) * (self.__t_arr[_] - self.__Y[_])
            print("e{}(2) = f' * (t{} - y{}) = {}".format(_+1, _+1, _+1, np.round(self.__output_e[_], 3)))
        # II.2
        for _ in range(self.__J):
            hid_sum = 0
            for __ in range(self.__M):
                hid_sum += self.__output_w[__][_+1] * self.__output_e[__]
            self.__hidden_e[_] = self.d_activation_func(self.__hidden_net[_]) * hid_sum
            print("e{}(1) = f' * sum(w2[] * e(2)) = {}".format(_+1, np.round(self.__hidden_e[_], 3)))
    
    def weights_correction(self) -> None:
        # III.1
        for _ in range(len(self.__hidden_w)):
            for __ in range(len(self.__hidden_w[_])):
                self.__hidden_w[_][__] += self.__learning_rate * self.__hidden_x[__] * self.__hidden_e[_]
                print("w1[{},{}] += h * x{} * e{}(1) = {}".format(__, _+1, __, _+1, np.round(self.__hidden_w[_][__], 3)))
        # III.2
        self.__output_x = [1] + self.__output_x
        for _ in range(self.__M):
            for __ in range(self.__J + 1):
                self.__output_w[_][__] += self.__learning_rate * self.__output_x[__] * self.__output_e[_]
                print("w2[{},{}] += h * x{} * e{}(2) = {}".format(__, _+1, __, _+1, np.round(self.__output_w[_][__], 3)))

    def __print_epoch(self, E, ifEnd=0) -> None:
        print("Эпоха #{} | Ошибка E = %.7f".format(self.__k) % np.round(E, 7))
        print("Y = (", end="")
        for _ in range(len(self.__Y)):
            print("%.3f" % np.round(self.__Y[_], 3), end=", " * (_ != len(self.__Y) - 1))
        print(")")
        print("~" * 16)
        print()

    def start(self, err) -> None:
        while err > self.__eps_cap:
            self.solve_y()
            self.rate_errors()
            self.weights_correction()
            err = 0
            for _ in range(self.__M):
                err += (self.__t_arr[_] - self.__Y[_]) ** 2
            err = np.sqrt(err)
            self.__k += 1
        return True

    def solve(self) -> None:
        err = 1
        for _ in range(2):
            print("~" * 16)
            self.solve_y()
            self.rate_errors()
            self.weights_correction()

            err = 0
            for _ in range(self.__M):
                err += (self.__t_arr[_] - self.__Y[_]) ** 2
            err = np.sqrt(err)
            self.__print_epoch(err, 0)
            self.__k += 1

        print("\nПроверка на возможность обучения...")
        with suppress_stdout():
            if self.start(err):
                pass
        print("\nМожно обучить\n")
        


def main() -> None:
    # 8 Variant
    net = MultilayerNNetwork(
        N = 1, J = 1, M = 3,
        X = [1, -2],
        T = [2, 1, 3],
        learning_rate = 1
    )
    net.solve()

if __name__ == "__main__":
    main()