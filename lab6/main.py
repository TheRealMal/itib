import numpy as np

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

        self.__hidden_w       = [[0.0 for __ in range(N + 1)] for _ in range(J)]
        self.__output_w       = [[0.0 for __ in range(J + 1)] for _ in range(M)]

        self.__hidden_net     = [0.0] * J
        self.__output_net     = [0.0] * M

        self.__hidden_e       = [0.0] * J
        self.__output_e       = [0.0] * M

        self.__hidden_x       = [0] * (N + 1)
        self.__output_x       = [0] * (J + 1)

    def activation_func(self, net) -> float:
        return (1 - np.exp(-net)) / (1 + np.exp(-net))
    def d_activation_func(self, net) -> float:
        return (1 - self.activation_func(net) ** 2) / 2
    
    def solve_y(self) -> None:
        # I.1
        for _ in range(self.__N + 1):
            self.__hidden_x[_] = self.__x_arr[_]
        # I.2
        for _ in range(self.__J):
            self.__hidden_net[_] = self.__hidden_w[_][0]
            for __ in range(self.__N):
                self.__hidden_net[_] += self.__hidden_w[_][__ + 1] * self.__hidden_x[__]
        # I.3
        for _ in range(self.__J):
            self.__output_x[_] = self.activation_func(self.__hidden_net[_])

        # I.4
        for _ in range(self.__M):
            self.__output_net[_] = self.__output_w[_][0]
            for __ in range(self.__J):
                self.__output_net[_] += self.__output_w[_][__ + 1] * self.__output_x[__]

        # I.5
        for _ in range(self.__M):
            self.__Y[_] = self.activation_func(self.__output_net[_])

    def rate_errors(self) -> None:
        # II.1
        for _ in range(self.__M):
            self.__output_e[_] = self.d_activation_func(self.__output_net[_]) * (self.__t_arr[_] - self.__Y[_])
        # II.2
        for _ in range(self.__J):
            shitSum = 0
            for __ in range(self.__M):
                shitSum += self.__hidden_w[_][__] * self.__output_e[__]
            self.__hidden_e[_] = self.d_activation_func(self.__hidden_net[_]) * shitSum
    
    def weights_correction(self) -> None:
        # III.1
        for _ in range(len(self.__hidden_w)):
            for __ in range(len(self.__hidden_w[_])):
                self.__hidden_w[_][__] += self.__learning_rate * self.__hidden_x[__] * self.__hidden_e[_]
        # III.2
        for _ in range(len(self.__output_w)):
            for __ in range(len(self.__output_w[_])):
                self.__output_w[_][__] += self.__learning_rate * self.__output_x[__] * self.__output_e[_]

    def __print_epoch(self, E, ifEnd=0) -> None:
        print("~" * 16)
        print("Эпоха #{} | Ошибка E = %.7f".format(self.__k) % np.round(E, 7))
        print("Y = (", end="")
        for _ in range(len(self.__Y)):
            print("%.3f" % np.round(self.__Y[_], 3), end=", " * (_ != len(self.__Y) - 1))
        print(")")
        print("~" * (16 * ifEnd), end="\n"*ifEnd)

    def start(self) -> None:
        err = 1
        while err > self.__eps_cap:
            self.solve_y()
            self.rate_errors()
            self.weights_correction()

            err = 0
            for _ in range(self.__M):
                err += (self.__t_arr[_] - self.__Y[_]) ** 2
            err = np.sqrt(err)

            self.__print_epoch(err, err <= self.__eps_cap)
            self.__k += 1


def main():
    tmp = MultilayerNNetwork(
        N = 3, J = 3, M = 4,
        X = [1, 0.3, -0.1, 0.9],
        T = [0.1, -0.6, 0.2, 0.7],
        learning_rate = 1,
        epsilon_cap = 0.001
    )
    tmp.start()

if __name__ == "__main__":
    main()