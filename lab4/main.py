import numpy as np

def generate_sets(n) -> list:
    from itertools import product as pr
    return list(pr([0, 1], repeat=n))

class NNet:
    def __init__(self, n = 0.3) -> None:
        self.__learning_rate      = n                         # Норма обучения
        self.__k                  = 0                         # Эпохи
        self.__center_coords      = self.find_center_coords() # Скрытых RBF-Нейронов
        self.__J                  = len(self.__center_coords)
        self.__weights            = np.zeros(self.__J + 1)    # Синаптические веса

    def find_center_coords(self) -> int:
        J0, J1 = [], []
        for _ in generate_sets(4):
            if self.logic_function(_):
                J1.append(_)
            else:
                J0.append(_)
        if len(J0) < len(J1):
            return J0
        return J1

    def logic_function(self, v) -> bool:
        # 8 Variant
        return (v[0] or v[1] or v[3]) and v[2]
        # Example
        #return not (v[0] and v[1]) and v[2] and v[3]
        # Ram
        #return not v[0] and v[1] and (not v[2] or v[3])
    
    def phi_function(self, x, c) -> float:
        return np.exp(-np.sum((np.array(x) - np.array(c)) ** 2))
    
    def net(self, x) -> float:
        net = self.__weights[0]
        for c, w in zip(self.__center_coords, self.__weights[1:]):
            net += w * self.phi_function(x, c)
        return net
    
    #
    # Функции активации
    #
    def lg2_activation_func(self, net) -> int:
        return ((net / (1 + abs(net)) + 1)) / 2
    
    def lg3_activation_func(self, net) -> int:
        return 1 / (1 + np.exp(-net))
    
    def lg4_activation_func(self, net) -> int:
        return (np.tanh(net) + 1) / 2

    def th_activation_func(self, net) -> int:
        return int(net >= 0)
    
    def activation_func(self, net, c = 0) -> int:
        if c == 0:
            return self.th_activation_func(net)
        elif c == 1:
            return self.th_activation_func(net)
        elif c == 2:
            return self.lg2_activation_func(net)
        elif c == 3:
            return self.lg3_activation_func(net)
        elif c == 4:
            return self.lg4_activation_func(net)
    # --------------------------------------------

    def get_ham_dist(self, learning_selection, current_selection) -> float:
        target_values = []
        for _ in learning_selection:
            target_values.append(self.logic_function(_))
        return np.count_nonzero(np.array(current_selection) != np.array(target_values))

    def build_RBF_model(self, learning_selection):
        while True:
            out_vector = [] # Выходной вектор текущей эпохи
            for x in learning_selection:
                net = self.net(x) 
                y = self.activation_func(net, 0) # Выбрать функцию активации (Второй аргумент)
                out_vector.append(y)
                error = self.logic_function(x) - y
                phi_arr = [1] + [self.phi_function(x, c) for c in self.__center_coords]
                delta = self.__learning_rate * error * np.array(phi_arr)
                self.__weights += np.round(delta, 3)
            ham_dist = self.get_ham_dist(learning_selection, out_vector)
            self.__print_epoch(out_vector, ham_dist, int(ham_dist == 0))
            self.__k += 1

            if ham_dist == 0:
                break


    def __print_epoch(self, out_vector, E, ifEnd=0) -> None:
        print("~" * 16)
        print("Эпоха #{} | Ошибка E = {}".format(self.__k, E))
        print("W = (", end="")
        for _ in range(len(self.__weights)):
            print("%.3f" % np.round(self.__weights[_], 3), end=", " * (_ != len(self.__weights) - 1))
        print(")")
        print("Y = (", end="")
        for _ in range(len(out_vector)):
            print(out_vector[_], end=", " * (_ != len(out_vector) - 1))
        print(")")
        print("~" * (16 * ifEnd), end="\n"*ifEnd)


def main():
    tmp = NNet(0.3)
    tmp.build_RBF_model(generate_sets(4))
    # 8 Variant
    #tmp.build_RBF_model([[0,0,0,0],[0,0,1,0],[0,1,1,0],[1,0,1,1],[1,1,0,1]])
    # Example
    #tmp.build_RBF_model([[0,0,0,1],[0,1,1,1],[1,0,1,0],[1,0,1,1],[1,1,1,0]])
    # Ram
    #tmp.build_RBF_model([[0,0,1,0],[0,1,0,1]])

if __name__ == "__main__":
    main()