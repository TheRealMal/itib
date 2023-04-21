import numpy as np

SAVED_IMAGES = [
    [-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1],
    [1,-1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1],
    [1,-1,1,-1,1,1,-1,1,-1,1,1,1,1,1,1]
]

class RecurrentNNetwork:
    def __init__(self, images) -> None:
        self.__images     = images                      # X
        self.__L          = len(images)                 # Количество образов
        self.__K          = len(images[0])              # Длина образа
        self.__weights    = self.__set_weights(images)  # W
        self.__previous_y = [[0] * self.__K, [0] * self.__K, [0] * self.__K, [0] * self.__K]
        self.__net        = [0] * self.__K              # net
        self.__epochs     = 0                           # Эпохи

        self.__pretty_print_weights_matrix()

    def __clear(self) -> None:
        self.__previous_y = [[0] * self.__K, [0] * self.__K, [0] * self.__K, [0] * self.__K]
        self.__net        = [0] * self.__K
        self.__epochs     = 0

    def __set_weights(self, images) -> list:
        weights = np.zeros((self.__K, self.__K))
        for _ in range(self.__K):
            for __ in range(self.__K):
                if _ == __:
                    continue
                for ___ in range(self.__L):
                    weights[_][__] += images[___][_] * images[___][__]
        
        return weights
    
    def __pretty_print_weights_matrix(self) -> None:
        print("W =")
        print('\n'.join(['\t'.join([str(int(cell)) for cell in row]) for row in self.__weights]))

    def activation_func(self, i) -> None:
        if self.__net[i] > 0:
            self.__previous_y[3][i] = 1
        elif self.__net[i] < 0:
            self.__previous_y[3][i] = -1
    
    def move_y(self) -> None:
        self.__previous_y[0], self.__previous_y[1], self.__previous_y[2] = self.__previous_y[1].copy(), self.__previous_y[2].copy(), self.__previous_y[3].copy()

    def check_cycling(self, image) -> (int, str):
        if self.__previous_y[0] == self.__previous_y[2] and self.__previous_y[1] == self.__previous_y[3]:
            return 1, "Зацикливание двух образов"
        elif image == self.__previous_y[3]:
            return 2, "Входной образ = выходной образ"
        elif self.__previous_y[3] == self.__previous_y[2]:
            return 3, "Входной образ = выходной образ v2"
        return 0, ""

    def sync_mode(self) -> None:
        for _ in range(self.__K):
            self.__net[_] = 0
            for __ in range(self.__K):
                if __ == _: continue
                self.__net[_] += self.__weights[__][_] * self.__previous_y[3][__]
        for _ in range(self.__K):
            self.activation_func(_)

    def async_mode(self) -> None:
        for _ in range(self.__K):
            self.__net[_] = 0
            for __ in range(self.__K):
                if __ == _: continue
                self.__net[_] += self.__weights[__][_] * self.__previous_y[3][__]
            self.activation_func(_)

    def __print_epoch(self) -> None:
        print("~" * 16)
        print("Эпоха #{}".format(self.__epochs))
        print("Y  = (", end="")
        for _ in range(len(self.__previous_y[3])):
            print(self.__previous_y[3][_], end=", " * (_ != len(self.__previous_y[3]) - 1))
        print(")")

    def recover_image(self, image, mode) -> bool:
        self.__clear()
        self.__previous_y[3] = image.copy()
        while True:
            self.__epochs += 1
            if mode == "sync":
                self.sync_mode()
            elif mode == "async":
                self.async_mode()
            self.__print_epoch()
            for _ in range(len(self.__images)):
                if self.__images[_] == self.__previous_y[3]:
                    print("Y' = (", end="")
                    for ___ in range(len(self.__previous_y[3])):
                        print(image[___], end=", " * (___ != len(image) - 1))
                    print(")")
                    return True
                
            check_result, check_mes = self.check_cycling(image)
            if check_result:
                print("Y' = (", end="")
                for ___ in range(len(self.__previous_y[3])):
                    print(image[___], end=", " * (___ != len(image) - 1))
                print(")")
                print("Невозможно распознать образ\n{}".format(check_mes))
                return False
            self.move_y()


def main():
    # Example
    net = RecurrentNNetwork(SAVED_IMAGES)
    net.recover_image([-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1], "sync")

if __name__ == "__main__":
    main()
