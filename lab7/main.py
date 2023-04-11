import numpy as np

class RecurrentNNetwork:
    def __init__(self, images) -> None:
        self.__images     = images                      # X
        self.__L          = len(images)                 # Количество образов
        self.__K          = len(images[0])              # Длина образа
        self.__weights    = self.__set_weights(images)  # W
        self.__previous_y = [0] * self.__K              # Y
        self.__net        = [0] * self.__K              # net
        self.__epochs     = 0                           # Эпохи

    def __clear(self) -> None:
        self.__previous_y = [0] * self.__K
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
    
    def activation_func(self, i) -> None:
        if self.__net[i] > 0:
            self.__previous_y[i] = 1
        elif self.__net[i] < 0:
            self.__previous_y[i] = -1
    
    def sync_mode(self) -> None:
        for _ in range(self.__K):
            self.__net[_] = 0
            for __ in range(self.__K):
                if __ == _: continue
                self.__net[_] += self.__weights[__][_] * self.__previous_y[__]
        for _ in range(self.__K):
            self.activation_func(_)

    def async_mode(self) -> None:
        for _ in range(self.__K):
            self.__net[_] = 0
            for __ in range(self.__K):
                if __ == _: continue
                self.__net[_] += self.__weights[__][_] * self.__previous_y[__]
            self.activation_func(_)

    def __print_epoch(self) -> None:
        print("~" * 16)
        print("Эпоха #{}".format(self.__epochs))
        print("Y  = (", end="")
        for _ in range(len(self.__previous_y)):
            print(self.__previous_y[_], end=", " * (_ != len(self.__previous_y) - 1))
        print(")")

    def recover_image(self, image, mode):
        self.__clear()
        self.__previous_y = image.copy()
        while True:
            self.__epochs += 1
            if mode == "sync":
                self.sync_mode()
            elif mode == "async":
                self.async_mode()
            self.__print_epoch()
            for _ in range(len(self.__images)):
                if self.__images[_] == self.__previous_y:
                    print("Y' = (", end="")
                    for ___ in range(len(self.__previous_y)):
                        print(image[___], end=", " * (___ != len(image) - 1))
                    print(")")
                    return

def main():
    '''net = RecurrentNNetwork([
        [-1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,1,1],
        [1,1,1,1,1,-1,-1,1,-1,-1,1,1,-1,1,1],
        [1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1]
    ])
    net.recover_image([1,1,1,1,1,-1,-1,1,-1,-1,1,1,-1,1,1], "async")'''

    # Example
    net = RecurrentNNetwork([
        [-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1],
        [1,-1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1],
        [1,-1,1,-1,1,1,-1,1,-1,1,1,1,1,1,1]
    ])
    net.recover_image([-1,-1,1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1], "sync")
if __name__ == "__main__":
    main()