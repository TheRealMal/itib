import numpy as np

class Clusters:
    def __init__(self, objects, clusters, d_func) -> None:
        self._d_funcs = [self._euclidean_d, self._chebyshev_d, self._taxicab_d]
        self.objects = objects
        self.clusters = clusters
        self._d_func = d_func
        self._distances = self._get_distances()
        self.groups = self._get_groups()

    def _euclidean_d(self, a, b) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    
    def _chebyshev_d(self, a, b) -> float:
        return np.max(abs(a[0] - b[0]), abs(a[1], b[1]))
    
    def _taxicab_d(self, a, b) -> float:
        return abs(a[0] - b[0]) + abs(a[1], b[1])
    
    def _distance(self, i, j, f_) -> float:
        return self._d_funcs[f_](self.objects[i], self.clusters[j])
    
    def _get_distances(self) -> list:
        distances = []
        for _ in range(len(self.objects)):
            distances.append([])
            for __ in range(len(self.clusters)):
                distances[_].append(self._distance(_, __, self._d_func))
        return distances
    
    def _get_groups(self) -> list:
        groups = [[] for _ in range(len(self.clusters))]
        for obj_group_i in range(len(self._distances)):
            groups[np.argmin(self._distances[obj_group_i])].append(obj_group_i)
        return groups
    
    def _print(self) -> None:
        print("Objects:\n {}".format(self.objects))
        print("Clusters:\n {}".format(self.clusters))
        print("Groups:\n {}".format(self.groups))
        print()

class KMeans(Clusters):
    def __init__(self, objects, clusters, d_func):
        super().__init__(objects, clusters, d_func)

    def start(self):
        while True:
            old_clusters = self.clusters.copy()
            self.step()
            Clusters._print(self)
            if old_clusters == self.clusters:
                break

    def step(self) -> list:
        for _ in range(len(self.clusters)):
            if len(self.groups[_]) > 0:
                mean = [0, 0]
                for obj_id in self.groups[_]:
                    mean[0] += self.objects[obj_id][0]
                    mean[1] += self.objects[obj_id][1]
                self.clusters[_] = [mean[0] / len(self.groups[_]), mean[1] / len(self.groups[_])]
        self._distances = Clusters._get_distances(self)
        self.groups = Clusters._get_groups(self)

def main() -> None:
    a = KMeans([[0,0], [1,1], [2,2], [3,3]], [[2,2], [3,3]], 0)
    a.start()

if __name__ == "__main__":
    main()