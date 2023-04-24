from tkinter import *
from Clusters import KMeans
import numpy as np
import random

OBJECTS_COORDS = []
CLUSTERS_COORDS = []
DISTANCE_FUNC = 0

def random_color():
    rand = lambda: random.randint(80, 200)
    return '#%02X%02X%02X' % (rand(), rand(), rand())

class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = '#000000'
    def __init__(self):
        self.root = Tk()
        self.root.title("LAB9 by therealmal")
        self.add_point_b = Button(self.root, text='Add point', command=self.use_point)
        self.add_point_b.grid(row=0, column=0)

        self.add_cluster_b = Button(self.root, text='Add cluster', command=self.use_cluster)
        self.add_cluster_b.grid(row=0, column=1)

        self.clear_b = Button(self.root, text='Clear', command=self.clear)
        self.clear_b.grid(row=0, column=2)

        self.start_b = Button(self.root, text='Start', command=self.start)
        self.start_b.grid(row=0, column=3)
        
        self.start_b = Button(self.root, text='One Step', command=self.step)
        self.start_b.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def start(self):
        global OBJECTS_COORDS, CLUSTERS_COORDS, DISTANCE_FUNC
        solution = KMeans(OBJECTS_COORDS.copy(), CLUSTERS_COORDS.copy(), DISTANCE_FUNC)
        solution.start()

        for coord in CLUSTERS_COORDS:
            self.c.create_line(
                coord[0], coord[1], coord[0] + 1, coord[1],
                width=self.DEFAULT_PEN_SIZE * 2 + 2, fill="#FFFFFF",
                capstyle=ROUND
            )
        colors = [random_color() for _ in range(len(solution.clusters))]
        for coord_i in range(len(solution.clusters)):
            coord = np.round(solution.clusters[coord_i])
            self.c.create_line(
                coord[0], coord[1], coord[0] + 1, coord[1],
                width=self.DEFAULT_PEN_SIZE * 2, fill=colors[coord_i],
                capstyle=ROUND
            )
        for obj_group in range(len(solution.groups)):
            for obj_i in solution.groups[obj_group]:
                coord = np.round(solution.objects[obj_i])
                self.c.create_line(
                    coord[0], coord[1], coord[0] + 1, coord[1],
                    width=self.DEFAULT_PEN_SIZE, fill=colors[obj_group],
                    capstyle=ROUND
                )
        CLUSTERS_COORDS = solution.clusters

    def step(self):
        global OBJECTS_COORDS, CLUSTERS_COORDS, DISTANCE_FUNC
        solution = KMeans(OBJECTS_COORDS.copy(), CLUSTERS_COORDS.copy(), DISTANCE_FUNC)
        solution.step()

        for coord in CLUSTERS_COORDS:
            self.c.create_line(
                coord[0], coord[1], coord[0] + 1, coord[1],
                width=self.DEFAULT_PEN_SIZE * 2 + 2, fill="#FFFFFF",
                capstyle=ROUND
            )
        colors = [random_color() for _ in range(len(solution.clusters))]
        for coord_i in range(len(solution.clusters)):
            coord = np.round(solution.clusters[coord_i])
            self.c.create_line(
                coord[0], coord[1], coord[0] + 1, coord[1],
                width=self.DEFAULT_PEN_SIZE * 2, fill=colors[coord_i],
                capstyle=ROUND
            )
        for obj_group in range(len(solution.groups)):
            for obj_i in solution.groups[obj_group]:
                coord = np.round(solution.objects[obj_i])
                self.c.create_line(
                    coord[0], coord[1], coord[0] + 1, coord[1],
                    width=self.DEFAULT_PEN_SIZE, fill=colors[obj_group],
                    capstyle=ROUND
                )
        CLUSTERS_COORDS = solution.clusters

    def setup(self):
        self.line_width = self.DEFAULT_PEN_SIZE
        self.active_button = self.add_point_b
        self.c.bind('<Button-1>', self.paint)

    def use_point(self):
        self.line_width = self.DEFAULT_PEN_SIZE
        self.active_button.config(relief=RAISED)
        self.add_point_b.config(relief=SUNKEN)
        self.active_button = self.add_point_b

    def use_cluster(self):
        self.line_width = self.DEFAULT_PEN_SIZE * 2
        self.active_button.config(relief=RAISED)
        self.add_cluster_b.config(relief=SUNKEN)
        self.active_button = self.add_cluster_b

    def paint(self, event):
        self.c.create_line(
            event.x, event.y, event.x + 1, event.y,
            width=self.line_width, fill=self.DEFAULT_COLOR,
            capstyle=ROUND
        )

        if self.active_button == self.add_cluster_b:
            CLUSTERS_COORDS.append([event.x, event.y])
        else:
            OBJECTS_COORDS.append([event.x, event.y])

    def clear(self):
        global OBJECTS_COORDS, CLUSTERS_COORDS
        for coord in CLUSTERS_COORDS:
            self.c.create_line(
                coord[0], coord[1], coord[0] + 1, coord[1],
                width=self.DEFAULT_PEN_SIZE * 2 + 2, fill="#FFFFFF",
                capstyle=ROUND
            )
        for coord  in OBJECTS_COORDS:
            self.c.create_line(
                coord[0], coord[1], coord[0] + 1, coord[1],
                width=self.DEFAULT_PEN_SIZE + 2, fill="#FFFFFF",
                capstyle=ROUND
            )
        OBJECTS_COORDS = []
        CLUSTERS_COORDS = []


if __name__ == '__main__':
    Paint()