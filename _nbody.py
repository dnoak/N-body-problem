import numpy as np 
import cv2
from dataclasses import dataclass

@dataclass
class Body:
    c: tuple
    r: float
    m: float
    x: float
    y: float
    vx: float = 0
    vy: float = 0

    def draw(self, frame):
        coords = (int(self.x), int(self.y))
        circle_image = cv2.circle(frame.copy(), coords, int(self.r), self.c, -1)
        return circle_image, coords
    
    def update(self, bodies, t):
        ...

@dataclass
class World:
    size: np.ndarray
    bodies: list[Body]
    frame_rate: 60

    def __post_init__(self):
        self.frame = 255 * np.ones((self.size[1], self.size[0], 3), dtype=np.uint8)

    def show(self):
        cv2.imshow('image', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw(self):
        #self.frame.fill(255)
        for body in self.bodies:
            self.frame = body.draw(self.frame)

    def update(self):
        for body in self.bodies:
            body.update([b for b in self.bodies if b != body], 1/self.frame_rate)
        print('-'*20)

    def start(self):
        while True:
            self.draw()
            self.show()
            self.update()


h, w = 600, 600
world = World(
    size=np.array([w, h]),
    bodies=[
        Body(c=(0, 0, 255), r=30, m=10e14, x=200, y=200, vx=100, vy=30),
        Body(c=(0, 255, 0), r=30, m=10e14, x=400, y=400, vx=-100, vy=30),
    ],
    frame_rate=10,
).start()


# TEM ACELERAÇÃO DA ACELERAÇÃO
