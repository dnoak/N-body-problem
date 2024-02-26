import numpy as np
import cv2
from dataclasses import dataclass, field
import random

G = 6.6743e-11

@dataclass
class Body:
    name: str
    m: float
    s: np.ndarray = field(default_factory=lambda: np.random.rand(2) * 1000)
    v: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    a: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    f: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    world_scale: float = 1
    color: tuple = field(default_factory=lambda: tuple(random.choices(range(256), k=3)))

    def __repr__(self):
        s = f"({int(self.s[0])}, {int(self.s[1])})"
        v = f"{self.v[0]:.1f}, {self.v[1]:.1f}"
        a = f"{self.a[0]:.1f}, {self.a[1]:.1f}"
        return f"{self.name} - s: {s} v: {v} a: {a}"

    def update(self, body, t):
        #for body in bodies:
        dx = body.s[0] - self.s[0]
        dy = body.s[1] - self.s[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 20:
            distance = 20
        direction = np.sign([dx, dy])
        self.f = ( body.m * self.m * G * 1 / distance ** 2 ) * direction
        self.a = self.f / self.m
        self.v = self.v + self.a * t
        self.s = self.s + self.v * t * self.world_scale


@dataclass
class World:
    resolution: tuple
    bodies: list[Body]
    #cm: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.float32))
    world_scale: float = 1
    framerate: int = 60
    
    def __post_init__(self):
        self.screen = 255 * np.ones((*self.resolution[::-1], 3), dtype=np.uint8)
        for body in self.bodies:
            body.world_scale = self.world_scale

    def show(self):
        cv2.imshow('image', self.screen)
        cv2.waitKey(33)
    
    def render(self):
        self.screen.fill(255)
        for body in self.bodies:
            x, y = int(body.s[0]), int(body.s[1])
            cv2.circle(self.screen, (x, y), 12, body.color, -1)
            cv2.putText(self.screen, body.name, (x-10, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        self.show()

    def update(self):
        t = 1 / self.framerate
        for body in self.bodies:
            another_bodies = [
                body.update(another_bodie, t) for another_bodie in self.bodies 
                if another_bodie != body
            ]
            #body.update(another_bodies, t)
        
    def start(self):
        while True:
            self.update()
            self.render()

world = World(
    resolution=(1600, 800),
    bodies=[
        # Body(name='1', m=10e15, s=np.array([300, 300]), color=(0, 255, 255)),
        # Body(name='2', m=10e15, s=np.array([700, 300]), color=(255, 0, 100)),
        # Body(name='3', m=10e15, s=np.array([1200, 600]), color=(0, 255, 100)),
        # *[Body(name=f'{name}', m=10e14) for name in range(1, 10)],
        Body(name='1', m=30e15, s=np.array([800, 400]), color=(0, 255, 255)),
        Body(name='2', m=10e15, s=np.array([700, 400]), v=np.array([0, 450]), color=(255, 0, 255)),
        Body(name='3', m=10e15, s=np.array([900, 400]), v=np.array([0, -450]), color=(255, 0, 255)),
    ],
    world_scale=10e-2,
    framerate=30,
).start()

        