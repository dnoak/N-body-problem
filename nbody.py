import threading
import numpy as np
import cv2
from dataclasses import dataclass, field
import numba

G = 6.6743e-11

@dataclass(kw_only=True)
class Body:
    name: str
    m: float
    s: np.ndarray = field(default_factory=lambda: np.random.rand(2) * 1000)
    v: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    a: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    path_history: np.ndarray = None
    color: np.ndarray = field(default_factory=lambda: np.random.randint(0, 255, 3))

    def __repr__(self):
        s = f"({int(self.s[0])}, {int(self.s[1])})"
        v = f"{self.v[0]:.1f}, {self.v[1]:.1f}"
        a = f"{self.a[0]:.1f}, {self.a[1]:.1f}"
        return f"{self.name} - s: {s} v: {v} a: {a}"
    
    def create_path_array(self, path_size):
        self.path_history = np.zeros((path_size, 2), dtype=np.float32)
        self.path_history[0] = self.s
        return self

    @staticmethod
    @numba.njit
    def update_history_jit(path_history, s):
        new_history = np.zeros(path_history.shape, dtype=np.float32)
        new_history[1:] = path_history[:-1]
        new_history[0] = s
        return new_history
    
    @staticmethod
    @numba.njit
    def update_position_jit(a, v, s, body_s, body_m, t):
        dx = body_s[0] - s[0]
        dy = body_s[1] - s[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 30:
            distance = 30
        
        dxdy = np.array([dx, dy])
        direction = dxdy/np.abs(dxdy) #np.sign([dx, dy])

        a = ( body_m * G / distance ** 2 ) * direction
        v = v + a * t
        s = s + v * t
        return s, v, a
    
    def update_history(self):
        self.path_history = np.roll(self.path_history, 1, axis=0)
        self.path_history[0] = self.s
        return self

    def update_position(self, body, t):
        dx = body.s[0] - self.s[0]
        dy = body.s[1] - self.s[1]
        distance = np.sqrt(dx**2 + dy**2)
        direction = np.sign([dx, dy])
        self.a = ( body.m * G * 1 / distance ** 2 ) * direction
        self.v = self.v + self.a * t # if distance > 20 else np.zeros(2)
        self.s = self.s + self.v * t
    
    def update(self, body, t):
        self.s, self.v, self.a = self.update_position_jit(
            self.a, self.v, self.s, 
            body.s, body.m, t)
        self.path_history = self.update_history_jit(
            self.path_history, self.s)

@dataclass
class World:
    resolution: tuple
    bodies: list[Body]
    bodies_path_size: int = 1
    time_scale: int = 60
    
    def __post_init__(self):
        self.screen = 255 * np.ones((*self.resolution[::-1], 3), dtype=np.uint16)
        for body in self.bodies:
            body.create_path_array(self.bodies_path_size)

    def get_bodie_name(self):
        names = [-1]
        for name in [body.name for body in self.bodies]:
            if name.isdigit(): names.append(int(name))
        return max(names) + 1

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONUP:
            #with threading.Lock():
            self.bodies.append(
                Body(
                    name=str(self.get_bodie_name()),
                    m=np.random.randint(1e14, 1e16, dtype=np.uint64),
                    s=np.array([x, y]),
                ).create_path_array(self.bodies_path_size)
            )
        
        if event == cv2.EVENT_RBUTTONUP:
            #with threading.Lock():
            self.bodies = list(filter(
                lambda body: not (x-10 < body.s[0] < x+10 and y-10 < body.s[1] < y+10), 
                self.bodies))

        if event == cv2.EVENT_MBUTTONUP:
            #with threading.Lock():
            self.bodies = []

    def show(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.mouse_callback)
        cv2.imshow('image', self.screen.astype(np.uint8))
        cv2.waitKey(33)
    
    def render(self):
        self.screen.fill(255)
        for body in self.bodies:
            diff_to_255 = ( np.array([255, 255, 255]) - body.color )
            alpha_colors = np.tile(np.arange(
                0, 1, 1/self.bodies_path_size), (3, 1)
            ).T ** 1.5 * diff_to_255 + body.color
        
            for alpha, (x, y) in enumerate(body.path_history[1:]):
                #body_color = body.color + color_alpha * alpha
                cv2.circle(self.screen, (int(x), int(y)), 1, alpha_colors[alpha], -1)
                # cv2.circle(self.screen, (int(x), int(y)), 1, body_color.tolist(), -1)
            
            x, y = int(body.s[0]), int(body.s[1])
            cv2.circle(self.screen, (x, y), 12, body.color.tolist(), -1)
            cv2.putText(self.screen, body.name, (x-10, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        self.show()

    def is_inside(self, body):
        x_condition = 0 <= body.path_history[-1][0] <= self.resolution[0]
        y_condition = 0 <= body.path_history[-1][1] <= self.resolution[1]
        return x_condition and y_condition

    def update(self):
        t = self.time_scale
        #with threading.Lock():
        for body in self.bodies:
            for another_bodie in self.bodies:
                if another_bodie != body:
                    body.update(another_bodie, t)
        self.bodies = list(filter(
            lambda body: self.is_inside(body), 
            self.bodies))
        #print(f'Body count: {len(self.bodies)}')
       

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
        *[Body(name=f'{name}', m=10e14) for name in range(1, 20)],
        Body(name='0', m=10e15, s=np.array([800, 400])),
        # Body(name='1', m=30e15, s=np.array([800, 400]), color=(0, 255, 255)),
        # Body(name='2', m=10e15, s=np.array([700, 400]), v=np.array([0, 450]), color=(255, 0, 255)),
        # Body(name='3', m=10e15, s=np.array([900, 400]), v=np.array([0, -450]), color=(255, 0, 255)),
    ],
    bodies_path_size=700,
    time_scale=1/60,
).start()

