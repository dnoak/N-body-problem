# N-Body Problem Simulation

Simulação do problema N-Body usando Python e OpenCV.

## Descrição

Este é um programa simples que simula o problema N-Body, onde múltiplos corpos celestes interagem gravitacionalmente uns com os outros. A simulação utiliza a biblioteca OpenCV para renderizar visualmente o movimento dos corpos em uma janela.

## Requisitos

- Python 3.x
- Bibliotecas: NumPy, OpenCV

Você pode instalar as bibliotecas necessárias utilizando o seguinte comando:

```bash
pip install numpy opencv-python
```

## Uso

Execute o script `nbody_simulation.py` para iniciar a simulação. Você pode ajustar os parâmetros e adicionar/remover corpos no bloco de inicialização no final do script.

```python
world = World(
    resolution=(1600, 800),
    bodies=[
        Body(name='1', m=30e15, s=np.array([800, 400]), color=(0, 255, 255)),
        Body(name='2', m=10e15, s=np.array([700, 400]), v=np.array([0, 450]), color=(255, 0, 255)),
        Body(name='3', m=10e15, s=np.array([900, 400]), v=np.array([0, -450]), color=(255, 0, 255)),
    ],
    world_scale=10e-2,
    framerate=30,
).start()
```

## Personalização

- `Body`: Representa um corpo celeste. Você pode personalizar a massa (`m`), posição inicial (`s`), velocidade inicial (`v`), aceleração (`a`), e cor (`color`).

- `World`: Representa o ambiente da simulação. Personalize a resolução da tela (`resolution`), a lista de corpos (`bodies`), escala do mundo (`world_scale`), e a taxa de quadros por segundo (`framerate`).
