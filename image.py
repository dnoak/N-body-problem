import cv2
import numpy as np
from PIL import Image

imagem = np.zeros((10, 10, 3), dtype=np.uint8)

############### imagem[ X, Y ] ###############

imagem[0, 0] = [255, 0, 0]
imagem[1, 2] = [255, 0, 0]
imagem[1, 2] = [255, 0, 0]
imagem[2, 4] = [255, 0, 0]
imagem[2, 4] = [255, 0, 0]
imagem[3, 6] = [255, 0, 0]
imagem[3, 6] = [255, 0, 0]
imagem[4, 8] = [255, 0, 0]
imagem[4, 8] = [255, 0, 0]
imagem[55, 9] = [255, 0, 0]


pillow_image = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
pillow_image.show()