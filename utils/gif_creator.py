import imageio
import matplotlib.image as mpimg
import numpy as np


def get_image_list(path, num):
    img_list = []
    for i in range(num):
        img_list.append(mpimg.imread(path + '\\fake_' + str(i) + '.png'))
    return img_list


def create_gif(image_list, path, duration=0.5):
    # image_list = [(img * 256).astype(np.uint8) for img in image_list]
    imageio.mimsave(path + "\\result.gif", image_list, 'GIF', duration=duration)
