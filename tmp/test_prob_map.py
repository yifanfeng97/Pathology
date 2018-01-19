from PIL import Image
import numpy as np
file_dir = '/media/fengyifan/16F8F177F8F15589/RJPathData/Experiments6_bk/prob_map/bk/4011_p_map_img.txt'

p_map = np.loadtxt(file_dir)

p_map_img = Image.fromarray(p_map*255)
p_map_img.show()
p_map_img = p_map_img.convert('RGB')
p_map_img.show()
print(p_map.shape)