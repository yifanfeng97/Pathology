from PIL import Image
import numpy as np
from api import head_map_fun
import matplotlib



# file_dir = '/media/fengyifan/16F8F177F8F15589/RJPathData/Experiments6_bk/prob_map/bk/4011_p_map_img.txt'
file_dir = '/home/duanqi01/Documents/Testfyf/Experiments_SVS1/prob_map/bk1/7070_p_map_img.txt'

p_map = np.loadtxt(file_dir)

heat_map = head_map_fun.get_heat_map_from_prob(p_map)
heat_map.show()

# # plt.figure(figsize=(p_map.shape[1], p_map.shape[0]))
# plt.imshow(p_map, cmap='jet')
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# plt.savefig('test.png', bbox_inches='tight')
# # plt.show()
# p_map_img = Image.fromarray(p_map*255)
# p_map_img.show()
# p_map_img = p_map_img.convert('RGB')
# p_map_img.show()
# print(p_map.shape)