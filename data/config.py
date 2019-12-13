# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~/Projects/SSD")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (211, 221, 225)

# SSD512 CONFIGS
sixray = {
    'num_classes': 3,
    'lr_steps': (60000, 68000, 74000),
    'max_iter': 74000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [51, 102, 189, 276, 363, 450],
    'max_sizes': [102, 189, 276, 363, 450, 537],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SIXRAY',
}
