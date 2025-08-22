import os
from os.path import join
import numpy as np

from datalink.cpi_optaxial_tracking.moving_patterns import BigStepForward_SmallStepBack, SinusoidalForward, Refocused_range, shift

"""
Run in the parent folder of neural_network.
Preparing a data path with corresponding expected distance to the focal plane.
Divide the training data and validation data.
"""


pattern_p = {
'f14_b4':{
  'spatial': 'dataset/datalink/f14_b4/data/spatial'},
'f16_b8':{
  'spatial': 'dataset/datalink/f16_b8/data/spatial'},
# 'f16_b8_x_axis':{
#   'spatial': dataset/datalink/f16_b8_x_axis/data/spatial},
'fixed_acquisition':{
  'spatial': 'dataset/datalink/fixed_acquisition/data/spatial'},
'non_stop':{
  'spatial': 'dataset/datalink/non_stop/data/spatial'},
'sinusoidal':{
  'spatial': 'dataset/datalink/sinusoidal/data/spatial'}
}

data_patterns = {
    'f14_b4': Refocused_range(shift, BigStepForward_SmallStepBack(0, 17, pattern=np.array((14, -4))).pos_frames()['pos']).bigf_smallb()[1],
    'f16_b8': Refocused_range(shift, BigStepForward_SmallStepBack(0, 17, pattern=np.array((16, -8))).pos_frames()['pos']).bigf_smallb()[1],
    'fixed_acquisition': 6.87 - 0.25 * np.array(range(68)),
    'non_stop': 6.87 - 0.5 * np.array(range(34)),
    'sinusoidal': Refocused_range(shift, SinusoidalForward(0, 17, 90, frequency=3, amp=3).pos_frames()['pos']).sinusoidal()[1]}

train_pattern_file_paths = dict()
val_pattern_file_paths = dict()

for pattern in pattern_p.keys():
    file_names = sorted(os.listdir(pattern_p[pattern]['spatial']))
    train_ind = int(0.8 * len(file_names))
    train_paths = []
    val_paths = []
    for ind, name in enumerate(file_names):
        if ind < train_ind:
            train_paths.append(join(pattern_p[pattern]['spatial'], name))
        else:
            val_paths.append(join(pattern_p[pattern]['spatial'], name))

    train_pattern_file_paths[pattern] = train_paths
    val_pattern_file_paths[pattern] = val_paths

with open('./dataset/train.txt', 'w') as f:
    for pattern in train_pattern_file_paths.keys():
        for ind, path in enumerate(train_pattern_file_paths[pattern]):
            f.write(path + ' ' + str(data_patterns[pattern][ind]) + '\n')
with open('./dataset/val.txt', 'w') as f:
    for pattern in val_pattern_file_paths.keys():
        for ind, path in enumerate(val_pattern_file_paths[pattern]):
            f.write(path + ' ' + str(data_patterns[pattern][ind]) + '\n')