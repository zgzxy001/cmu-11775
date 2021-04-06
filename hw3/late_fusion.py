import numpy as np


bof_video = np.load('./video_pred_bof.npy')
video = np.load('./video_pred.npy')

fuse = (bof_video+video) / 2

predictions = np.argmax(fuse, axis=1)
with open('./late_fusion_bof_resnext.csv', 'w') as out_f:
    for pred in predictions:
        print(pred, file=out_f)

