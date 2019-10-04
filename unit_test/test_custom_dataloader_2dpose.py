import sys, os
sys.path.insert(0, "/export/guanghan/CenterNet-Gluon/dataset")
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/dataset")
from cocohp_centernet import CenterMultiPoseDataset
import mxnet as mx
from mxnet import nd, gluon, init
import gluoncv
from gluoncv.data.batchify import Tuple, Stack, Pad

def test_load():
    from opts import opts
    opt = opts().init()

    batch_size = 16
    # inp, hm, wh, reg, dep, dim, rotbin, rotres, ind, reg_mask, rot_mask, meta
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack(), Stack())
    num_workers = 2

    train_dataset = CenterMultiPoseDataset(opt, split = 'train')
    train_loader = gluon.data.DataLoader( train_dataset,
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    ctx = [mx.gpu(int(i)) for i in opt.gpus_str.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    '''
           inp, ind, \
           hm, wh, reg, reg_mask, \         # 2d detection: center, wh, offset
           kps, kps_mask, \                 # 2d pose: joint locations relative to center
           hm_hp, hp_offset, hp_ind, hp_mask  # 2d pose: joint heapmaps (and offset to compoensate discretization)
    '''

    for i, batch in enumerate(train_loader):
        print("{} Batch".format(i))
        print("image batch shape: ", batch[0].shape)
        print("indices batch shape", batch[1].shape)

        print("center batch shape", batch[2].shape)
        print("2d wh batch shape", batch[3].shape)
        print("2d offset batch shape", batch[4].shape)
        print("2d offset mask batch shape", batch[5].shape)

        print("pose relative to center batch shape", batch[6].shape)
        print("pose relative to center mask batch shape", batch[7].shape)

        print("pose heatmap batch shape", batch[8].shape)
        print("pose heatmap offset batch shape", batch[9].shape)
        print("pose heatmap ind shape", batch[10].shape)
        print("pose heatmap mask batch shape", batch[11].shape)

        X = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        targets_inds = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        targets_center = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)  # heatmaps: (batch, num_classes, H/S, W/S)
        targets_2d_wh = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)  # scale: wh (batch, 2, H/S, W/S)
        targets_2d_offset = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0) # offset: xy (batch, 2, H/s, W/S)
        targets_2d_wh_mask = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)

        targets_poserel = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
        targets_poserel_mask = gluon.utils.split_and_load(batch[7], ctx_list=ctx, batch_axis=0)

        targets_posemap = gluon.utils.split_and_load(batch[8], ctx_list=ctx, batch_axis=0)
        targets_posemap_offset = gluon.utils.split_and_load(batch[9], ctx_list=ctx, batch_axis=0)
        targets_posemap_ind = gluon.utils.split_and_load(batch[10], ctx_list=ctx, batch_axis=0)
        targets_posemap_mask = gluon.utils.split_and_load(batch[11], ctx_list=ctx, batch_axis=0)

        print("len(targets_center): ", len(targets_center))
        print("First item: image shape: ", X[0].shape)
        print("First item: center heatmap shape: ", targets_center[0].shape)
    return


if __name__ == "__main__":
    test_load()
