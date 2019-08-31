import sys
sys.path.insert(0, "/Users/guanghan.ning/Desktop/dev/CenterNet-Gluon/")

from models.decoder import *
from test_hourglass_network import Y
from models.decoder import _topk, decode_centernet

def test_topk():
    scores = nd.random.uniform(shape=(16,80,128,128))
    batch, cat, height, width = scores.shape
    [topk_scores, topk_inds] = nd.topk(nd.reshape(scores, (batch, cat, -1)), ret_typ='both', k=10)  # return both value and indices
    print("topk_scores.shape = ", topk_scores.shape)
    print("topk_inds.shape = ", topk_inds.shape)

    topk_score, topk_inds, topk_clses, topk_ys, topk_xs = _topk(scores, K=10)
    print("topk_score.shape = ", topk_score.shape)


def test_decode_centernet():
    print("output: heatmaps", Y[0]["hm"].shape)
    print("output: wh_scale", Y[0]["wh"].shape)
    print("output: xy_offset", Y[0]["reg"].shape)

    heatmaps, scale, offset = Y[0]["hm"], Y[0]["wh"], Y[0]["reg"]
    detections = decode_centernet(heat=heatmaps, wh=scale, reg=offset, cat_spec_wh=False, K=10, flag_split=False)
    print(detections.shape)


def display_func_name(func):
    print("\n \t testing {}".format(func.__name__))


if __name__ == "__main__":
    function_list = [test_topk, test_decode_centernet]
    #function_list = [test_topk]
    for unit_test in function_list:
        display_func_name(unit_test)
        unit_test()