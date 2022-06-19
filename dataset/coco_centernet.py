from mxnet import nd, gluon, init
from gluoncv import data as gdata

import sys
import numpy as np
import math, json, os, cv2

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

import mxnet.gluon.data as data

class COCO(data.Dataset):
 
  def __init__(self, opt, split):
    super(COCO, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'image_info_test-dev2017.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'instances_{}2017.json').format(split)
    
    
  



class CenterCOCODataset():
     num_classes = 80
     default_resolution = [512, 512]
     mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
     std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
     class_name = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
     valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90]

    def __init__(self, opt, coco_path, split):
        super(CenterCOCODataset, self).__init__()
        self.split = split
        self.opt = opt
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        
        print('==> initializing coco 2017 {} data.'.format(split))
        images = []
        categories = []
        # category_id is from 1 for coco, not 0
        for i, name in enumerate(self.CLASSES):
            categories.append({'supercategory':'none',
                              'id': i+1,
                              'name': name})
        annotations = []
        instance_counter = 1
        image_counter = 1

        with open(coco_path,'r') as fp:
            lines=fp.readlines()

        for line in lines:
            # split any white space
            img_path, ann_path = line.strip().split()
            # img = Image.open(img_path)
            # width, height = img.size
            width, height = imagesize.get(img_path)
            images.append(
                dict(id=image_counter,
                     file_name=img_path,
                     ann_path=ann_path,
                     width=width,
                     height=height))

            try:
                anns = self.get_txt_ann_info(ann_path)
            except Exception as e:
                print(f'bad annotation for {ann_path} with {e}')
                anns = []

            for ann in anns:
                ann['image_id']=image_counter
                ann['id']=instance_counter
                annotations.append(ann)
                instance_counter+=1

            image_counter+=1

        ### pycocotool coco init
        self.coco = COCO()
        self.coco.dataset['type']='instances'
        self.coco.dataset['categories']=categories
        self.coco.dataset['images']=images
        self.coco.dataset['annotations']=annotations
        self.coco.createIndex()
        
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
        self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.max_objs = 128

        print('Loaded {} {} samples'.format(split, self.num_samples))
        
        self._coco = []
        self._load_jsons()
        self.classes = self.class_name[1:]
        
   
    def get_txt_ann_info(self, txt_path):
        """Get annotation from TXT file by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        # img_id = self.data_infos[idx]['id']
        # txt_path = osp.splitext(img_path)[0]+'.txt'
        # txt_path = self.get_ann_path_from_img_path(img_path)
        anns = []
        if osp.exists(txt_path):
            with open(txt_path,'r') as fp:
                lines=fp.readlines()
        else:
            lines=[]
        for line in lines:
            obj=[int(x) for x in line.strip().split(',')]
            # YMIR category id starts from 0, coco from 1
            category_id, xmin, ymin, xmax, ymax = obj
            bbox = [xmin, ymin, xmax, ymax]
            h,w=ymax-ymin,xmax-xmin
            ignore = 0
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = 1

            ann = dict(
                segmentation=[[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
                area=w*h,
                iscrowd=0,
                image_id=None,
                bbox=[xmin, ymin, w, h],
                category_id=category_id+1, # category id is from 1 for coco
                id=None,
                ignore=ignore
            )
            anns.append(ann)
        return anns

    
    def _load_jsons(self):
        _coco = self.coco
        json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
        if self.json_id_to_contiguous is None:
            self.json_id_to_contiguous = json_id_to_contiguous
            self.contiguous_id_to_json = {
                v: k for k, v in self.json_id_to_contiguous.items()}
        else:
            assert self.json_id_to_contiguous == json_id_to_contiguous

    ''' above are for official gluon coco evaluation metrics '''

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.flip:
            flipped = True
            img = img[:, ::-1, :]
            c[0] =  width - c[0] - 1


        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)  # xyc -> cxy

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                        draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])

            if flipped:
              bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                if self.split == 'train':
                    gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
                else:
                    gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, cls_id])

        if self.split == 'val':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                   np.zeros((1, 5), dtype=np.float32)
            return inp, gt_det

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                   np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return inp, hm, wh, reg, ind, reg_mask
    
    
    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
          for cls_ind in all_bboxes[image_id]:
            category_id = self._valid_ids[cls_ind - 1]
            for bbox in all_bboxes[image_id][cls_ind]:
              bbox[2] -= bbox[0]
              bbox[3] -= bbox[1]
              score = bbox[4]
              bbox_out  = list(map(self._to_float, bbox[0:4]))

              detection = {
                  "image_id": int(image_id),
                  "category_id": int(category_id),
                  "bbox": bbox_out,
                  "score": float("{:.2f}".format(score))
              }
              if len(bbox) > 5:
                  extreme_points = list(map(self._to_float, bbox[5:13]))
                  detection["extreme_points"] = extreme_points
              detections.append(detection)
        return detections

      def __len__(self):
        return self.num_samples

      def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                    open('{}/results.json'.format(save_dir), 'w'))

      def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
