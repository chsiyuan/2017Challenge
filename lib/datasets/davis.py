# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from fast_rcnn.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

def _filter_crowd_proposals(roidb, crowd_thresh):
    """
    Finds proposals that are inside crowd regions and marks them with
    overlap = -1 (for all gt rois), which means they will be excluded from
    training.
    """
    for ix, entry in enumerate(roidb):
        overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(overlaps.max(axis=1) == -1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        iscrowd = [int(True) for _ in xrange(len(crowd_inds))]
        crowd_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        overlaps[non_gt_inds[bad_inds], :] = -1
        roidb[ix]['gt_overlaps'] = scipy.sparse.csr_matrix(overlaps)
    return roidb

class davis(imdb):
    def __init__(self, image_set, year):
        imdb.__init__(self, 'davis_' + year + '_' + image_set)
        # COCO specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'crowd_thresh' : 0.7,
                       'min_size' : 2}
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'davis')
        # load COCO API, classes, class <-> id mappings
        self._DAVIS = COCO(self._get_ann_file())
        cats = self._DAVIS.loadCats(self._DAVIS.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_davis_cat_id = dict(zip([c['name'] for c in cats],
                                              self._DAVIS.getCatIds()))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self.set_proposal_method('selective_search')
        self._roidb_handler = self.gt_roidb
	self.competition_mode(False)

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014' : 'val2014',          # 5k val2014 subset
            'valminusminival2014' : 'val2014',  # val2014 \setminus minival2014
        }
        coco_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[coco_name]
                           if self._view_map.has_key(coco_name)
                           else coco_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
                             else 'image_info'
        return osp.join(self._data_path, 'annotations',
                        prefix + '_' + self._image_set + self._year + '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._DAVIS.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._DAVIS.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    #change in mask rcnn
    def gtmask_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.gtmask_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = ('DAVIS_' + self._data_name + '_' +
                     str(index).zfill(6) + '.jpg')
        image_path = osp.join(self._data_path, 'images',
                              self._data_name, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    #change in mask rcnn
    def gtmask_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # path e.g., /data/coco/gt_mask/train2014/groundt_train2017_000000000009.png
        file_name = ('groundt_' + self._data_name + '_' +
                     str(index).zfill(6) + '.mat')
        gtmask_path = osp.join(self._data_path, 'gt_mask',
                              self._data_name, file_name)
        assert osp.exists(gtmask_path), \
                'Path does not exist: {}'.format(gtmask_path)
        return gtmask_path


    def selective_search_roidb(self):
        return self._roidb_from_proposals('selective_search')

    def edge_boxes_roidb(self):
        return self._roidb_from_proposals('edge_boxes_AR')

    def mcg_roidb(self):
        return self._roidb_from_proposals('MCG')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_davis_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_davis_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._DAVIS.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._DAVIS.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._DAVIS.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # change in mask rcnn
        ann_id = np.zeros((num_objs), dtype = np.int32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        davis_cat_id_to_class_ind = dict([(self._class_to_davis_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = davis_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0
            # change in mask rcnn
            ann_id[ix] = obj['id']


        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'ann_id': ann_id}

