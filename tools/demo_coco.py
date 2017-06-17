import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import random
import colorsys
import pdb


# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

#CLASSES = ('__background__','person','bike','motorbike','car','bus')

CLASSES = ('__background__',
           'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
           'stop sign', 'parking meter',  'bench',  'bird',  'cat',  
           'dog',  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
           'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
           'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
           'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
           'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
           'hair drier', 'toothbrush')

def rand_hsl():
    '''Generate a random hsl color.'''
    h = random.uniform(0.02, 0.31) + random.choice([0, 1/3.0,2/3.0])
    l = random.uniform(0.3, 0.8)
    s = random.uniform(0.3, 0.8)

    rgb = colorsys.hls_to_rgb(h, l, s)
    return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def vis_detections(im, im_mask, class_name, dets, masks, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    #pdb.set_trace()
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im_mask

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')


        # change in mask rcnn
        mask = masks[i,:,:]
        bbox_round = np.around(bbox).astype(int)
        mask_h = (float)(mask.shape[1])
        mask_w = (float)(mask.shape[0])
        height = bbox_round[3]-bbox_round[1]+1
        width  = bbox_round[2]-bbox_round[0]+1
        fx = width/mask_w
        fy = height/mask_h
        mask_resize = cv2.resize(mask, None, fx=fx, fy=fy)
        mask_resize += 0 # <0.1 -> 0; >0.1 -> 1
        mask_resize = np.around(mask_resize).astype(im.dtype)
        rand_color = rand_hsl()
        im_mask_temp = np.zeros(im.shape).astype(im.dtype)
        #pdb.set_trace()
        for ii in range(3):
            im_mask_temp[bbox_round[1]:bbox_round[3]+1, bbox_round[0]:bbox_round[2]+1, ii] \
            += rand_color[ii]*mask_resize
        im_mask += im_mask_temp



    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #pdb.set_trace()
    return im_mask


def demo(sess, net, image_name, force_cpu):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, '../', image_name)
    # pdb.set_trace()
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, masks = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    # pdb.set_trace()
    im = im[:, :, (2, 1, 0)]

    # change in mask rcnn
    im_mask = np.zeros(im.shape).astype(im.dtype)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH, force_cpu)
        # pdb.set_trace()
        dets = dets[keep, :]
        # change in mask rcnn
        mask = masks[keep, :, :]
        #pdb.set_trace()
        print ('After nms, {:d} object proposals').format(dets.shape[0])
        im_mask = vis_detections(im, im_mask, cls, dets, mask, ax, thresh=CONF_THRESH)

    im += im_mask/2;
    im_mask_grey = cv2.cvtColor(im_mask, cv2.COLOR_RGB2GRAY)
    im_mask_grey[np.where(im_mask_grey!=0)] = 255
    cv2.imwrite('data/test/result/img_with_mask.png', im[:,:,(2,1,0)])
    cv2.imwrite('data/test/result/mask.png',im_mask_grey)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    force_cpu = False
    if args.cpu_mode == True:
	   print 'Use cpu mode'
	   cfg.USE_GPU_NMS = False
	   force_cpu = True
    else:
	   print 'Use gpu mode'
	   cfg.USE_GPU_NMS = True
	   cfg.GPU_ID = args.gpu_id

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _= im_detect(sess, net, im)


    im_names = ['data/test/images/COCO_val2014_000000003964.jpg']


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name, force_cpu)

    plt.savefig('data/test/result/result_COCO_val2014_000000003964.png')

