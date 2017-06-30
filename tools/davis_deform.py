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
from deform import deform
from PIL import Image

DEBUG = True


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



def filter_mask(bbox, mask, deformed_masks, masks_filtered):
    """
    Resize mask according to the bounding box and calculate overlap with every instance.
    Keep the mask that has overlap rate higher than FILTER_THRESH.

    If the mask are highly overlaped with multiple gt instances, it will be rejected. 
    """
    # Resize mask based on the bbox.
    # Put it into a zero image which has the same size with the original image.
    bbox_round = np.around(bbox).astype(int)
    mask_h = (float)(mask.shape[1])
    mask_w = (float)(mask.shape[0])
    height = bbox_round[3]-bbox_round[1]+1
    width  = bbox_round[2]-bbox_round[0]+1
    fx = width/mask_w
    fy = height/mask_h
    mask_resize = cv2.resize(mask, None, fx=fx, fy=fy)
    mask_binary = np.around(mask_resize).astype(int)
    mask_full_size = np.zeros([deformed_masks.shape[0],deformed_masks.shape[1]]).astype(int)
    mask_full_size[bbox_round[1]:bbox_round[3]+1, bbox_round[0]:bbox_round[2]+1] = mask_binary

    # Calculate overlap rate of the mask with every instance.
    is_overlap = 0
    # area = np.sum(mask_binary):
    for i in range(deformed_masks.shape[2]):
        deformed_masks.astype(int)
        gt_mask_ins = deformed_masks[:,:,i]
        overlap = float(np.sum(mask_full_size & gt_mask_ins))/float(np.sum(mask_full_size ^ gt_mask_ins))
        print('intersection:' + str(np.sum(mask_full_size&gt_mask_ins)))
        print('union:' + str(np.sum(mask_full_size^gt_mask_ins)))
        if overlap >= cfg.TEST.FILTER:
            is_overlap = 1
            break

    # If the mask matches one of the instances, keep it.
    if is_overlap == 1:
        if np.max(masks_filtered[:,:,i]) != 0:
            overlap0 = float(np.sum(masks_filtered[:,:,i] & gt_mask_ins))/float(np.sum(masks_filtered[:,:,i] ^ gt_mask_ins))
            if overlap0 < overlap:
                masks_filtered[:,:,i] = mask_full_size
        else:
            masks_filtered[:,:,i] = mask_full_size
    return masks_filtered

def generate_mask(scores, boxes, masks, deformed_masks, force_cpu):
    """
    Input: scores(n*81), boxes(n*324) and masks(n*H*W) are output of the test.py
           deformed_masks is a [H*W*n] matrix containing the deformed mask for all instances in the last frame
    Output: filtered mask(H*W) with labels of instance 1,2,3...
    """

    masks_filtered = np.zeros(deformed_masks.shape).astype(int)  # filtered mask

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        # Do NMS
        keep = nms(dets, cfg.TEST.NMS, force_cpu)
        dets = dets[keep, :]
        mask_nms = masks[keep, :, :]
        print('class no. {:d} - {}: after nms, {:d} object proposals').format(cls_ind, cls, dets.shape[0])

        # Remove masks that have class scores lower than the threshold.
        inds = np.where(dets[:, -1] >= cfg.TEST.CONF)[0]
        dets = dets[inds, :]
        mask = mask_nms[inds, :, :]
        print ('{:d} object proposals are higher than threshold {:.2f}').format(dets.shape[0], cfg.TEST.CONF)

        # Remove masks that don't match the gt masks.
        for i in range(mask.shape[0]):
        #pdb.set_trace()
            masks_filtered = filter_mask(dets[i,0:4], mask[i,:, :], deformed_masks, masks_filtered)  
    return masks_filtered


def demo2(sess, net, image_name, deformed_mask, ids, force_cpu):
    # deform mask is a list
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, '../', image_name)
    im0 = cv2.imread(im_file)

    # Detect objects
    timer = Timer()
    timer.tic()
    scores, boxes, masks = im_detect(sess, net, im0)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Process masks
    filtered_mask = generate_mask(scores, boxes, masks, deformed_mask, force_cpu)
    filtered_mask_tmp = np.copy(filtered_mask)
    pdb.set_trace()
    for i in range(filtered_mask.shape[2]):
        if np.max(filtered_mask[:,:,i]) == 0:
            tmp = np.copy(deformed_mask[:,:,i])
            filtered_mask[:,:,i] = tmp
            filtered_mask_tmp[:,:,i] = np.copy(tmp)
        filter_mask_tmp[:,:,i] = filter_mask_tmp[:,:,i] *ids[i]

    filtered_mask_merge = np.amax(filtered_mask_tmp, axis=2)
    #output_mask = np.zeros([filtered_mask.shape[0], filtered_mask.shape[1], 3])
    max_value = np.max(filtered_mask_merge).astype(float)
    filtered_mask_merge = (filtered_mask_merge/max_value*255).astype(np.uint8)
    #for i in range(3):
    #    output_mask[:,:,i] = filtered_mask/max_value*255
    #output_mask.astype(np.uint8)
    return filtered_mask_merge, filtered_mask


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

    CONF_THRESH = 0.7
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

    return im_mask

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

    # if DEBUG:
    #     pdb.set_trace()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    rootdir = 'data/DAVIS_test-dev-480/JPEGImages/480p/'
    anndir = 'data/DAVIS_test-dev-480/Annotations/480p/'
    maskdir = 'data/DAVIS_test-dev-480/out_mask/'
    #boxdir = 'data/DAVIS/out_box'

    if not os.path.isdir(maskdir):
        os.mkdir(maskdir)
    # if not os.path.isdir(boxdir):
    #     os.mkdir(boxdir)

    list = os.listdir(rootdir)

    if DEBUG:
        list = ['tennis-vest']

    for line in list:
        filepath = os.path.join(rootdir,line)
        if os.path.isdir(filepath):
            maskpath = os.path.join(maskdir,line)
            #boxpath = os.path.join(boxdir,line)
            # read in the mask for the first frame
            annpath = os.path.join(anndir,line,'00000.png')

            if not os.path.isdir(maskpath):
                os.mkdir(maskpath)

            annotation = np.atleast_3d(Image.open(annpath))[...,0]
            maskout = os.path.join(maskpath,'00000.jpg')
            cv2.imwrite(maskout, annotation)
            ids = sorted(np.unique(annotation))
            # Remove unknown-label
            ids = ids[:-1] if ids[-1] == 255 else ids
            # Handle no-background case
            ids = ids if ids[0] else ids[1:]
            mask_pick= np.zeros([annotation.shape[0], annotation.shape[1], len(ids)]).astype(int) 
            for i in ids:
                anno_single = np.copy(annotation)
                #pdb.set_trace()
                anno_single[np.where(annotation != i)] = 0
                anno_single[np.where(annotation == i)] = 1
                mask_pick[:,:,i-1] = anno_single

            # if DEBUG:
            #     pdb.set_trace()

            
            # if not os.path.isdir(boxpath):
            #     os.mkdir(boxpath)
            framelist = os.listdir(filepath)
            for filename in framelist:
                if os.path.splitext(filename)[1] == '.jpg' and os.path.splitext(filename)[0] != '00000':
                    im_name = os.path.splitext(filename)[0]
                    imgpath = os.path.join(filepath,filename)
                    print 'Demo for ' + imgpath
                    # deform_ann is a list 
                    deform_ann = deform(mask_pick, ids)
                    if DEBUG:
                        pbs.set_trace()
                    im_mask_merge, mask_pick = demo2(sess, net, imgpath, deform_ann, ids, force_cpu)
                    maskout = os.path.join(maskpath,filename)
                    #boxname = im_name + '.png'
                    #boxout = os.path.join(boxpath, boxname)
                    cv2.imwrite(maskout, im_mask_merge)
                    # plt.savefig(boxout)
                    #pdb.set_trace()


    # im_names = ['COCO_val2014_000000003964', 'COCO_val2014_000000000474']

    # for im_name in im_names:
    #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print 'Demo for data/test/images/{}.jpg'.format(im_name)
    #     imgpath = 'data/test/images/{}.jpg'.format(im_name)
    #     im_mask = demo(sess, net, imgpath, force_cpu)
    #     maskout = 'data/test/result/mask_{}.png'.format(im_name)
    #     boxout = 'data/test/result/box_{}.png'.format(im_name)
    #     cv2.imwrite(maskout, im_mask)
    #     plt.savefig(boxout)

