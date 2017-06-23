import numpy as np
from fast_rcnn.config import cfg
import cv2

def generate_mask(scores, boxes, masks, deformed_masks, force_cpu=False):
	"""
	Input: scores(n*81), boxes(n*324) and masks(n*H*W) are output of the test.py
		   deformed_masks(H*W) are the output mask in last frame
	Output: filtered mask(H*W) with labels of instance 1,2,3...
	"""

	ins_num = 0  # number of instances in this image
	masks_filtered = np.zeros(deformed_masks.shape).as_type(int)  # filtered mask

	for cls_ind, cls in enumerate(CLASSES):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        # Do NMS
        keep = nms(dets, NMS_THRESH, force_cpu)
        dets = dets[keep, :]
        masks = masks[keep, :, :]

        # Remove masks that have class scores lower than the threshold.
        inds = np.where(dets[:, -1] >= cfg.CONF_THRESH)[0]
        dets = dets[inds, :]
        masks = masks[inds, :, :]

        # Remove masks that don't match the gt masks.
        for i in range(mask.shape[0]):
        	[masks_filtered, ins_num] = filter_mask(dets[i,0:4], masks[i,:, :], deformed_masks, masks_filtered, ins_num)
    
    return masks_filtered

def filter_mask(bbox, mask, deformed_masks, masks_filtered, idx):
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
    mask_full_size = np.zeros(masks_filtered.shape).astype(int)
    mask_full_size[bbox_round[1]:bbox_round[3]+1, bbox_round[0]:bbox_round[2]+1] = mask_binary

    # Calculate overlap rate of the mask with every instance.
    is_overlap = 0
    for i in range(np.max(deformed_masks)):
    	gt_mask_ins = np.copy(deformed_masks)
    	gt_mask_ins[np.where(mask!=(i+1))] = 0;
    	gt_mask_ins[np.where(mask!=(i+1))] = 1;
    	gt_mask_ins.astype(int)
    	overlap = np.sum(mask_full_size & gt_mask_ins)/np.sum(mask_binary)
    	if overlap >= cfg.FILTER_THRESH:
    		is_overlap += 1

    # If the mask matches one of the instances, keep it.
    if is_overlap == 1
		idx += 1
		masks_filtered[bbox_round[1]:bbox_round[3]+1, bbox_round[0]:bbox_round[2]+1] = int(mask_binary*idx)

    return masks_filtered, idx