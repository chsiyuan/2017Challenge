import cv2
import numpy as np
from image_wrap import warp_images
import pdb
from PIL import Image
#thin plate
# ps = [90 90;90 height-89;width-89 height-89; width-89 90;round(width/2) round(height/2)];
# pd = ps;
# pd = [0.2*width*(rand(5,1)-0.5),0.2*height*(rand(5,1)-0.5)]+ps;
# [mask_tps m_mask]=rbfwarp2d(mask_t,ps,pd,'thin');
DEBUG = False

def deform(annotation,ids):
    """ Count number of objects from segmentation mask"""


    img_deforms = np.zeros(annotation.shape).astype(int) 
    n = annotation.shape[2]
    for i in range(n):
        anno_single = np.copy(annotation[:,:,i])
        img_deform = deform_instance(anno_single)
        # pdb.set_trace()
        img_deforms[:,:,i] = img_deform

    return img_deforms


def deform_instance(img):
    #scale 0.95-1.05
    rows,cols = img.shape
    center = [cols/2,rows/2];
    scale = (np.random.rand()-0.5)/5;
    # print('scale = %f', scale)
    H_s = np.float32([[1+scale, 0, -scale*center[0]], [0, 1+scale, -scale*center[1]]]);
    if DEBUG:
        pdb.set_trace()
    img = cv2.warpAffine(img.astype(np.float32),H_s,(cols,rows))
    #cv2.imwrite('after_scale.jpg', img*80)
    

    #img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

    #translation +-10%
    x_scale = (np.random.rand() - 0.5)/20;
    y_scale = (np.random.rand() - 0.5)/20;

    rows,cols = img.shape
    M = np.float32([[1,0,x_scale*cols],[0,1,y_scale*rows]])
    img = cv2.warpAffine(img,M,(cols,rows))
    #cv2.imwrite('after_trans.jpg', img*80)

    #thin plate
    height = rows
    width = cols
    ps = np.array([[90,90],\
        [90,height-89],\
        [width-89, height-89],\
        [width-89,90],\
        [width/2,height/2]])
    pd = ps
    pd = np.array([[0.1*width*(np.random.rand()-0.5), 0.1*height*(np.random.rand()-0.5)]]) + ps
    output_region = (0,0,height-1,width-1) #(xmin, ymin, xmax, ymax) 
    img = warp_images(ps, pd, [img], output_region, interpolation_order = 1, approximate_grid=2)[0]
    #cv2.imwrite('after_thin.jpg', img*80)

    #dilate
    kernel = np.ones((int(0.01*cols),int(0.01*rows)),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = cv2.resize(img, (cols, rows)) 
    img.astype(int)
    return img

# filename = 'Annotations/Full-resolution/golf/00000.png'
# annotation = np.atleast_3d(Image.open(filename))[...,0]
# img_deform,ids = deform(annotation)
