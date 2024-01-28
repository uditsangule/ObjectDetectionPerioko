import cv2
import numpy as np
from datetime import datetime , timedelta
import os

def uniquecols(n_cols=100 , normalized=False):
    """
    Returns greater than n unique colors which are distinguishable in color space
    :param n_cols: number of color rows required in array
    :param normalized: if True , returns color in normalized form.
    :return: RGB colors in [N,3] shape
    """
    rnge_ = np.arange(0,255,255//n_cols)
    cols =np.vstack((rnge_ ,np.random.randint(0,255,rnge_.shape[0]) ,rnge_[::-1])).T.astype(np.uint8)
    if normalized: cols = cols/255
    return cols

def save_(image , outputh):
    """
    writes image in the output path
    :param image: image in np.uint8 format
    :param outputh: path to save the image
    :return: None
    """
    return cv2.imwrite(filename=outputh + '.png' , img=image)

def putconts(image , contours , color , filled=True):
    """
    draw contours on image with specified color
    :param image: image to draw
    :param contours: points in [[pixel,pixel]]
    :param color: color of contour
    :param filled: if True , returns filled contour else returns edges
    :return: None
    """
    if filled:
        return cv2.fillPoly(img=image ,pts=[contours.astype(np.int32)] , color=color.tolist())
    else:
        return cv2.polylines(img=image, pts=[contours.astype(np.int32)], isClosed=True, color=color.tolist(), thickness=2)

def get_currDT(format = "%Y%m%d_%H-%M", timeonly=False):
    """
    gives the current date,time in given format
    """
    if timeonly:datetime.now().strftime(format.split('_')[1])
    return datetime.now().strftime(format)

def Fstatus(fpath):
    """
    checks the folder path , if doesnt exist, it creates that path
    """
    if not os.path.exists(fpath):os.makedirs(fpath, exist_ok=True)
    return fpath

def showim(image , windowname='output' , waitkey=1, dest=False):
    """
    image vizualizer
    """
    cv2.namedWindow(winname=windowname ,flags=cv2.WINDOW_NORMAL)
    cv2.imshow(windowname , image)
    cv2.waitKey(waitkey)
    if dest:cv2.destroyWindow(winname=windowname)
    return

def puttext(image,text , pos = (30,30) , color = (255,0,0)):
    """write texts on image"""
    thickness = 2 if image.shape[0] > 1080 else 1
    fontscale = 1 if image.shape[0] > 1080 else 0.5
    return cv2.putText(img=image , text=str(text) , org=pos , fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=fontscale, thickness=thickness ,color=color,lineType=cv2.LINE_AA )

def putbbox(image , boxlist,mode='xyxy' , color=(255,0,0)):
    """put bounding box in image"""
    thickness = 2 if image.shape[0] > 1080 else 1
    for (x,y,x1,y1) in boxlist:
        if mode in ['xywh']: cv2.rectangle(img=image , pt1=(int(x),int(y)) , pt2=(int(x + x1) , int(y + y1) ),color=color,thickness=thickness)
        if mode in ['xyxy']: cv2.rectangle(img=image , pt1=(int(x),int(y)) , pt2=(int(x1),int(y1)) , color=color,thickness=thickness)
    return image

