from utility import helpers as imp
from utility import CodeProfiler as cpf
import numpy as np
import cv2
import argparse
import math
import time
from collections import defaultdict
import os
from ultralytics import YOLO



def show_roi(frame_ ,out, frequency , args):
    frequency = {classname: freq - 1 for classname, freq in frequency.items() if freq >= 0}
    [cv2.destroyWindow(winname=key) for key, freq in frequency.items() if frequency[key] <= 0]
    return frequency

def streamVideo(videopath, outputpath ,args):
    accptedcls = ['cup', 'tv', 'cell phone']
    cap = cv2.VideoCapture(videopath)#0 if os.path.exists(videopath) else videopath)
    currtime = imp.get_currDT()
    if args.verbose:print(f"Started Video at:{currtime} with containing fps:{cap.get(cv2.CAP_PROP_FPS)}")
    time1 = time.time()
    modelpath = os.path.join('..','Sources','model','yolov8m-seg.pt')
    model = YOLO(modelpath)
    accpt_cls_ids = np.array([id for id , name in model.names.items() if name in accptedcls]) if not args.all else None
    freq = dict()
    fno = 0
    uniqcols = imp.uniquecols(n_cols=len(model.names))
    while 1:
        tic = time.time()
        ret_ , frame_ = cap.read()
        if not ret_ or cv2.waitKey(1) == ord('q') : break
        fno += 1
        if args.downscale > 1: frame_ = cv2.resize(src=frame_ , dsize=(frame_.shape[1]//args.downscale,frame_.shape[0]//args.downscale),interpolation=cv2.INTER_NEAREST_EXACT)
        ##main part of objdetection
        tracker = model.track(source=frame_ , persist=True , verbose=False ,retina_masks=True,max_det=300 if accpt_cls_ids is None else 10 , classes=accpt_cls_ids, tracker='botsort.yaml',device=args.device , conf=args.confthresh)
        mask_ = np.zeros(shape=frame_.shape , dtype=np.uint8)
        #out = model.predict(frame_ , verbose=False , classes=accpt_cls_ids)
        for i_ in range(len(tracker[0].boxes.cls)):
            bbox = tracker[0].boxes.xyxy[i_].int().cpu()
            clsid , conf , trackid = int(tracker[0].boxes.cls[i_]) , float(tracker[0].boxes.conf[i_]) ,int(tracker[0].boxes.id[i_])

            label = model.names.__getitem__(clsid)
            freq['{0}_{1}'.format(trackid,label)] = args.alive
            w , h = tracker[0].boxes.xywh[i_].int().cpu()[2:].tolist()
            pixelEx = w // 4
            tempframe = tracker[0].orig_img.copy()
            tempframe = imp.putbbox(image=tempframe , boxlist=[bbox] , color=(0,0,0))
            tempframe = imp.puttext(image=tempframe , pos=(int(bbox[0]) , int(bbox[1]-pixelEx//2)) , text=str(np.round(conf*100,2)) , color=(0,0,0))
            tempframe = imp.puttext(image=tempframe , pos=(int(bbox[0]+ w//3) , int(bbox[1]-pixelEx//2)) , text=label , color=(0,0,0))

            frame_roi = tempframe[max(bbox[1]-pixelEx,0):min(bbox[3] + pixelEx,frame_.shape[1]),
                        max(bbox[0] - pixelEx , 0): min(bbox[2] + pixelEx , frame_.shape[0])]


            if conf > 0.8 :imp.save_(image=frame_roi , outputh=imp.Fstatus(args.output+os.sep+os.path.basename(videopath).split('.',-1)[0] + os.sep + '{0}_{1}'.format(trackid,label) + os.sep) + str(fno))
            imp.showim(image=frame_[bbox[1]:bbox[3], bbox[0]:bbox[2], :], windowname=str(trackid)+'_' + str(label))
            mask_ = imp.putconts(mask_ , contours=tracker[0].masks.xy[i_] , color=uniqcols[clsid],filled=True)
            frame_ = imp.putconts(frame_ , contours=tracker[0].masks.xy[i_] , color=uniqcols[clsid],filled=False)
            frame_ = imp.puttext(image=frame_ , pos=(int(bbox[0]+bbox[2])//2 , int(bbox[1]+bbox[3])//2) , text=label , color=tuple(uniqcols[clsid].tolist()))
            print(f"Clip:{os.path.basename(videopath)} - Timestamp:{imp.timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))} - Object:{label} - Confidence:{np.round(100*conf,2)}%")
        freq = show_roi(frame_=frame_ ,frequency=freq,out=tracker , args=args)
        fps = str(round(1/(time.time() - tic),2))
        frame_ = imp.putbbox(image=frame_ , boxlist=tracker[0].boxes.xyxy.int().cpu())
        OutputFrame = np.concatenate((frame_, mask_), axis=1)
        imp.showim(image=OutputFrame,windowname=os.path.basename(videopath))


    if args.verbose : print(f"Done:{os.path.basename(videopath)}\tElapsed time:{round((time.time() - time1)/60,3)} mins")
    cap.release()
    cv2.destroyAllWindows()

    return
def main(videopath = os.path.join('..','Sources','Individual Clips') , outpath = os.path.join('..','Outputs')):

    parse =argparse.ArgumentParser()
    parse.add_argument("-i","--input" ,type=str , help="inputpath of video or folder containg videos",default=videopath)
    parse.add_argument("-o" ,"--output",type=str , help="outputpath to save all output",default=outpath)
    parse.add_argument("-d","--downscale" , type=int , help="downscale the size of frame , default=2" ,default=2)
    parse.add_argument("-v","--verbose" , type=int,help="turn on/off print flag for the outputs\n[1->yes , 0->No],default=1\n" ,default=1)
    parse.add_argument("-a" ,"--all" , action="store_true" , help="detect all objects in coco,current classes:['cup','tv','cellphone']")
    parse.add_argument("--alive", type=int,
                                    help="aliveness of detected object , helps in tracking , default=3", default=3)
    args = parse.parse_args()
    args.device = 'cpu'
    args.confthresh = 0.5

    if args.verbose:print(f"running file : {args.input}\nOutputpath:{outpath}\n")
    if os.path.isfile(args.input):
        streamVideo(videopath = args.input , outputpath= imp.Fstatus(os.path.join(args.output , os.path.basename(args.input).split('.')[0])),args=args)
    else:
        if args.verbose:print(f"videos found in folder:{len(os.listdir(args.input))}")
        for video in os.listdir(args.input):
            if not os.path.isfile(os.path.join(args.input , video)): continue
            streamVideo(videopath=os.path.join(args.input , video) , outputpath=imp.Fstatus(os.path.join(args.output,video.split('.')[0])) , args=args)
    return 1

if __name__ == '__main__':
    inpvid = ''
    cpf.StartCodeProfiler()
    main()
    cpf.DisplayCodeProfilerResultsAndStopCodeProfiler()