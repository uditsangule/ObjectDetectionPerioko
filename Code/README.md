
*Object Detection and Tracking Masks Using YOLOV8*
-----------------------------------------
**1.steps:**
_________________________________________
* yolo may required some downloading which may be done automaticaly, showing progress bar! keep patience!
* installing requirements : ```python -m pip install -r requirements.txt```
* running file : ```python main.py -i <inputpath> -o <outputpath> -d <int:donwscale videoframes>```
* ```-i <inputfilepath>``` can contain filepath of video or a folder containing videofiles.
* --all enables detecting all objects in COCO Dataset else major objects will be detected like tv,cell phone, cup.
* ```utility/helper.py``` contains all the neccasary function used in ```main.py```.
* get more help for arguments in commandline:```python main.py --help```.
___________________________________________
**2.Understanding the output:**
* The window will open with Videoname and orignal frames with detected boundingbox and edges will appear with respective object color.
* Alongside , segmented mask of the current frame will appear.
* A popup window will stream with objectname which is being tracked and automatically closed , if not more than 3 times
it didnt appear,called aliveness. you can control that aliveness by passing argument after main.py ```--alive <int>```.DEFAULT is 3.
* If detected object contains confidence > 80% it will be saved in : ```Outputs/<videoname>/<id_classname>/<frame_no>.png```
____________________________________________
***3.AboutAuthor***
Name: Udit Sangule [uditsangule@gmail.com]