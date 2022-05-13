# computer-vision-assignment-erc

This is my attempt at the computer vision induction assignment for the ERC induction, using Python and OpenCV.
Note: This code doesn't run in WSL, as there are no webcam drivers in WSL.

### How to run
1. Clone this repo
2. Confirm that hand_template.jpeg is in the same folder as hand_detector.py and that you have opencv-python and numpy installed.
3. Run the hand_detector.py file.
4. Press 'q' to exit the video stream.

### Design
We begin by loading the video source, which is our webcam in this case, using:
```py 
video_feed=cv2.VideoCapture(0)
```
Note2: This might lead to problems if your PC has multiple webcams. If you get an error in this line change ```cv2.VideoCapture(0)``` to ```cv2.VideoCapture(1)```
This stores a B&W image of hand_template.jpeg, and finds height and width of the template.
```py
hand_template_gray=cv2.imread('hand_template.jpeg',0)
height, width= hand_template_gray.shape[::1]
```
We use a while loop to work on individual images from the webcam video feed, using:
```py
ret1, image=video_feed.read()
```
I used a skin colour mask from source[2] to filter out whatever doesn't look like (human) skin. I changed a few values of the mask to make it better at detecting hands, through trial and error.
```py
    hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 38, 70], dtype="uint8")
    upper_bound = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsvimage, lower_bound, upper_bound)
```
Blurring the masked image removes noise. I chose a small kernel size so that clarity is not lost.
```py
    blurred = cv2.blur(mask, (2, 2))
````
Threshing to find what actually classifies as skin. I used binary threshing because we have already applied a skin mask to filter out non-skin stuff.
```py
ret2, threshed = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
```
Now, using the jpeg image to be compared to this. cv2.matchTemplate returns a matrix with corresponding elements showing the fraction of threshed image pixels matching with superimposed template image. I got the threshold value through trial and error, and it feels like the best compromise between not detecting hands and detecting faces as hands.
valid_points is used to find points are above the threshold, and therefore a match.
```py
    match = cv2.matchTemplate(threshed, hand_template_gray, cv2.TM_CCOEFF_NORMED)
    threshold =0.49
    valid_points = np.where(match >= threshold)
```
Below lines of code draw a rectangle staring at all valid points, of the size of the template image.
```py
    for point in zip(*valid_points[::-1]):
        cv2.rectangle(image, point, (point[0] + width, point[1] + height), (255, 0, 0), 1,)
```


### Sources
1.  Sentdex's Youtube Playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq
2.  This Medium article, for the skin mask: https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
3.  Official OpenCV documentation: https://docs.opencv.org/4.x/
