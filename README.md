# computer-vision-assignment-erc

This is my attempt at the computer vision induction assignment for the ERC induction, using Python and OpenCV.

Note: This code doesn't run in WSL, as there are no webcam drivers in WSL.

### How to run
1. Clone this repo
2. Confirm that hand_template.jpeg is in the same folder as hand_detector.py and that you have opencv-python and numpy installed.
3. Run the hand_detector.py file.
4. Move your hand in front of your camera. You should see a blue box around your hand confirming detection.
5. Press 'q' to exit the video stream.

### Design
We begin by loading the video source, which is our webcam in this case, using:
```py 
video_feed=cv2.VideoCapture(0)
```
Note2: This might lead to problems if your PC has multiple webcams. If you get an error in this line change ```cv2.VideoCapture(0)``` to ```cv2.VideoCapture(1)```

This stores a B&W image of hand_template.jpeg, and finds height and width of the template. It also generates an image of hand_template.jpeg flipped about y-axis, which yields better results in detecting hands than one single template of either hand. 
```py
hand_template_gray=cv2.imread('hand_template.jpeg',0)
hand_template_gray_flipped=cv2.flip(hand_template_gray,1)
height, width= hand_template_gray.shape[::1]
```
We use a while loop to work on individual images from the webcam video feed, using:
```py
while True:
    ret1, image=video_feed.read()
```
I used a skin colour mask from source[2] to filter out whatever doesn't look like (human) skin. I changed a few values of the mask to make it better at detecting hands, through trial and error.
```py
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 38, 70], dtype="uint8")
    upper_bound = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
```
Blurring the masked image removes noise. I chose a small kernel size so that clarity is not lost.
```py
    blurred = cv2.blur(mask, (2, 2))
````
Using a threshold to find what actually classifies as skin. I used a binary threshold because we have already applied a skin mask to filter out non-skin stuff.
```py
    ret2, threshed = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
```
Now, comparing threshed to hand_template_gray, cv2.matchTemplate returns a matrix with corresponding elements showing the fraction of threshed image pixels matching with superimposed template image. I got the threshold value through trial and error, and it feels like the best compromise between not detecting hands and detecting faces as hands.
valid_points is used to find points are above the threshold, and therefore a match.
```py
    match1 = cv2.matchTemplate(threshed, hand_template_gray, cv2.TM_CCOEFF_NORMED)
    threshold =0.49
    valid_points1 = np.where(match >= threshold)
```
Below lines of code draw a rectangle staring at all valid points, of the size of the template image.
```py
    for point in zip(*valid_points1[::-1]):
        cv2.rectangle(image, point, (point[0] + width, point[1] + height), (255, 0, 0), 1,)
```
This is repeated for hand_template_gray_flipped, resulting in better hand detection.

This  displays the final image, and stops the video feed if you press the 'q' key
```py
    cv2.imshow("Video Feed",image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
```
Finally the below lines of code stop the video feed and close the video window once you press 'q'
```py
video_feed.release()
cv2.destroyAllWindows()
```

### Sources
1.  Sentdex's Youtube Playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq
2.  This Medium article, for the skin mask: https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
3.  Official OpenCV documentation: https://docs.opencv.org/4.x/
