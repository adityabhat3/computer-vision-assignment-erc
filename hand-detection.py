
import cv2
import numpy as np

#for video feed from webcam
video_feed=cv2.VideoCapture(0)

#template image for hand detection
hand_template_gray=cv2.imread('hand_template.jpeg',0)
height, width= hand_template_gray.shape[::1]


while True:
    ret1, image=video_feed.read()
    #applying skin colour mask
    hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 38, 70], dtype="uint8")
    upper_bound = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsvimage, lower_bound, upper_bound)
    #blurring to remove noise, with a kernel of 2x2 size
    blurred = cv2.blur(mask, (2, 2))
    #threshing to find valid regions(binary thresh because we have already used a skin mask)
    ret2, threshed = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    # template matching to hand_template
    match = cv2.matchTemplate(threshed, hand_template_gray, cv2.TM_CCOEFF_NORMED)
    threshold =0.49
    valid_points = np.where(match >= threshold)
    for point in zip(*valid_points[::-1]):
        cv2.rectangle(image, point, (point[0] + width, point[1] + height), (255, 0, 0), 1,)
    #show video with hand detection
    cv2.imshow("Video Feed",image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video_feed.release()
cv2.destroyAllWindows()
