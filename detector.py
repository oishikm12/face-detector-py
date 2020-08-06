# Main OpenCV library
import cv2


# We will be using Haar Cascade models for face frontal recognition
faceData = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Next we need an input for the video ~ webcam
webcam = cv2.VideoCapture(0)

# Loop over video feed forever
while True:

    # True or false , Current Frame 
    success, feed = webcam.read()

    # We need to convert it to grayscale for algorithm to work
    gs_img = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)

    # Next we need to detect face even if it moves or scales
    face = faceData.detectMultiScale(gs_img)

    # Now we draw that tracking rectangle
    # The params we pass are (img, start(x, y), end(x, y), color(green aka 0,255,0), thickness(2))
    for x, y, w, h in face:
        cv2.rectangle(feed, (x, y), (x+w, y+h), (0, 255, 0), 10)

    
    # Printing the face
    cv2.imshow('Face Detection', feed)

    # Checking for key press every 0.1ms
    key = cv2.waitKey(1)

    # Represents Q key for quit
    if key == 81 or key == 113:
        break


# Release the webcam feed
webcam.release()

# Goodbye
print("Goodbye !!!")

