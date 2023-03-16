# send post requests to localhost:5000/video_feed

# send a post request to localhost:5000/video_feed in response you will get a json with the distance and a frame with the bounding box

# import the necessary packages
from imutils.video import VideoStream
import imutils
import cv2
import requests
import json
import time
from requests_toolbelt import MultipartDecoder
import numpy as np

# create class to get the frame from the camera
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # initialize the video stream and allow the cammera sensor to warmup
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = self.vs.read()
        frame = imutils.resize(frame, width=400)
        # encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

# while true loop to get the frame from the camera and send it to the flask api
def gen(camera):
    while True:
        frame = camera.get_frame()
        # send post request to flask api
        r = requests.post('http://localhost:5000/video_feed', files={'frame': frame})
        # get the json response
        r.raise_for_status()
        if r.status_code != 200:
            print('Error')

        # get the json response in format multipart/form-data holding the frame and the json with the distance
        resp = r.content
        decoder = MultipartDecoder(resp, r.headers['Content-Type'])
        for part in decoder.parts:
            if part.headers[b'Content-Disposition'].decode().startswith('form-data; name="frame"'):
                frame = part.content
            if part.headers[b'Content-Disposition'].decode().startswith('form-data; name="distance"'):
                distance = part.content

        # convert the frame from jpeg bytes to numpy array and decode it to opencv format 
        frame = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # show the frame
        cv2.imshow('frame', frame)
        print(distance)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    videoCam = VideoCamera()
    gen(videoCam)