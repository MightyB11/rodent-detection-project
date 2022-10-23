from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 25
camera.brightness = 60


camera.start_preview()
time.sleep(5)
camera.stop_preview()   

camera.start_recording('/home/pi/Desktop/video2.h264')
time.sleep(3600)
camera.stop_recording()
