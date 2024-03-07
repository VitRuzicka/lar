#!/usr/bin/env python

from __future__ import print_function

import cv2

import numpy as np

from robolab_turtlebot import Turtlebot, Rate, get_time, detector
import time

MOVE = 1
ROTATE = 2

linear_vel = 0.3
angular_vel = 0.4

WINDOW = 'obstacles'
DRIVE = False


running = True
active = True
bumper = False
start = False
frame = []


def click(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Retrieve the color of the pixel at the (x, y) location
        color = frame[y, x]
        # Convert the color from BGR to HSV
        color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"BGR Color: {color}, HSV Color: {color_hsv}")
def bumper_cb(msg):
    global bumper
    """Bumber callback."""
    # msg.bumper stores the id of bumper 0:LEFT, 1:CENTER, 2:RIGHT
    #bumper = bumper_names[msg.bumper]
    # msg.state stores the event 0:RELEASED, 1:PRESSED
    #state = state_names[msg.state]
    print('{} bumper {}'.format(msg.bumper, msg.state))
    # Print the event
    #print('{} bumper {}'.format(bumper, state))
   
    if(msg.state == 1):
        print("bumped into wall")
        bumper = True
def button_cb(msg):
    global start
    if msg.button == 0 and msg.state == 1:
        start = True

def pull_out():
    rate = Rate(10)
    t = get_time()

    while get_time() - t < .5:
        turtle.cmd_velocity(linear=0)
        print("sending command")
        rate.sleep()

    t = get_time()

    while get_time() - t < 2:
        turtle.cmd_velocity(linear=-0.1, angular=.8)
        print("sending command")
        rate.sleep()
    t = get_time()

    while get_time() - t < .5:
        turtle.cmd_velocity(linear=0)
        print("sending command")
        rate.sleep()
def colorMask(turtle): #function to detect the color of obstacles and return masked frame

    global frame
    frame = turtle.get_rgb_image() #read color image
    if not frame.any():
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

     # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Define range of blue color in HSV
    lower_blue = np.array([90, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Threshold the HSV image to get only red and blue colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image to highlight the colors
    red_highlight = cv2.bitwise_and(frame, frame, mask=mask_red)
    blue_highlight = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Combine the highlighted red and blue in one frame
    combined_highlight = cv2.addWeighted(red_highlight, 1, blue_highlight, 1, 0)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def find_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    # Find the centroid of the largest blue and orange contours
    # Assuming the largest contour corresponds to the target color region
    centroid_blue = find_centroid(max(contours_blue, key=cv2.contourArea)) if contours_blue else None
    centroid_red = find_centroid(max(contours_red, key=cv2.contourArea)) if contours_red else None

    print("centroid red "+ str(centroid_red))
    print("centroid blue "+ str(centroid_blue))

    if centroid_blue:
        cv2.circle(combined_highlight, centroid_blue, 5, (255, 0, 0), -1)  # Blue dot
    if centroid_red:
        cv2.circle(combined_highlight, centroid_red, 5, (0, 140, 255), -1)  # Red dot
    # Display the resulting frame
    return combined_highlight

def main():
    global turtle
    global bumper
    global start


    turtle = Turtlebot(pc=True, rgb=True)
    turtle.register_bumper_event_cb(bumper_cb)
    turtle.register_button_event_cb(button_cb)

    print('Waiting for point cloud ...')
    turtle.wait_for_point_cloud()
    direction = None
    print('First point cloud recieved ...')

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, click)
    turtle.play_sound(0) #play the sound - ready

    # while not start:  #wait for the user to start the robot
    #     time.sleep(0.1) 
    turtle.play_sound(1)
    while not turtle.is_shutting_down():
        # get point cloud
        pc = turtle.get_point_cloud()
        
        if pc is None:
            print('No point cloud')
            continue
        

       
        # mask out floor points
        mask = pc[:, :, 1] < 0.2

        # mask point too far
        mask = np.logical_and(mask, pc[:, :, 2] < 3.0)

        #if np.count_nonzero(mask) <= 0:
        #    print('All point are too far ...')
        #    continue

        # empty image
        image = np.zeros(mask.shape)

        # assign depth i.e. distance to image
        image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)
        im_color = cv2.applyColorMap(255 - image.astype(np.uint8),
                                     cv2.COLORMAP_JET)

        # show image
        #cv2.imshow(WINDOW, im_color)  #depth mask
        cv2.imshow(WINDOW, colorMask(turtle))#
        cv2.waitKey(1)

        # check obstacle
        mask = np.logical_and(mask, pc[:, :, 1] > -0.2)
        data = np.sort(pc[:, :, 2][mask])
        if bumper:
            bumper = False
            #pull_out()
            exit() #exit the program

        state = MOVE
        if data.size > 50:
            dist = np.percentile(data, 10)
            if dist < 0.6:
                state = ROTATE
        if DRIVE:
            # command velocity
            if active and state == MOVE:
                turtle.cmd_velocity(linear=linear_vel)
                direction = None

            # ebstacle based rotation
            elif active and state == ROTATE:
                if direction is None:
                    direction = np.sign(np.random.rand() - 0.5)
                turtle.cmd_velocity(angular=direction*angular_vel)
        


if __name__ == '__main__':
    main()
