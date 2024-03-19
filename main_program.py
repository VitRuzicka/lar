#!/usr/bin/env python
# TODO: dont convert depth mask to int
# make sure the depth map has y,x coordinates intead of x, y
# make 


from __future__ import print_function

import cv2

import numpy as np

from robolab_turtlebot import Turtlebot, Rate, get_time, detector
import time

MOVE = 1
ROTATE = 2

linear_vel = 0.2
angular_vel = 0.35

WINDOW = 'obstacles'
WINDOW2 = 'mask'
DRIVE = True  #disables the movement of the robot

DEPTH_HYST = 40 #bulgarian constant for detecting if the poles belong together
DEPTH_THR = 35  #threshold for stopping before obstacles

SLOW_SPEED = 0.5
DONT_LOOK_THR = 3000 #1500ms since detecting the last pole starts detecting again
SLOWING_THRESH = 80  #distance at which the robot starts slowing down before poles
P_FACTOR = 0.8


life_phase = "searching"  #liofe phase of the robot 
running = True
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
    lower_red1 = np.array([0, 150, 100])  # 0, 150, 70
    upper_red1 = np.array([8, 255, 255]) # 10, 255, 255

    # Define range of blue color in HSV
    lower_blue = np.array([90, 150, 100])
    upper_blue = np.array([140, 255, 255])

    # Threshold the HSV image to get only red and blue colors
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    ignore_mask = np.zeros(frame.shape[:2], dtype="uint8")  # Start with a completely black mask
    cv2.rectangle(ignore_mask, (0, int(frame.shape[0] * 0.25)), (frame.shape[1], frame.shape[0]), 255, -1)  # Add a white rectangle covering the lower 3/4 of the mask

    # Apply the ignore mask to the color masks
    mask_red = cv2.bitwise_and(mask_red, mask_red, mask=ignore_mask)
    mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=ignore_mask)

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

    # print("centroid red "+ str(centroid_red))
    # print("centroid blue "+ str(centroid_blue))

    if centroid_blue:
        cv2.circle(combined_highlight, centroid_blue, 5, (255, 0, 0), -1)  #combined_highlight Blue dot
    if centroid_red:
        cv2.circle(combined_highlight, centroid_red, 5, (0, 140, 255), -1)  # Red dot
    # Display the resulting frame
    return combined_highlight, centroid_blue, centroid_red


def centeringPoles(blue, red, slow): #function centers poles with robot's trajectory
    correction = [0,0]
    FRAME = 640/2
    error = (red[0] + blue[0])/2
    error = (FRAME - error ) * P_FACTOR
    correction[1] = error/320
    #either go full speed or crawl to poles
    if(not slow):
        correction[0] = 1
        print("speeding up")
    else:
        correction[0] = SLOW_SPEED
        print("turtle mode")
    return correction    

def calcDepthAvg(depth_image, centroid): # calculates the distance of one centroid from the depth map
    # Define the size of the area to sample around the centroid
    sample_size = 3  # A 3x3 area
    half_size = sample_size // 2

    # Initialize variables to calculate the average
    depth_sum = 0
    count = 0

    # Calculate the bounds of the sample area, ensuring they are within the image
    start_x = max(0, centroid[0] - half_size)
    end_x = min(depth_image.shape[1], centroid[0] + half_size + 1)
    start_y = max(0, centroid[1] - half_size)
    end_y = min(depth_image.shape[0], centroid[1] + half_size + 1)

    # Sum up the depth values within the sample area and count the number of valid points
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            depth = depth_image[y, x]   #TODO CHECK THIS
            if depth > 0:  # Check to ensure depth is valid (non-zero)
                depth_sum += depth
                count += 1

    # Calculate the average depth if there are any valid points
    if count > 0:
        return depth_sum / count
    else:
        return None

def centroidDist(depth_image, centroid_blue, centroid_red ):
    # Initialize a list to hold the average depth values for each centroid
    depths = []

    # Calculate the average depth around the blue centroid if it exists
    if centroid_blue and centroid_red:
        depth_blue = calcDepthAvg(depth_image, centroid_blue)
        depth_red = calcDepthAvg(depth_image, centroid_red)
        
        print("Centroid red: ", centroid_red, depth_image[centroid_red[1], centroid_red[0]])
        print("Centroid blue: ", centroid_blue, depth_image[centroid_blue[1], centroid_blue[0]])

        if depth_blue != None and depth_red != None and  abs(depth_blue - depth_red) < DEPTH_HYST:  #if the poles belong together depth wise
            return min(depth_blue,depth_red)
        else:
            return None
def writeInfo(frame):  #write information on the video feed
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    position = (230, 450)  # Text position (bottom-left corner)
    font_scale = 1  # Font scale (font size)
    color = (0, 0, 255)  # Text color in BGR (blue, green, red)
    thickness = 5  # Text thickness

    # Put the text on the frame
    cv2.putText(frame, life_phase, position, font, font_scale, color, thickness)
    return frame
def checkDir(centroid_blue, centroid_red): #check which way are the poles oriented for rotating the robot
    if centroid_blue and centroid_red:
        if centroid_red[0] > centroid_blue[0]:  #red on the right -> drive right
            direction = -1
        elif centroid_red[0] < centroid_blue[0]: 
            direction = 1
    return direction
def millis():
    return round(time.time() * 1000)

def main():
    global turtle
    global bumper
    global start
    global life_phase
    slowing_down = False
    last_pole_time = 2000 # last depth of detected poles before ROTATE
    corr = [1,0]  #default val
    state = ROTATE #state machine to control the movement of the robot
    direction = 1
    turtle = Turtlebot(pc=True, rgb=True)
    turtle.register_bumper_event_cb(bumper_cb)
    turtle.register_button_event_cb(button_cb)

    turtle.wait_for_point_cloud()

    cv2.namedWindow(WINDOW)
    cv2.namedWindow(WINDOW2)
    cv2.setMouseCallback(WINDOW2, click)
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
        mask = np.logical_and(mask, pc[:, :, 2] < 3) #the camera doesnt see further than 2.5m

        #if np.count_nonzero(mask) <= 0:
        #    print('All point are too far ...')
        #    continue

        # empty image
        #depth_image = np.zeros(mask.shape)  #old far points are being marked as 0
        depth_image = np.full(mask.shape, 255, dtype=np.uint8)
        # assign depth i.e. distance to image
        depth_image[mask] = np.uint8(pc[:, :, 2][mask] / 3 * 255)
        #im_color = cv2.applyColorMap(255 - image.astype(np.uint8), cv2.COLORMAP_JET)
        # show image
        cv2.imshow(WINDOW, depth_image)  #depth mask
        maskaVole, centroid_blue, centroid_red = colorMask(turtle) 
        maskaVole = writeInfo(maskaVole)
        cv2.imshow(WINDOW2, maskaVole)
        cv2.waitKey(1)

        # check obstacle
        #mask = np.logical_and(mask, pc[:, :, 1] > -0.2)
        #data = np.sort(pc[:, :, 2][mask])
        if bumper:
            exit() #exit the program

        
        # if data.size > 50:   #old piece of code that rotated the bot when he came close to obstacle
        #     dist = np.percentile(data, 10)
        #     if dist < 0.6:
        #         state = ROTATE
        dont_look = 1 if (millis() - last_pole_time) < DONT_LOOK_THR else 0
        #our brand new shity code, supported by: chatgpt
        if centroid_blue and centroid_red: #centroids detected
            depth = centroidDist(depth_image, centroid_blue, centroid_red)
            if state != ROTATE:
                corr = centeringPoles(centroid_blue, centroid_red, slowing_down) #correct the drunk robot   
            if depth != None : #the depth of the centroids is invalid
                #print("depth: ", depth, " centering: ", corr)
                if state == MOVE:
                    print("correction: ", corr, depth, slowing_down)
                    if (depth < SLOWING_THRESH) :
                        slowing_down = True
                    if depth < DEPTH_THR: #close to the obstacle, switch the state machine
                        last_pole_time = millis() #save the last depth so the robot doesnt follow the old poles
                        state = ROTATE
                        direction = checkDir(centroid_blue, centroid_red) #sets the direction based on orientation of poles
                        life_phase = "turning"
                        print("beginning rotation rotate", direction)
                        slowing_down = False
                    
                elif state == ROTATE and not dont_look :  #need to exit the rotation somehow, dont look is protection interval
                    #if centroid_blue and centroid_red: #new poles detected, stop the rotation
                    state = MOVE

        if DRIVE:
            # command velocity
            if state == MOVE: #see the centroids so move in their dir
                life_phase = "moving to poles"
                turtle.cmd_velocity(linear=linear_vel*corr[0], angular=corr[1])

            # ebstacle based rotation
            elif state == ROTATE:
                turtle.cmd_velocity(angular=direction*angular_vel)
        

#TODO: Doodle with stopping distance when reaching poles
#      Toggling rotate function when reaching poles, now it oscilates due to centeringPoles

if __name__ == '__main__':
    main()
