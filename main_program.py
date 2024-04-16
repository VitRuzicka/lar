#!/usr/bin/env python
# TODO: dont convert depth mask to int
# make sure the depth map has y,x coordinates intead of x, y
# make 


from __future__ import print_function

import cv2
import sys

import numpy as np
from math import *

from robolab_turtlebot import Turtlebot, Rate, get_time, detector
import time

MOVE = 1
ROTATE = 2
CORIGATING = 3
ANGLE_CORR = 4



linear_vel = 0.2
angular_vel = 0.35

WINDOW = 'obstacles'
WINDOW2 = 'mask'
DRIVE = True  #disables the movement of the robot
DEPTH_DEBUG = False


DEPTH_HYST = 40 #bulgarian constant for detecting if the poles belong together
DEPTH_THR = 30  #threshold for stopping before obstacles

SLOW_SPEED = 0.5
DONT_LOOK_THR = 3000 #1500ms since detecting the last pole starts detecting again
SLOWING_THRESH = 80  #distance at which the robot starts slowing down before poles 
POLE_DIST_THRESH = 20  #threshold for dist between poles (in cm) - they belong together or no  
CORRECTED = 0.2  #threshold to detect when to stop centering
PAR_ANGLE_THRESHOLD = 20  #threshold for deciding whether to correct for pole angle or not [in deg]
RIDE_OUT_OF_COLLISION_DIST = 15  #distance to ride when running out of collision
BULGANG = 9*pi/40  #initial rotation for approximately pi/3
NUM_ROT = 7 #rotate t
BACK_OFF_DIST = 5 #reverse drive from pylons


BOOST = 3
P_FACTOR = 0.8
CAM_FOV = 86
DEG2RAD = 3.1415/180.0
RAD2DEG = 1/DEG2RAD
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


life_phase = "first"  #liofe phase of the robot 
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
        sys.exit()  #auf viedersehen
def button_cb(msg):
    global start
    if msg.button == 0 and msg.state == 1:
        start = True


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

    # Define range of green color in HSV
    lower_green = np.array([35, 70, 100])
    upper_green = np.array([80, 160, 255])

    # Threshold the HSV image to get only red, blue and green colors
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    ignore_mask = np.zeros(frame.shape[:2], dtype="uint8")  # Start with a completely black mask
    cv2.rectangle(ignore_mask, (0, int(frame.shape[0] * 0.25)), (frame.shape[1], frame.shape[0]), 255, -1)  # Add a white rectangle covering the lower 3/4 of the mask

    # Apply the ignore mask to the color masks
    mask_red = cv2.bitwise_and(mask_red, mask_red, mask=ignore_mask)
    mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=ignore_mask)
    mask_green = cv2.bitwise_and(mask_green, mask_green, mask=ignore_mask)
    

    # Bitwise-AND mask and original image to highlight the colors
    red_highlight = cv2.bitwise_and(frame, frame, mask=mask_red)
    blue_highlight = cv2.bitwise_and(frame, frame, mask=mask_blue)
    green_highlight = cv2.bitwise_and(frame, frame, mask=mask_green)

    # Combine the highlighted red and blue in one frame
    #Old: combined_highlight = cv2.addWeighted(red_highlight, 1, blue_highlight, 1, green_highlight, 1, 0)
    # First, combine red and blue highlights
    temp_combined = cv2.addWeighted(red_highlight, 1, blue_highlight, 1, 0)

    # Then, combine the result with the green highlight
    combined_highlight = cv2.addWeighted(temp_combined, 1, green_highlight, 1, 0)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
    sorted_contours_green = sorted(contours_green, key=cv2.contourArea, reverse=True)[:2]

    # Initialize an empty list to store centroids
    centroids_green = []

    # Loop through the sorted contours (up to 2)
    for contour in sorted_contours_green:
        centroids_green.append(find_centroid(contour))

    # print("centroid red "+ str(centroid_red))
    # print("centroid blue "+ str(centroid_blue))

    if centroid_blue:
        cv2.circle(frame, centroid_blue, 5, (255, 0, 0), -1)  #combined_highlight Blue dot
    if centroid_red:
        cv2.circle(frame, centroid_red, 5, (0, 140, 255), -1)  # Red dot
    if len(centroids_green) >= 2: #found more than one green centroids
        cv2.circle(combined_highlight, centroids_green[0], 5, (0, 255, 0), -1)
        cv2.circle(combined_highlight, centroids_green[1], 5, (0, 255, 0), -1)
        return combined_highlight, centroid_blue, centroid_red, centroids_green[0], centroids_green[1]

    return frame, centroid_blue, centroid_red, None, None  #frame


def centeringPoles(blue, red): #function centers poles with robot's trajectory
    correction = [1,0]
    FRAME = 640/2
    error = (red[0] + blue[0])/2
    error = (FRAME - error ) * P_FACTOR
    correction[1] = error/320
    return correction    

def filterAvg(col_depths): #function to filter out the depth data (take out max, min and calculate average fo the rest)
    blue_depths = [pair[0] for pair in col_depths]
    red_depths = [pair[1] for pair in col_depths]
    par_angs = [pair[2] for pair in col_depths]

    # Remove max and min from blue depths
    if len(blue_depths) > 2:
        blue_depths.remove(max(blue_depths))
        blue_depths.remove(min(blue_depths))

    # Remove max and min from red depths
    if len(red_depths) > 2:
        red_depths.remove(max(red_depths))
        red_depths.remove(min(red_depths))

    # Remove max and min from paralel angles
    if len(par_angs) > 2:
        par_angs.remove(max(par_angs))
        par_angs.remove(min(par_angs))

    # Calculate the averages
    depth_blue = sum(blue_depths) / len(blue_depths) if blue_depths else 0
    depth_red = sum(red_depths) / len(red_depths) if red_depths else 0
    par_ang = sum(par_angs) / len(par_angs) if par_angs else 0

    return depth_blue, depth_red, par_ang

def calcDepthAvg(depth_image, centroid):
    # Define the height of the area to sample around the centroid
    sample_height = 3  # A 1x3 area (vertical column)
    half_height = sample_height // 2

    # Initialize variables to calculate the average
    depth_sum = 0
    count = 0

    # Calculate the bounds of the sample area, ensuring they are within the image
    # For vertical averaging, the x-coordinate remains constant
    x = centroid[0]

    # Ensure the x-coordinate is within the image width
    if 0 <= x < depth_image.shape[1]:
        start_y = max(0, centroid[1] - half_height)
        end_y = min(depth_image.shape[0], centroid[1] + half_height + 1)

        # Sum up the depth values within the sample area and count the number of valid points
        for y in range(start_y, end_y):
            depth = depth_image[y, x]
            if depth > 0:  # Check to ensure depth is valid (non-zero)
                depth_sum += depth
                count += 1

        # Calculate the average depth if there are any valid points
        if count > 0:
            return depth_sum / count
        else:
            return None
    else:
        return None


def centroidParams(depth_image, centroid_blue, centroid_red): #return the calculated values of poles 
    if centroid_blue and centroid_red:
        depth_blue = calcDepthAvg(depth_image, centroid_blue)
        depth_red = calcDepthAvg(depth_image, centroid_red)
        #calculate the distance between pole and center of view
        angles = ( abs(FRAME_WIDTH/2-centroid_blue[0])/320.0*(CAM_FOV/2)  ,   abs(FRAME_WIDTH/2-centroid_red[0])/320.0*(CAM_FOV/2) )
        #projected dist of the poles
        distances = ( min(depth_blue, depth_red)*sin(angles[0]*DEG2RAD), min(depth_blue,depth_red)*sin(angles[1]*DEG2RAD) )  #chose distance acc to the closest pole (direct projection)
        #angle between the center of poles viewed from above compared to tangent line of the front of the robot
        paralel_angle = atan( abs(depth_blue-depth_red) / abs(distances[0]-distances[1]) )   #calculates the real distance between poles
        
        #print("found poles, angles:", angles, " dist:", abs(distances[0] - distances[1]) )
        #print("angle between poles", paralel_angle*RAD2DEG)
        return(angles, distances, paralel_angle, depth_blue, depth_red)
    return None

def centroidDist(depth_image, centroid_blue, centroid_red ):
    # Initialize a list to hold the average depth values for each centroid
    depths = []

    # Calculate the average depth around the blue centroid if it exists
    angles , distances , angles, depth_blue, depth_red = centroidParams(depth_image, centroid_blue, centroid_red)

    if abs(distances[0] - distances[1]) < POLE_DIST_THRESH: #detect if the centroids belong together based on distance
        if depth_blue != None and depth_red != None and  abs(depth_blue - depth_red) < DEPTH_HYST:  # detect if the poles belong together based on depth
            return min(depth_blue,depth_red)
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
    is_green = False
    snail = False
    ignore = False
    rotate_cnt = 0
    avg_cnt = 0
    last_pole_time = 2000 # last depth of detected poles before ROTATE
    corr = [1,0]  #default val
    state = ROTATE #state machine to control the movement of the robot
    direction = 1
    prev_ang = 0 #for rotating one sixth of full revolution in first life phase
    centroid_dist_arr = []
    col_depths = []  #filter for filtering the distance
    corrected_from_corigating = 0  #keep track of the angle when correcting the poles to center in standstill
    turtle = Turtlebot(pc=True, rgb=True)
    turtle.register_bumper_event_cb(bumper_cb)
    turtle.register_button_event_cb(button_cb)

    turtle.reset_odometry()

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
        
        if bumper:
            exit() #exit the program
        
        # mask out floor points
        mask = pc[:, :, 1] < 0.2
        # mask point too far
        mask = np.logical_and(mask, pc[:, :, 2] < 3) #the camera doesnt see further than 2.5m

        # empty image
        #depth_image = np.zeros(mask.shape)  #old far points are being marked as 0
        depth_image = np.full(mask.shape, 255, dtype=np.uint8)
        # assign depth i.e. distance to image
        depth_image[mask] = np.uint8(pc[:, :, 2][mask] / 3 * 255)
        #im_color = cv2.applyColorMap(255 - image.astype(np.uint8), cv2.COLORMAP_JET)
        # show image
        cv2.imshow(WINDOW, depth_image)  #depth mask
        maskaVole, centroid_blue, centroid_red, centroid_green1, centroid_green2 = colorMask(turtle) 
        maskaVole = writeInfo(maskaVole)
        cv2.imshow(WINDOW2, maskaVole)
        cv2.waitKey(1)

        # if centroid_green1 != None and centroid_green2 != None and centroid_blue != None and centroid_red != None :
        #     if centroidDist(depth_image, centroid_green1, centroid_green2) < centroidDist(depth_image, centroid_blue, centroid_red):
        #         centroid_blue = centroid_green1
        #         centroid_red = centroid_green2
        #         is_green = True                
        if centroid_blue and centroid_red :
            angles , distances , par_ang, depth_blue, depth_red = centroidParams(depth_image, centroid_blue, centroid_red)
        if DEPTH_DEBUG:
            print("paralell pole angle:", par_ang*RAD2DEG, "IPA blue,red", depth_blue, depth_red)
            time.sleep(1)
            direction = checkDir(centroid_blue, centroid_red) #check the dir in which to turn
            if(avg_cnt < 5):
                avg_cnt += 1
                #check the angle we need to correct
                angles , distances , par_ang, depth_blue, depth_red = centroidParams(depth_image, centroid_blue, centroid_red)
                col_depths.append([depth_blue, depth_red, par_ang])
            else:
                avg_cnt = 0
                depth_blue, depth_red, par_ang = filterAvg(col_depths)
                print("filtered blue, red:",depth_blue, depth_red, "filtered paralel angle: ",RAD2DEG*par_ang)
                col_depths = []
        if DRIVE:
            # command velocity
            #life_phase = "moving to poles"
            if state == MOVE: #see the centroids so move in their dir
                #our brand new shity code, supported by: chatgpt
                if centroid_blue and centroid_red : #centroids detected
                    depth = centroidDist(depth_image, centroid_blue, centroid_red)
                    print(depth)
                    if state != ROTATE:
                        corr = centeringPoles(centroid_blue, centroid_red) #correct the drunk robot   
                    if depth != None : #the depth of the centroids is invalid
                        #print("depth: ", depth, " centering: ", corr)
                        if state == MOVE:
                            #print("correction: ", corr, depth)
                            if (depth < SLOWING_THRESH) :
                                #detecting if robot is at the center of poles
                                snail = True
                                if(not ignore):

                                    turtle.cmd_velocity(linear=0, angular=0)
                                    turtle.reset_odometry()
                                    time.sleep(1)
                                    angles , distances , par_ang, depth_blue, depth_red = centroidParams(depth_image, centroid_blue, centroid_red)
                                    print("pole paralell angle:", par_ang*RAD2DEG)
                                    state = MOVE

                                    if(par_ang*RAD2DEG > PAR_ANGLE_THRESHOLD): #here correct the angle of poles compared to robot
                                        life_phase = "angling"
                                        state = ANGLE_CORR
                                        tacho = turtle.get_odometry()
                                        print("corrected angle during coringating:", tacho[2]*RAD2DEG)
                                        corrected_from_corigating = tacho[2]
                                    else:
                                        ignore = True
                            if depth < DEPTH_THR: #close to the obstacle, switch the state machine
                                if(is_green):
                                    turtle.play_sound(2)
                                    sys.exit()
                                state = ROTATE
                                direction = checkDir(centroid_blue, centroid_red) #sets the direction based on orientation of poles
                                life_phase = "turning"
                                print("beginning rotation in dir", direction)
                                turtle.reset_odometry()
                                time.sleep(0.1)
                    else: 
                        print("Poles jsou daleko, seru na tebe")
                if snail:
                    turtle.cmd_velocity(linear=linear_vel*SLOW_SPEED, angular=corr[1])       
                else:
                    turtle.cmd_velocity(linear=linear_vel, angular=corr[1])

            # rotate fixed 90deg
            elif state == ROTATE:
                if life_phase == "first":
                    curr_ang = 0
                    if rotate_cnt <= NUM_ROT: #
                        print("Counter: ",rotate_cnt)
                        turtle.reset_odometry()
                        time.sleep(0.1)
                        if centroid_blue and centroid_red:
                            centroid_dist_arr.append((rotate_cnt, (depth_blue+depth_red)/2))
                            print("found some poles, saving", centroid_dist_arr[-1][0], centroid_dist_arr[-1][1])
                        while (BULGANG > abs(curr_ang) and not turtle.is_shutting_down()):
                            curr_ang = turtle.get_odometry()[2]
                            turtle.cmd_velocity(linear=0, angular=direction*angular_vel*2)
                        print("Done rotating for pi/4")
                        turtle.cmd_velocity(linear=0, angular=0)
                        time.sleep(0.5)
                        rotate_cnt += 1
                    elif (rotate_cnt > NUM_ROT):  #evaluate the closest centroids
                        turtle.play_sound(0)
                        turtle.reset_odometry()
                        min_dir = (0,1000) #default values
                        print(centroid_dist_arr)
                        for dir in centroid_dist_arr:
                            if dir[1] < min_dir[1]:
                                min_dir = dir
                        if min_dir != (0,1000):
                            curr_ang = 0
                            print("rotating to closest poles", min_dir)
                            for i in range(0, min_dir[0]):
                                print(i)
                                turtle.reset_odometry()
                                time.sleep(0.1)
                                curr_ang = 0
                                while (BULGANG > abs(curr_ang) and not turtle.is_shutting_down()):
                                    curr_ang = turtle.get_odometry()[2]
                                    turtle.cmd_velocity(linear=0, angular=direction*angular_vel*BOOST)
                                turtle.cmd_velocity(linear=0, angular=0)
                                time.sleep(0.5)
                            state = CORIGATING
                            life_phase = "corigating"
                            print("corigating")
                            turtle.reset_odometry()
                            time.sleep(1)
                            
                        else:
                            sys.exit()
                        

                elif life_phase == "turning":
                    snail = False
                    ignore = False
                    turtle.reset_odometry()
                    time.sleep(0.1)
                    print("Backing off")
                    while not turtle.is_shutting_down():  #BACK OFF
                        tacho = turtle.get_odometry()
                        #print(sqrt(pow(tacho[0], 2)+pow(tacho[1], 2)))
                        if (sqrt(pow(tacho[0], 2)+pow(tacho[1], 2)) < BACK_OFF_DIST/100):
                            turtle.cmd_velocity(linear=-linear_vel, angular=0)
                        else:
                            break
                    print("turning 90deg")
                    while not turtle.is_shutting_down():  #rotate 90 deg
                        tacho = turtle.get_odometry()
                        #print(tacho)
                        if(abs(tacho[2]) < 4*pi/10):
                            turtle.cmd_velocity(linear=0, angular=direction*angular_vel)
                        else:
                            break
                    turtle.reset_odometry()
                    time.sleep(0.1)
                    while not turtle.is_shutting_down():  #ride out of potentional colision
                        tacho = turtle.get_odometry()
                        #print(sqrt(pow(tacho[0], 2)+pow(tacho[1], 2)))
                        if (sqrt(pow(tacho[0], 2)+pow(tacho[1], 2)) < RIDE_OUT_OF_COLLISION_DIST/100):
                            turtle.cmd_velocity(linear=linear_vel, angular=0)
                        else:
                            break
                    turtle.cmd_velocity(linear=0, angular=0)
                    state = CORIGATING
                    life_phase = "corigating"
                    turtle.reset_odometry()
                    time.sleep(0.1)
                    print("corigating now")    

            elif state == CORIGATING: #turn in the direction of the poles but dont move, check their angle
                if centroid_blue and centroid_red : #centroids detected
                    depth = centroidDist(depth_image, centroid_blue, centroid_red)
                    corr = centeringPoles(centroid_blue, centroid_red) #correct the drunk robot   
                    turtle.cmd_velocity(linear=0, angular=corr[1]*angular_vel*BOOST)
                    #print("correction now:", corr)
                    if(abs(corr[1]) < CORRECTED):  #the robot is facing the poles
                        print("robot is facing the poles")
                        #check for angle of the centers of poles compared to robot
                        turtle.cmd_velocity(linear=0, angular=0)
                        time.sleep(1)
                        state = MOVE
                        life_phase = "moving to poles"

            elif state == ANGLE_CORR:  #now correct for the angle of poles
                if centroid_blue and centroid_red :
                    turtle.cmd_velocity(linear=0, angular=0)
                    time.sleep(1)
                    direction = checkDir(centroid_blue, centroid_red) #check the dir in which to turn
                    if(avg_cnt < 5):
                        avg_cnt += 1
                        #check the angle we need to correct
                        angles , distances , par_ang, depth_blue, depth_red = centroidParams(depth_image, centroid_blue, centroid_red)
                        col_depths.append([depth_blue, depth_red, par_ang])
                    else:
                        avg_cnt = 0
                        print("List of depths: ", col_depths)
                        depth_blue, depth_red, par_ang = filterAvg(col_depths)
                        col_depths = []
                        
                    if depth_blue > depth_red: #red is closer, flip the wanted dir
                        rotate_dir = -1*direction
                    else: #turn the other way if the poles are 
                        rotate_dir = direction
                    #now rotate_dir shows which way to rotate -1 CW, 1 CCW
                    print("Poles are that far away: ",depth_blue, depth_red)
                    print("so we turn ", 90 - par_ang*RAD2DEG + corrected_from_corigating*RAD2DEG, "right" if rotate_dir < 0 else "left", "and drive", min(depth_blue, depth_red)*sin(par_ang) )
                    angle_to_rotate = 4*pi/10 - abs(par_ang) #+ corrected_from_corigating
                    angle_to_return = 4*pi/10 #+ abs(par_ang) #+ corrected_from_corigating
                    dist_to_drive = (min(depth_blue, depth_red)*sin(par_ang)/100)  #the robot works in centimeters
                    turtle.reset_odometry()
                    time.sleep(0.1)
                    print("rotating the angle to align with axis of poles")
                    while not turtle.is_shutting_down():  #rotate the requested angle
                        tacho = turtle.get_odometry()
                        #print("Turned deg: ", tacho[2])
                        if(abs(tacho[2]) < angle_to_rotate):
                            turtle.cmd_velocity(linear=0, angular=rotate_dir*angular_vel)
                        else:
                            break
                        time.sleep(0.01)
                    print("moving to align")
                    while not turtle.is_shutting_down():  #move the requested dist
                        tacho = turtle.get_odometry()
                        #print("Ujetá vzdálenost píčo: ", sqrt(pow(tacho[0], 2)+pow(tacho[1], 2)))
                        if (sqrt(pow(tacho[0], 2)+pow(tacho[1], 2)) < dist_to_drive):
                            turtle.cmd_velocity(linear=linear_vel, angular=0)
                        else:
                            break
                        time.sleep(0.01)
                    turtle.reset_odometry()
                    time.sleep(0.1)
                    print("rotating back:", angle_to_return*RAD2DEG)
                    while not turtle.is_shutting_down():  #rotate the requested angle
                        tacho = turtle.get_odometry()
                        #print("Turned deg back: ", tacho[2])
                        if(abs(tacho[2]) < angle_to_return):
                            turtle.cmd_velocity(linear=0, angular=-1*rotate_dir*angular_vel)
                        else:
                            break
                        time.sleep(0.01)
                    state = CORIGATING
                    life_phase = "corigating"
                    direction = -1*rotate_dir
                    
if __name__ == '__main__':
    main()
