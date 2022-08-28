#Helper program to draw a 'meter' on the cv2 frame to show the user how memorable their image is.
import cv2

def mem_meter(mem_score, frame):#, frame_width, frame_height):

    width, height, channels = frame.shape
    height = height-150 # Lower boxes of equalizer-style meter fall off the
                        # edge without this adjustment.

    # Color RGB definitions (they're actually represented as BGR for some reason)
    red = (20, 20, 230)
    orange = (0, 150, 250)
    yellow = (79, 223, 234)
    yellow_green = (79, 223, 163)
    strong_green = (74, 205, 43)


    color = ()
    num_bars = 0

    # Number of bars and color thresholds

    if mem_score == 0.0 or mem_score < 0.20:
        color = red
        num_bars = 1
    elif mem_score == 0.20 or mem_score < 0.40:
        color = orange
        num_bars = 2
    elif mem_score == 0.40 or mem_score < 0.60:
        color = yellow
        num_bars = 3
    elif mem_score == 0.60 or mem_score < 0.80:
        color = yellow_green
        num_bars = 4
    elif mem_score == 0.80 or mem_score > 0.80:
        color = strong_green
        num_bars = 5

    # Generate list of coordinates/dimensions for each box
                  # top left coordinates  # bottom right coordinates
    pts_list = [((20, height -75 - 55*i),(100, height - 30 - 55*i)) for i in range(num_bars)]
    #print(pts_list)

    for i in range(num_bars):
        # Draw all of the boxes over the frame
        cv2.rectangle(frame, pts_list[i][0], pts_list[i][1], color, -1, cv2.LINE_AA)


