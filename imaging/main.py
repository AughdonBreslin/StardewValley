import cv2 as cv
import numpy as np
import os

from time import time

from window_capture import WindowCapture

def find(needle, haystack, threshold=0.99, color=(0, 255, 0)):
    """
    Find the needle in the haystack.
    :param needle: The image to find.
    :param haystack: The image to search in.
    :return: A list of tuples containing the coordinates of the found needle.
    """
    result = cv.matchTemplate(haystack, needle, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    print(f'Best match top left corner: {max_loc}, confidence: {max_val}')
    if max_val < threshold:
        print(f'No needle found.  {max_val} < {threshold} threshold')
        return []
    needle_height, needle_width = needle.shape[:2]
    top_left_corner = max_loc
    bottom_right_corner = (top_left_corner[0] + needle_width, top_left_corner[1] + needle_height)
    cv.rectangle(haystack, top_left_corner, bottom_right_corner, color, thickness=2, lineType=cv.LINE_4)
    return max_loc

def calculate_box_dimensions(box_top, box_bottom):
    """
    Calculate the dimensions of the box based on the top and bottom images.
    :param box_top: The top image of the box.
    :param box_bottom: The bottom image of the box.
    :return: A tuple containing the width and height of the box.
    """
    print(box_top, box_bottom)
    return 0, 0

def main():
    windowCapture = WindowCapture()
    
    box_top = cv.imread('images/box_top.png', cv.IMREAD_UNCHANGED)
    box_top = cv.cvtColor(box_top, cv.COLOR_BGRA2BGR)

    box_bottom = cv.imread('images/box_bottom.png', cv.IMREAD_UNCHANGED)
    box_bottom = cv.cvtColor(box_bottom, cv.COLOR_BGRA2BGR)

    fish = cv.imread('images/fish.png', cv.IMREAD_UNCHANGED)
    fish = cv.cvtColor(fish, cv.COLOR_BGRA2BGR)
    
    fish_out = cv.imread('images/fish_out.png', cv.IMREAD_UNCHANGED)
    fish_out = cv.cvtColor(fish_out, cv.COLOR_BGRA2BGR)

    last_time = time()
    for image in os.listdir('images'):
    # while True:
        screenshot = windowCapture.get_screenshot()
        # print(f'images/{image}')
        screenshot = cv.imread(f'images/{image}', cv.IMREAD_UNCHANGED)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGRA2BGR)

        box_top_loc = find(box_top, screenshot, 0.90, color=(255, 0, 0))
        box_bottom_loc = find(box_bottom, screenshot, 0.90, color=(0, 255, 0))
        fish_loc = find(fish, screenshot, 0.90, color=(0, 0, 255)) or find(fish_out, screenshot, 0.90, color=(0, 255, 255))

        if box_top_loc and box_bottom_loc:
            # Calculate the dimensions of the box
            box_width, box_height = calculate_box_dimensions(box_top, box_bottom)

            # Draw the box on the screenshot
            cv.rectangle(screenshot, (box_top_loc[0], box_top_loc[1]), 
                         (box_top_loc[0] + box_width, box_top_loc[1] + box_height), 
                         (255, 0, 0), thickness=2)
        # Display the frame 
        cv.imshow('Camera Feed', screenshot)
        cv.waitKey(0)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

        print(f'FPS: {1 / (time() - last_time):.2f}')
        last_time = time()


if __name__ == '__main__':
    main()