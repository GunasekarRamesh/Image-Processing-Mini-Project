import cv2  # Import OpenCV library for video capture and image processing
import mediapipe as mp  # Import MediaPipe for hand tracking
import numpy as np  # Import NumPy for numerical operations
from collections import deque  # Import deque for storing hand positions
import math  # Import math for mathematical calculations

# Initialize mediapipe hands and drawing utils
mp_hands = mp.solutions.hands  # Access the hands module from MediaPipe
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Initialize hands with confidence thresholds
mp_draw = mp.solutions.drawing_utils  # Access the drawing utilities for visualizing landmarks

# Coordinates for finger tips and thumb
fingers_coordinate = [(8, 6), (12, 10), (16, 14), (20, 18)]  # Indices for finger tips and joints
thumb_coordinate = (4, 3)  # Indices for thumb tip and joint

# Variables for hand movement analysis
hand_positions = deque(maxlen=10)  # Store recent hand positions
hand_speeds = []  # Store calculated hand speeds

# Start video capture
cap = cv2.VideoCapture(0)  # Initialize video capture from default camera

while cap.isOpened():  # Loop until the video capture is open
    total_fingers = 0  # Initialize total finger count
    success, img = cap.read()  # Read a frame from the video capture
    if not success:  # If frame not read successfully, continue to next iteration
        continue
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
    results = hands.process(converted_image)  # Process the image to detect hand landmarks

    if results.multi_hand_landmarks:  # If hand landmarks are detected
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):  # Iterate over each detected hand
            upcount = 0  # Initialize count of raised fingers
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)  # Draw hand landmarks and connections
            h, w, c = img.shape  # Get image dimensions
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]  # Convert landmarks to pixel coordinates

            # Only process if lm_list has enough points
            if len(lm_list) > 20:
                # Hand movement analysis
                wrist_pos = lm_list[0]  # Get wrist position
                hand_positions.append(wrist_pos)  # Add wrist position to deque

                # Calculate speed if there are enough positions
                if len(hand_positions) > 1:
                    dx = hand_positions[-1][0] - hand_positions[-2][0]  # Calculate horizontal distance
                    dy = hand_positions[-1][1] - hand_positions[-2][1]  # Calculate vertical distance
                    distance = math.sqrt(dx**2 + dy**2)  # Calculate Euclidean distance
                    hand_speeds.append(distance)  # Add distance to speeds list

                    # Draw trajectory
                    for i in range(1, len(hand_positions)):
                        cv2.line(img, hand_positions[i - 1], hand_positions[i], (0, 255, 0), 2)  # Draw trajectory line

                # Calculate and display average speed
                if hand_speeds:
                    avg_speed = sum(hand_speeds) / len(hand_speeds)  # Calculate average speed
                    if avg_speed > 100:  # Normalize very high initial speeds
                        avg_speed = avg_speed / 10
                    cv2.putText(img, f"Speed: {avg_speed:.2f}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)  # Display speed

                for coordinate in fingers_coordinate:
                    if lm_list[coordinate[0]][1] < lm_list[coordinate[1]][1]:  # Check if finger tip is above joint
                        upcount += 1  # Increment raised finger count

                # Thumb detection: Adjust for left and right hand
                if (lm_list[thumb_coordinate[0]][0] > lm_list[thumb_coordinate[1]][0] and lm_list[0][0] < lm_list[9][0]) or \
                   (lm_list[thumb_coordinate[0]][0] < lm_list[thumb_coordinate[1]][0] and lm_list[0][0] > lm_list[9][0]):  # Adjust thumb logic based on hand orientation
                    upcount += 1

            total_fingers += upcount  # Add the count of the current hand to total fingers
            cv2.putText(img, f"Hand {idx + 1}: {upcount}", (10, 70 + 30 * idx), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)  # Display finger count for each hand

    cv2.putText(img, f"Total: {total_fingers}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)  # Display total finger count
    cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)  # Create full screen window
    cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set window property to full screen
    cv2.imshow("Hand Tracking", img)  # Show processed frame

    if cv2.waitKey(1) == 113:  # 113 - Q : Press Q to close video
        break

cap.release()  # Release video capture object
cv2.destroyAllWindows()  # Destroy all OpenCV windows
