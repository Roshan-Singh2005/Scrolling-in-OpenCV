import cv2
import mediapipe as mp
import pyautogui
import math
import keyboard

# Change this to False if you don't want to see the camera window
SHOW_CAMERA_WINDOW = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Safety feature: Prevents errors if your mouse hits the edge of the screen
pyautogui.FAILSAFE = False 

#  ---- INSTRUCTIONS TO CONTROL ----
# 1. Point Index UP   -> SCROLL UP
# 2. Point Index DOWN -> SCROLL DOWN
# 3. Open Palm        -> PAUSE
# 4. Fist             -> IDLE

print ("Program is running...")
print(f"Window Mode: {'VISIBLE' if SHOW_CAMERA_WINDOW else 'HIDDEN'}")
print ("--- To Terminate the program press 'q'---")

if SHOW_CAMERA_WINDOW:
    print("""
    NOTE: If you don't want to show the Camera frame, 
    change 'SHOW_CAMERA_WINDOW = True' to 'False' in the code.
    """)
    cv2.namedWindow("Hand Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Control", 100,100)

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    if keyboard.is_pressed('q'):
        print(" 'q' pressed. Exiting...")
        break

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_lms.landmark[0]
            index_tip = hand_lms.landmark[8]
            index_pip = hand_lms.landmark[6]
            
            is_index_open = get_distance(wrist, index_tip) > get_distance(wrist, index_pip)

            palm_fingers_open = True
            other_finger_ids = [(12, 10), (16, 14), (20, 18)] 
            
            for tip_id, pip_id in other_finger_ids:
                tip = hand_lms.landmark[tip_id]
                pip = hand_lms.landmark[pip_id]
                if get_distance(wrist, tip) < get_distance(wrist, pip):
                    palm_fingers_open = False

            # PRIORITY 1: PAUSE (If Palm is Open)

            if palm_fingers_open:
                cv2.putText(img, "STATUS: PAUSED (Palm)", (10, 50), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            # PRIORITY 2: SCROLL (If Index is Open AND Palm is NOT open)
            elif is_index_open:
            
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                
                if index_tip.y < index_pip.y:
                    cv2.putText(img, "SCROLL UP", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                    pyautogui.scroll(60)
                    
                else:
                    cv2.putText(img, "SCROLL DOWN", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                    pyautogui.scroll(-60)

            # PRIORITY 3: IDLE (Fist)
            else:
                cv2.putText(img, "STATUS: IDLE", (10, 50), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 3)

    if SHOW_CAMERA_WINDOW:
        cv2.imshow("Hand Control", img)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print ("Program terminated successfully")
