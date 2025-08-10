import cv2
import mediapipe as mp
import numpy as np
import os

# Khởi tạo Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Màu và độ dày nét vẽ
brush_color = (0, 0, 255)  # Đỏ
brush_thickness = 5

# Canvas để vẽ
canvas = None

# Thư mục lưu ảnh
save_dir = os.path.join("assets", "samples")
os.makedirs(save_dir, exist_ok=True)

# Mở camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập camera")
            break

        frame = cv2.flip(frame, 1)  # Lật ngang như gương

        if canvas is None:
            canvas = np.zeros_like(frame)

        # Chuyển sang RGB cho Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)

                # Vẽ trên canvas
                cv2.circle(canvas, (x, y), brush_thickness, brush_color, -1)

                # Vẽ skeleton bàn tay
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Gộp canvas với video
        frame_out = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

        cv2.imshow('Hand Drawing', frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas = np.zeros_like(frame)
        elif key == ord('s'):
            filename = os.path.join(save_dir, "draw_result.png")
            cv2.imwrite(filename, frame_out)
            print(f"Ảnh đã lưu: {filename}")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
