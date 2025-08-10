# HandDrawProject 🎨📷

Ứng dụng vẽ trực tiếp lên màn hình video từ camera, sử dụng **Python**, **OpenCV** và **MediaPipe**.  
Chương trình nhận diện bàn tay và cho phép bạn vẽ, thay đổi màu bút, tẩy nét bằng cử chỉ.

---

## 📂 Cấu trúc thư mục

```plaintext
HandDrawProject/
│── main.py                  # Chương trình chính
│── requirements.txt         # Danh sách thư viện cần cài
│── README.md                 # Hướng dẫn sử dụng
│
├── utils/                    # Thư viện phụ trợ
│   └── __init__.py
│   └── hand_tracking.py      # Module nhận diện bàn tay
│
├── assets/
│   └── icons/                # Icon SVG/PNG cho nút (brush, eraser, save...)
│
└── .venv/                    # Môi trường ảo Python (tạo sau khi cài)
