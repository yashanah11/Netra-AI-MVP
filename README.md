Netra-AI/
│
├── 📂 src/
│   ├── main.py            # Central Controller (Handles mode switching)
│   ├── vision.py          # YOLOv8 Logic (Object Detection)
│   ├── reader.py          # Tesseract & Barcode Logic (OCR)
│   └── voice.py           # Text-to-Speech Engine (Threaded)
│
├── 📂 assets/
│   ├── demo_video.mp4     # Backup video if live demo fails
│   └── architecture.png   # The diagram below
│
├── 📂 models/
│   └── yolov8n.pt         # Pre-trained YOLO nano model
│
├── requirements.txt       # Dependencies (opencv, ultralytics, etc.)
└── README.md              # Project Documentation
