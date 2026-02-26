# -AI-Smart-Attendance-System-with-Advanced-Anti-Spoofing
📌 Project Description

This project is an AI-powered facial recognition attendance system built using Django REST Framework, InsightFace, and OpenCV, enhanced with advanced anti-spoofing security mechanisms to prevent fake attendance using photos or mobile screens.

The system allows secure user registration and real-time attendance punching by verifying facial embeddings while performing deep anti-spoofing analysis to detect:

📱 Mobile screen attacks

🖼 Printed photo spoofing

💻 Digital display replay attacks

🔆 Screen glare and reflection artifacts

🧠 Core Technologies

Django & Django REST Framework – Backend API

InsightFace – Face detection & embedding extraction

OpenCV – Image processing & anti-spoofing analysis

NumPy – Numerical computation

Cosine Similarity – Face matching algorithm

🚀 System Workflow
1️⃣ User Registration

User submits: user_id, name, and face_image

Anti-spoofing validation is performed

Face embedding is generated using InsightFace

Embedding is securely stored in the database

2️⃣ Attendance Punch

User uploads a live face image

Anti-spoofing verification is executed

Face embedding is extracted

Cosine similarity comparison with stored embeddings

If similarity > threshold → Attendance recorded

🛡 Advanced Anti-Spoofing Techniques

The system includes multi-layer spoof detection:

🔍 1. FFT-Based Screen Pattern Detection

Detects digital display moiré patterns using frequency spectrum analysis.

🎨 2. Color Entropy Analysis

Screens often show unnatural color uniformity; entropy is measured to detect it.

🧱 3. Texture & Depth Analysis

Laplacian variance (texture sharpness)

Gradient magnitude analysis (surface depth detection)

🔆 4. Reflection & Glare Detection

Detects abnormal brightness patterns caused by screens.

🔎 Face Matching Logic

The system uses cosine similarity for comparing embeddings:

similarity = dot(A, B) / (||A|| * ||B||)

A similarity threshold (e.g., 0.4) determines recognition success.

📂 API Endpoints
🔹 Register User

POST /register/

Required Fields:

user_id

name

face_image

🔹 Punch Attendance

POST /punch/

Required Field:

face_image

Response:

Attendance status

Confidence score

User ID

Name

🎯 Key Features

✔ Secure face recognition attendance
✔ Advanced anti-spoofing detection
✔ REST API architecture
✔ Embedding storage using binary serialization
✔ Real-time verification support
✔ Designed for scalable enterprise deployment

💡 Use Cases

🏫 Schools & Colleges

🏢 Offices & Enterprises

🏭 Industrial Workforce Management

🏥 Secure Access Control Systems

🔮 Vision

This project aims to build a secure, AI-driven biometric authentication system capable of preventing spoof attacks while maintaining fast and accurate facial recognition.
