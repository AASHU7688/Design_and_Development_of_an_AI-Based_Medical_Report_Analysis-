# MediScan - Medical Report Analysis & X-Ray Diagnosis Platform

**MediScan** is a full-stack web application designed for medical diagnostics using **OCR-based report analysis** and **AI-powered X-ray disease classification**.

---

## 🚀 Features Overview

### 🏠 Home Page
- Modern and responsive layout  
- Showcase features, statistics, and call-to-action  
- Professional medical theme and icons  

### 📄 Medical Report Analysis
- Upload medical reports (PDF, PNG, JPG, JPEG)
- OCR-based text extraction using **Tesseract**
- AI-driven report interpretation using **Google Gemini**
- Displays categorized results with clean medical formatting

### 🩻 X-Ray Disease Detection (NEW)
- Integrated **YOLOv8 Deep Learning Model**
- Can classify **17 types of X-ray related diseases**
- Accepts chest X-ray image uploads
- Displays predictions with confidence percentage
- Medical-grade output visualization

> 🦴 **Example X-Ray diagnoses include**: Pneumonia, Cardiomegaly, Edema, Hernia, Infiltration, Pleural Effusion, Fibrosis, and more.

### 👥 About Page
- Team and mission details  
- Timeline and milestones  

### 📞 Contact Page
- Validated contact form  
- Business hours, email, and location  

### 🔐 Authentication (Login/Signup)
- Secure login and registration
- Social login options
- Form validation with error alerts

---

## 🛠 Technology Stack

| Component | Technology |
|----------|------------|
| Backend | Django 3.2 |
| Frontend | HTML, CSS, JavaScript |
| OCR Engine | Tesseract OCR |
| PDF Processing | pdfplumber |
| AI (Report Analysis) | Google Gemini |
| **X-Ray Model** | **YOLOv8 (Ultralytics)** 🩻 |
| Styling | Custom CSS |
| Icons | Font Awesome |
| Fonts | Google Fonts (Poppins) |

---

## 📌 Installation & Setup

### 🔧 Prerequisites
- Python 3.7+
- Tesseract OCR installed
- Google Gemini API Key
- YOLOv8 dependencies (`ultralytics`)

### 📥 Steps to Install & Run

```bash
git clone <repository-url>
cd ocr_project
