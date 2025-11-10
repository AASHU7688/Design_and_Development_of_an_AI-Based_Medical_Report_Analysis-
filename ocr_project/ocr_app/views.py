import os
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from PIL import Image
import pdfplumber
import pytesseract
from tensorflow.keras.models import load_model
from io import BytesIO
from typing import List, Tuple, Optional
import math
try:
    import numpy as np
except Exception:
    np = None
try:
    import fitz  # PyMuPDF for PDF rasterization (OCR fallback)
except Exception:
    fitz = None

# Google Gemini
from google.generativeai import configure, GenerativeModel



# model = load_model("model.h5")



# Configure Gemini API
configure(api_key="")
model = GenerativeModel('gemini-2.5-flash')

# Tesseract for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# X-ray YOLO model (lazy load)
XRAY_YOLO_MODEL = None
XRAY_CLASSES = [
    'pneumosclerosis', 'hydrothorax', 'post_inflammatory_changes', 'abscess', 'hydropneumothorax',
    'post_traumatic_ribs_deformation', 'fracture', 'cardiomegaly', 'sarcoidosis', 'emphysema',
    'scoliosis', 'atherosclerosis_of_the_aorta', 'venous_congestion', 'pneumonia', 'atelectasis',
    'tuberculosis'
]

def get_xray_model():
    global XRAY_YOLO_MODEL
    if XRAY_YOLO_MODEL is not None:
        return XRAY_YOLO_MODEL
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return None

    weights_path = os.environ.get('XRAY_YOLO_WEIGHTS')
    if not weights_path:
        # Prefer specific classification weights if present
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls_candidate = os.path.join(root, 'xray_yolo_cls_model.pt')
        if os.path.exists(cls_candidate):
            weights_path = cls_candidate
        else:
            # try to find a .pt in project tree that looks like xray/yolo
            candidate = None
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith('.pt') and ('xray' in fn.lower() or 'yolo' in fn.lower()):
                        candidate = os.path.join(dirpath, fn)
                        break
                if candidate:
                    break
            weights_path = candidate
    if not weights_path or not os.path.exists(weights_path):
        return None
    try:
        from ultralytics import YOLO  # type: ignore
        XRAY_YOLO_MODEL = YOLO(weights_path)
        return XRAY_YOLO_MODEL
    except Exception:
        return None

def is_xray_like_image(pil_image: Image.Image) -> bool:
    try:
        img = pil_image.convert('RGB')
        if np is None:
            # Fallback: approximate by checking if R,G,B channels are similar
            w, h = img.size
            sample = img.resize((max(1, w // 32), max(1, h // 32)))
            px = list(sample.getdata())
            diffs = [abs(r - g) + abs(g - b) + abs(b - r) for (r, g, b) in px]
            avg_diff = sum(diffs) / max(1, len(diffs))
            return avg_diff < 60  # small channel difference suggests grayscale-like
        # Use HSV saturation to check if mostly grayscale
        arr = np.asarray(img).astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = np.maximum(r, np.maximum(g, b))
        cmin = np.minimum(r, np.minimum(g, b))
        delta = cmax - cmin
        saturation = np.where(cmax == 0, 0, delta / (cmax + 1e-6))
        mean_sat = float(np.mean(saturation))
        # Chest X-rays are typically low saturation
        return mean_sat < 0.25
    except Exception:
        return True

def run_xray_inference(pil_image: Image.Image) -> List[Tuple[str, float]]:
    model = get_xray_model()
    if model is None:
        raise RuntimeError("X-ray model not configured. Install ultralytics and set XRAY_YOLO_WEIGHTS to your .pt file.")
    # Ensure 3 channels for YOLO
    try:
        pil_image = pil_image.convert('RGB')
    except Exception:
        pass
    results = model(pil_image, verbose=False, imgsz=640)
    # Prefer model-provided class names; fallback to configured list
    model_names = getattr(model, 'names', None)
    class_to_conf = {}

    # 1) Detection path (boxes)
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                try:
                    cls_idx = int(b.cls.item())
                    conf = float(b.conf.item())
                except Exception:
                    continue
                if conf < 0.15:
                    continue
                if isinstance(model_names, dict) and cls_idx in model_names:
                    cls_name = str(model_names[cls_idx])
                else:
                    cls_name = XRAY_CLASSES[cls_idx] if 0 <= cls_idx < len(XRAY_CLASSES) else str(cls_idx)
                class_to_conf[cls_name] = max(class_to_conf.get(cls_name, 0.0), conf)

    # 2) Classification path (probs)
    if not class_to_conf:
        for r in results:
            probs = getattr(r, 'probs', None)
            if probs is None:
                continue
            # Try Ultralytics convenience attributes; fallback to numpy from tensor
            top5_idx, top5_conf = [], []
            try:
                top5_idx = list(getattr(probs, 'top5', []) or [])
                top5_conf = list(getattr(probs, 'top5conf', []) or [])
            except Exception:
                top5_idx, top5_conf = [], []
            if (not top5_idx or not top5_conf) and hasattr(probs, 'data'):
                try:
                    arr = probs.data.detach().cpu().numpy() if np is not None else None
                    if arr is not None and arr.size > 0:
                        order = np.argsort(arr)[::-1]
                        order = order[:5]
                        top5_idx = order.tolist()
                        top5_conf = [float(arr[i]) for i in order]
                except Exception:
                    top5_idx, top5_conf = [], []
            for idx, conf in zip(top5_idx, top5_conf):
                try:
                    cls_idx = int(idx)
                    conf_val = float(conf)
                except Exception:
                    continue
                if conf_val < 0.01:
                    continue
                if isinstance(model_names, dict) and cls_idx in model_names:
                    cls_name = str(model_names[cls_idx])
                else:
                    cls_name = XRAY_CLASSES[cls_idx] if 0 <= cls_idx < len(XRAY_CLASSES) else str(cls_idx)
                class_to_conf[cls_name] = max(class_to_conf.get(cls_name, 0.0), conf_val)

    return sorted([(k, v) for k, v in class_to_conf.items()], key=lambda x: x[1], reverse=True)
def login_required(view_func):
    """Decorator to check if user is logged in"""
    def wrapper(request, *args, **kwargs):
        if not request.session.get('is_authenticated'):
            messages.error(request, "Please log in to access this page.")
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper


def home(request):
    """Home page view - redirects to login"""
    if not request.session.get('is_authenticated'):
        return redirect('login')
    return render(request, "home.html")


@login_required
def about(request):
    """About page view"""
    return render(request, "about.html")


@login_required
def contact(request):
    """Contact page view"""
    if request.method == "POST":
        # Handle contact form submission
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        
        # Here you would typically save to database or send email
        # For now, we'll just show a success message
        messages.success(request, f"Thank you {name}! Your message has been sent successfully. We'll get back to you soon.")
        return redirect('contact')
    
    return render(request, "contact.html")


def login(request):
    """Login page view"""
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Simple demo authentication (in real app, use Django's auth system)
        if username and password:
            # For demo purposes, accept any non-empty username/password
            request.session['is_authenticated'] = True
            request.session['username'] = username
            messages.success(request, f"Welcome back, {username}!")
            return redirect('home')
        else:
            messages.error(request, "Please provide both username and password.")
    
    return render(request, "login.html")


def signup(request):
    """Signup page view"""
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        # Here you would typically create a new user
        # For now, we'll just show a demo message
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
        elif username and email and password:
            messages.success(request, f"Welcome {username}! Your account has been created successfully. Please log in.")
            return redirect('login')
        else:
            messages.error(request, "Please fill in all required fields.")
    
    return render(request, "signup.html")


@login_required
def report_analysis(request):
    """Report analysis page view (renamed from upload_file)"""
    extracted_text = ""
    summary = ""
    error_message = ""
    xray_results = []

    if request.method == "POST" and request.FILES.get("file"):
        analysis_type = request.POST.get('analysis_type', 'text').strip()
        uploaded_file = request.FILES["file"]
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        # X-ray branch (images only)
        if analysis_type == 'xray':
            supported_img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp')
            if not uploaded_file.name.lower().endswith(supported_img_ext):
                error_message = "X-ray mode requires an image file (PNG/JPG/JPEG/BMP/TIFF/GIF/WEBP)."
            else:
                try:
                    img = Image.open(full_path)
                    # Heuristic gate: ensure the image looks like a chest X-ray
                    if not is_xray_like_image(img):
                        error_message = "This image does not appear to be a chest X-ray. Please upload a clear chest X-ray image."
                    else:
                        xray_results = run_xray_inference(img)
                        if not xray_results:
                            # No confident finding: report as fine
                            summary = "The report claims that the person is fine."
                        else:
                            # Only show top-1 result in AI Medical Analysis
                            top_label, top_conf = xray_results[0]
                            summary = f"The report claims that the person has the following: {top_label} ({top_conf*100:.1f}%)."
                except Exception as e:
                    error_message = f"X-ray analysis failed: {e}"

        # Text report: Extract from PDF with improved settings and fallbacks
        elif uploaded_file.name.lower().endswith(".pdf"):
            with pdfplumber.open(full_path) as pdf:
                page_texts = []
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2, use_text_flow=True) or ""
                    if not text.strip():
                        # second attempt with wider tolerances
                        text = page.extract_text(x_tolerance=3.5, y_tolerance=3.5, use_text_flow=True) or ""
                    page_texts.append(text.strip())
                extracted_text = "\n\n".join([t for t in page_texts if t])

            # Fallback: scanned PDF without text layer → rasterize pages and OCR
            if not extracted_text.strip():
                if fitz is None:
                    extracted_text = (
                        "No selectable text found in PDF. OCR fallback requires PyMuPDF. "
                        "Please install PyMuPDF (pip install PyMuPDF) and try again."
                    )
                else:
                    try:
                        doc = fitz.open(full_path)
                        ocr_texts = []
                        for page in doc:
                            # Render page to image with reasonable resolution for OCR
                            pix = page.get_pixmap(dpi=200)
                            img_bytes = pix.tobytes("png")
                            pil_img = Image.open(BytesIO(img_bytes))
                            ocr_text = pytesseract.image_to_string(pil_img) or ""
                            if ocr_text.strip():
                                ocr_texts.append(ocr_text.strip())
                        doc.close()
                        extracted_text = "\n\n".join(ocr_texts).strip()
                    except Exception as e:
                        extracted_text = f"OCR fallback failed: {e}"

        # Text report: Extract from Image (handle various extensions)
        elif analysis_type == 'text':
            supported_img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp')
            if uploaded_file.name.lower().endswith(supported_img_ext):
                try:
                    img = Image.open(full_path)
                    extracted_text = pytesseract.image_to_string(img)
                except Exception as imerr:
                    extracted_text = f"Image could not be processed: {imerr}"
            else:
                extracted_text = "Unsupported image file type. Please upload a valid image file."
        else:
            error_message = "Invalid analysis type."

        os.remove(full_path)  # cleanup uploaded file

        # Text-mode validations and AI
        if analysis_type == 'text':
            # First: if no text at all, show unified non-medical error
            if not extracted_text.strip():
                error_message = "This file does not contain any text related to a medical report. Please upload a valid medical report."
                extracted_text = ""
                summary = ""
            else:
                # Diagram/image-only detection (low/no meaningful text or mostly non-alphanumeric)
                def is_diagram_text(text):
                    txt = text.strip()
                    if len(txt) < 20:  # too short
                        return True
                    alpha_count = sum(c.isalnum() for c in txt)
                    if (alpha_count / len(txt)) < 0.5:
                        return True
                    words = set(w for w in txt.split() if len(w) > 3)
                    if len(words) < 3:
                        return True
                    return False

                if is_diagram_text(extracted_text):
                    error_message = "This file does not appear to be a medical report. Only medical lab/reports are accepted."
                    extracted_text = ""
                    summary = ""
                elif not is_medical_report_text(extracted_text):
                    error_message = "This file does not appear to be a medical report. Only medical lab/reports are accepted."
                    extracted_text = ""
                    summary = ""

            # Send to Gemini for medical analysis
            if extracted_text.strip():
                prompt = f"""
                You are a medical report analysis assistant. 
                Analyze the following extracted text from a patient's medical report 
                and provide a concise summary: possible diseases, infections, or abnormalities.

                Report Text:
                {extracted_text}
                """
                try:
                    response = model.generate_content(prompt)
                    summary = response.text
                except Exception as e:
                    summary = f"Error while contacting Gemini API: {e}"
            prompt = f"""
            You are a medical report analysis assistant. 
            Analyze the following extracted text from a patient's medical report 
            and provide a concise summary: possible diseases, infections, or abnormalities.

            Report Text:
            {extracted_text}
            """
            try:
                response = model.generate_content(prompt)
                summary = response.text
            except Exception as e:
                summary = f"Error while contacting Gemini API: {e}"

    return render(request, "upload.html", {
        "extracted_text": extracted_text,
        "summary": summary,
        "error_message": error_message,
        "xray_results": xray_results
    })


# Keep the old function name for backward compatibility
def upload_file(request):
    """Alias for report_analysis for backward compatibility"""
    return report_analysis(request)


def logout(request):
    """Logout view"""
    request.session.flush()
    messages.success(request, "You have been logged out successfully.")
    return redirect('login')


@login_required
def chatbot(request):
    """Simple chatbot page using Gemini for medical Q&A."""
    # Initialize history in session
    chat_history = request.session.get('chat_history', [])

    if request.method == "POST":
        user_message = request.POST.get('message', '').strip()

        if not user_message:
            return JsonResponse({"error": "Empty message."}, status=400)

        prompt = f"""
        You are a helpful, safety-conscious medical report and diagnostic test assistant chatbot. You should only answer questions related to medical reports, lab test results, medical parameters, and diagnostic interpretations.
Provide clear, concise, and factual explanations about medical test values, their possible meanings, and general health context.

If the user asks for personal medical advice, diagnosis, or treatment recommendations, politely decline and include a brief disclaimer such as:

“I can only help interpret medical reports and test results. For personalized medical advice, please consult a qualified healthcare professional.”

Always maintain a professional, accurate, and empathetic tone.

        User question:
        {user_message}
        """

        try:
            response = model.generate_content(prompt)
            answer = getattr(response, 'text', '').strip() or "Sorry, I couldn't generate a response."
        except Exception as e:
            return JsonResponse({"error": f"Gemini error: {e}"}, status=500)

        # Save to session history and persist
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})
        request.session['chat_history'] = chat_history
        request.session.modified = True

        return JsonResponse({"reply": answer})

    # GET: render with any existing history
    return render(request, "chatbot.html", {"chat_history": chat_history})

# Utility: check if text is a likely medical report
MEDICAL_KEYWORDS = [
    "hemoglobin", "glucose", "cholesterol", "WBC", "RBC", "platelet", "urine", "serum",
    "testosterone", "estradiol", "bilirubin", "creatinine", "ALT", "AST", "thyroid", "TSH",
    "HDL", "LDL", "triglyceride", "ECG", "x-ray", "MRI", "CT scan",
    "reference range", "normal value", "diagnosis", "specimen", "doctor", "consultant", "patient",
    "physician", "investigation", "microscopy", "Positive", "Negative", "Medical Center"
]

def is_medical_report_text(text):
    text_lower = text.lower()
    hits = sum(kw.lower() in text_lower for kw in MEDICAL_KEYWORDS)
    # Accept if at least 2 medical phrases found and enough unique matches
    return hits >= 2 or (hits >= 1 and sum(text_lower.count(kw.lower()) for kw in MEDICAL_KEYWORDS) > 2)
