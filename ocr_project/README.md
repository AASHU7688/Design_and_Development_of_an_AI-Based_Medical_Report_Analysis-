# MediScan - Medical Report Analysis Platform

MediScan is a comprehensive web application that provides AI-powered medical report analysis using OCR (Optical Character Recognition) technology and advanced AI models.

## Features

### 🏠 **Home Page**
- Modern, responsive landing page
- Feature highlights and statistics
- Call-to-action sections
- Professional medical theme

### 📊 **Report Analysis**
- Upload medical reports (PDF, PNG, JPG, JPEG)
- Drag & drop file upload interface
- OCR text extraction from images and PDFs
- AI-powered medical analysis using Google Gemini
- Beautiful result display with categorized output

### 👥 **About Page**
- Company mission and values
- Team information
- Company timeline and milestones
- Professional presentation

### 📞 **Contact Page**
- Contact form with validation
- Business hours and location
- Multiple contact methods
- Interactive map placeholder

### 🔐 **Authentication Pages**
- **Login Page**: User authentication with social login options
- **Signup Page**: User registration with password validation
- Form validation and error handling
- Professional UI/UX design

## Technology Stack

- **Backend**: Django 3.2
- **Frontend**: HTML5, CSS3, JavaScript
- **OCR**: Tesseract OCR
- **PDF Processing**: pdfplumber
- **AI Analysis**: Google Gemini API
- **Styling**: Custom CSS with modern design
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Poppins)

## Installation & Setup

### Prerequisites
- Python 3.7+
- Tesseract OCR installed on your system
- Google Gemini API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ocr_project
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Configure Google Gemini API**
   - Get your API key from Google AI Studio
   - Update the API key in `ocr_app/views.py`

5. **Run migrations**
   ```bash
   python manage.py migrate
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Open your browser and go to `http://127.0.0.1:8000/`

## Project Structure

```
ocr_project/
├── ocr_app/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── ocr_project/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── about.html
│   ├── contact.html
│   ├── login.html
│   ├── signup.html
│   └── upload.html
├── manage.py
└── README.md
```

## URL Structure

- **Home**: `/` - Landing page
- **About**: `/about/` - Company information
- **Contact**: `/contact/` - Contact form and information
- **Login**: `/login/` - User authentication
- **Signup**: `/signup/` - User registration
- **Report Analysis**: `/report-analysis/` - Medical report upload and analysis
- **Upload (Legacy)**: `/upload/` - Backward compatibility

## Features in Detail

### Report Analysis
- **File Support**: PDF, PNG, JPG, JPEG formats
- **OCR Processing**: Text extraction from images and PDFs
- **AI Analysis**: Medical report interpretation using Google Gemini
- **User Interface**: Modern drag-and-drop upload with progress indicators
- **Results Display**: Categorized and formatted output

### Security & Privacy
- **HIPAA Compliant**: Designed with healthcare privacy in mind
- **Data Protection**: Secure file handling and processing
- **Form Validation**: Client and server-side validation
- **CSRF Protection**: Built-in Django security features

### Responsive Design
- **Mobile-First**: Optimized for all device sizes
- **Modern UI**: Clean, professional medical theme
- **Accessibility**: WCAG compliant design elements
- **Performance**: Optimized loading and processing

## Customization

### Styling
- All styles are in the respective template files
- Color scheme can be modified in the CSS variables
- Fonts can be changed in the base template

### Functionality
- Add new pages by creating templates and views
- Extend the base template for consistent styling
- Modify the navigation in `base.html`

### API Integration
- Replace Google Gemini with other AI services
- Add additional OCR engines
- Integrate with medical databases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Email: info@mediscan.com
- Phone: +1 (555) 123-4567
- Website: https://mediscan.com

## Future Enhancements

- User authentication and user management
- Report history and storage
- Advanced AI models integration
- Mobile app development
- API endpoints for third-party integration
- Advanced analytics and reporting
- Multi-language support
- Integration with medical databases

---

**MediScan** - Transforming medical report analysis with AI technology.

