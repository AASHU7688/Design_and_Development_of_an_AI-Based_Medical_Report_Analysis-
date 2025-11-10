from django.urls import path
from .views import home, about, contact, login, signup, report_analysis, upload_file, logout, chatbot

urlpatterns = [
    path('', home, name='home'),
    path('about/', about, name='about'),
    path('contact/', contact, name='contact'),
    path('login/', login, name='login'),
    path('signup/', signup, name='signup'),
    path('logout/', logout, name='logout'),
    path('report-analysis/', report_analysis, name='report_analysis'),
    path('upload/', upload_file, name='upload_file'),  # Keep for backward compatibility
    path('chatbot/', chatbot, name='chatbot'),
]
