# # attendance/urls.py
# from django.urls import path
# from .views import UserRegisterView, PunchAttendanceView

# urlpatterns = [
#     path('register/', UserRegisterView.as_view(), name='register'),
#     path('punch/', PunchAttendanceView.as_view(), name='punch'),
# ]
from django.urls import path
from .views import UserRegisterView, PunchAttendanceView, attendance_page

urlpatterns = [
    path('', attendance_page, name='attendance_page'),
    path('register/', UserRegisterView.as_view(), name='register'),
    path('punch/', PunchAttendanceView.as_view(), name='punch'),
]
