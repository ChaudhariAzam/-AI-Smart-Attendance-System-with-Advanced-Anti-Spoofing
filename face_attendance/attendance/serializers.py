# attendance/serializers.py
from rest_framework import serializers
from .models import User, Attendance

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['user_id', 'name']

class AttendanceSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField()
    class Meta:
        model = Attendance
        fields = ['user', 'punch_time']
