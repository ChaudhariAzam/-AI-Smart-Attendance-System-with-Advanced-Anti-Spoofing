from django.contrib import admin

# Register your models here.
# admin.py
from django.contrib import admin
from .models import User, Attendance

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'name')  # show these fields in the admin

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('get_user_name', 'punch_time')  # display user name instead of object

    # Custom method to fetch the user's name
    def get_user_name(self, obj):
        return obj.user.name
    get_user_name.short_description = 'User Name'  # column header in admin
