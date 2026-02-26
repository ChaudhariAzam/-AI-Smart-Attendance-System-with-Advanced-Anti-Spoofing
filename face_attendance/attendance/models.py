from django.db import models
from datetime import datetime
import pytz  # pip install pytz

# Utility function to get IST datetime
def get_kolkata_time(dt=None):
    kolkata = pytz.timezone('Asia/Kolkata')
    if dt is None:
        dt = datetime.now(tz=pytz.UTC)  # start with UTC
    return dt.astimezone(kolkata)  # keep tzinfo

class User(models.Model):
    user_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    embedding = models.BinaryField()  # store face embedding

# class Attendance(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     punch_time = models.DateTimeField()  # we will manually set this
class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    punch_time = models.DateTimeField(default=get_kolkata_time)  # sets IST automatically
