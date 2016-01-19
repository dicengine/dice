from datetime import date, datetime
import os

def now():
    now = str(datetime.now())
    now = now[:now.rfind('.')]
    return now

def append_time(text):
    line_length = 40
    text = text + " "*(line_length-len(text)) + now() + "\n"
    return text

def force_write(file):
    file.flush()
    os.fsync(file.fileno())
    return
