"""Mini-functions that do not belong elsewhere."""
from datetime import datetime


def current_time(template="%Y%m%d %H:%M:%S"):
    return datetime.now().strftime(template)
