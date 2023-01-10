import re
from pathlib import Path

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.area = width * height
        # causes "Explicit return in __init__" error
        return self.area



r = Rectangle(2, 2)