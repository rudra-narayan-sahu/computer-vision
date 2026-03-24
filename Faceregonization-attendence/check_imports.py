try:
    import cv2
    print("cv2 imported successfully, version:", cv2.__version__)
except ImportError as e:
    print("cv2 NOT found:", e)

try:
    import numpy
    print("numpy imported successfully, version:", numpy.__version__)
except ImportError as e:
    print("numpy NOT found:", e)

try:
    import face_recognition
    print("face_recognition imported successfully")
except ImportError as e:
    print("face_recognition NOT found:", e)
    print("Note: face_recognition requires dlib, which can be hard to install on Windows.")

try:
    import os
    print("os imported successfully")
except ImportError as e:
    print("os NOT found:", e)

try:
    from datetime import datetime
    print("datetime imported successfully")
except ImportError as e:
    print("datetime NOT found:", e)
