# Fire Detection with YOLOv8

This project detects fire and smoke using a YOLOv8 model.

## Setup & Execution

### 1. Using the Virtual Environment (Recommended)
A virtual environment is already set up in the `.venv` directory. To run the script using this environment:

**On Windows:**
```powershell
.\.venv\Scripts\python main.py --source fireIMg.jpg
```

Or simply run the provided batch file:
```powershell
.\run.bat
```

### 2. VS Code Configuration
If you see "Could not find import" errors in VS Code:
1. Press `Ctrl + Shift + P`.
2. Type and select **"Python: Select Interpreter"**.
3. Choose the interpreter located at `.\.venv\Scripts\python.exe`.

This will allow VS Code to recognize the `ultralytics` package and other dependencies.

## Dependencies
- `ultralytics` (YOLOv8)
- `opencv-python`
- `torch`
