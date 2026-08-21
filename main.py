"""
Main entry point for PDFLib Visual Search & Computer Vision PDF Parser.
Allows launching the Streamlit web application or running visual searches from the terminal.
"""
import sys
import os
import subprocess

def main():
    print("=" * 60)
    print("  PDFLib - Computer Vision Visual PDF Matcher")
    print("=" * 60)
    print("Starting Streamlit Web Application on http://localhost:8501...")
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    python_exe = sys.executable
    
    cmd = [python_exe, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped.")

if __name__ == "__main__":
    main()