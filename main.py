import tkinter as tk
from src.gui.nationality_gui import NationalityGUI

def main():
    root = tk.Tk()
    app = NationalityGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF logs
    
    print("\n[System] Initializing Nationality Detection Environment...")
    print("[System] Loading Graphical Interface...")
    
    main()
