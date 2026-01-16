import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import cv2
from PIL import Image, ImageTk
import sys
import os

# Append project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.modules.nationality_engine import NationalityEngine

class NationalityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nationality & Logic System") # Simple academic title
        self.root.geometry("1100x700")
        
        # Standard Theme (Clean, default look)
        style = ttk.Style()
        style.theme_use('clam') # 'clam' is standard and clean without being 'fancy'
        
        # Main Layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Side: Video/Image Preview
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Input Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right Side: Logic & Logs
        self.info_panel = ttk.Frame(self.main_frame, width=300)
        self.info_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Log Box
        self.log_group = ttk.LabelFrame(self.info_panel, text="Execution Logs")
        self.log_group.pack(fill=tk.BOTH, expand=True)
        
        # Standard white text box
        self.log_text = scrolledtext.ScrolledText(self.log_group, height=20, width=40, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls
        self.btn_frame = ttk.LabelFrame(self.info_panel, text="Controls")
        self.btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(self.btn_frame, text="Run System", command=self.start_camera)
        self.start_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.upload_btn = ttk.Button(self.btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.quit_btn = ttk.Button(self.btn_frame, text="Close", command=self.on_close)
        self.quit_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Engine State
        self.engine = None
        self.loading_thread = None
        self.cap = None
        self.running = False
        
        # Auto-load in background (Good UX, but standard implementation)
        self.start_loading()

    def start_loading(self):
        import threading
        self.start_btn.config(state='disabled', text="Loading Models...")
        self.loading_thread = threading.Thread(target=self._load_engine)
        self.loading_thread.daemon = True
        self.loading_thread.start()
        self.check_loading()

    def _load_engine(self):
        self.engine = NationalityEngine()

    def check_loading(self):
        if self.loading_thread.is_alive():
            self.root.after(100, self.check_loading)
        else:
            self.start_btn.config(state='normal', text="Run System")
            self.log("Models loaded. Read to start.")

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.log("Camera initializing...")
            self.update_frame()

    def update_frame(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    if self.engine:
                        results = self.engine.analyze_frame(frame)
                        
                        for res in results:
                            x, y, w, h = res['bbox']
                            nat = res['nationality']
                            branch = res['branch']
                            attrs = res['attributes']
                            
                            # Box color based on branch
                            if branch == "Indian":
                                color = (0, 165, 255) # Orange
                            elif branch == "United States":
                                color = (255, 100, 100) # Light Blue
                            elif branch == "African":
                                color = (0, 200, 200) # Yellow
                            else:
                                color = (0, 255, 0) # Green
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                            
                            # Nationality label with background
                            label = nat
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
                            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            
                            # Attributes panel (right side of face)
                            panel_x = x + w + 10
                            panel_y = y
                            
                            for k, v in attrs.items():
                                text = f"{k}: {v}"
                                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                # Background for readability
                                cv2.rectangle(frame, (panel_x - 2, panel_y), (panel_x + tw + 5, panel_y + th + 8), (50, 50, 50), -1)
                                cv2.putText(frame, text, (panel_x, panel_y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                panel_y += th + 12
                except Exception as e:
                    # Log error but continue showing feed
                    cv2.putText(frame, "Processing Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Always show frame in GUI
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            self.root.after(15, self.update_frame)

    def upload_image(self):
        """Handle image upload and analysis."""
        # Stop camera if running
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.log("Camera stopped for image upload.")
        
        # Open file dialog
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=filetypes
        )
        
        if not filepath:
            self.log("No image selected.")
            return
        
        self.log(f"Processing: {os.path.basename(filepath)}")
        
        # Load image
        frame = cv2.imread(filepath)
        if frame is None:
            self.log("Error: Could not read image file.")
            return
        
        # Process with engine
        try:
            if self.engine:
                results = self.engine.analyze_frame(frame)
                
                for res in results:
                    x, y, w, h = res['bbox']
                    nat = res['nationality']
                    branch = res['branch']
                    attrs = res['attributes']
                    
                    # Box color based on branch
                    if branch == "Indian":
                        color = (0, 165, 255)  # Orange
                    elif branch == "United States":
                        color = (255, 100, 100)  # Light Blue
                    elif branch == "African":
                        color = (0, 200, 200)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Nationality label with background
                    label = nat
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Attributes panel (right side of face)
                    panel_x = x + w + 10
                    panel_y = y
                    
                    for k, v in attrs.items():
                        text = f"{k}: {v}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (panel_x - 2, panel_y), (panel_x + tw + 5, panel_y + th + 8), (50, 50, 50), -1)
                        cv2.putText(frame, text, (panel_x, panel_y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        panel_y += th + 12
                    
                    # Log results
                    self.log(f"  Nationality: {nat} ({branch})")
                    for k, v in attrs.items():
                        self.log(f"    {k}: {v}")
                
                if not results:
                    self.log("No faces detected in image.")
            else:
                self.log("Engine not loaded yet.")
        except Exception as e:
            self.log(f"Processing error: {str(e)}")
        
        # Display processed image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
