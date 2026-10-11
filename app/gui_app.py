"""Desktop GUI Application for Bird Species Classification.

Matches the Tkinter desktop interface for uploading bird photos
and running CNN classification with Top-3 confidence bars.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import BirdPredictor


class CustomProgressBar(tk.Canvas):
    """Custom progress bar canvas widget matching the exact UI mockup."""

    def __init__(self, parent, width=180, height=18, bg_color="#e5e7eb", fill_color="#10b981", border_color="#d1d5db", **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0, **kwargs)
        self.bar_width = width
        self.bar_height = height
        self.bg_color = bg_color
        self.fill_color = fill_color
        self.border_color = border_color
        self.progress = 0.0
        self.draw()

    def set_progress(self, ratio: float):
        """Set ratio between 0.0 and 1.0."""
        self.progress = max(0.0, min(1.0, float(ratio)))
        self.draw()

    def draw(self):
        self.delete("all")
        # Draw background trough
        self.create_rectangle(
            0, 0, self.bar_width, self.bar_height,
            fill=self.bg_color, outline=self.border_color, width=1
        )
        # Draw filled portion
        fill_w = int(self.bar_width * self.progress)
        if fill_w > 0:
            self.create_rectangle(
                1, 1, fill_w, self.bar_height - 1,
                fill=self.fill_color, outline=""
            )


class BirdClassifierGUI:
    """Tkinter Desktop Interface for Bird Species Classification."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bird Species Classifier")
        self.root.geometry("540x690")
        self.root.minsize(480, 620)
        self.root.configure(bg="#f3f4f6")

        # Application state
        self.predictor = None
        self.current_image_path = None
        self.current_pil_image = None
        self.tk_image = None
        self.preview_size = (280, 260)

        # Build UI layout
        self._init_styles()
        self._build_header()
        self._build_image_area()
        self._build_action_buttons()
        self._build_prediction_display()
        self._build_top3_panel()
        self._build_status_bar()

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.browse_image())
        self.root.bind("<Return>", lambda e: self.classify_image())
        self.root.bind("<Escape>", lambda e: self.clear_all())

        # Load model asynchronously to avoid startup delay
        self.status_var.set("Loading neural network model...")
        threading.Thread(target=self._init_predictor, daemon=True).start()

    def _init_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(
            "TButton",
            font=("Segoe UI", 9),
            padding=(10, 4)
        )
        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 9, "bold"),
            padding=(12, 4)
        )

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg="#f3f4f6")
        header_frame.pack(fill=tk.X, pady=(16, 8), padx=20)

        title_label = tk.Label(
            header_frame,
            text="Bird Species Classifier",
            font=("Segoe UI", 18, "bold"),
            bg="#f3f4f6",
            fg="#111827"
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Upload a bird image and the CNN will identify the species",
            font=("Segoe UI", 9),
            bg="#f3f4f6",
            fg="#4b5563"
        )
        subtitle_label.pack(pady=(2, 0))

    def _build_image_area(self):
        container = tk.Frame(self.root, bg="#f3f4f6")
        container.pack(pady=(4, 12))

        # Box matching the grey framed container in the mockup
        self.image_canvas = tk.Canvas(
            container,
            width=self.preview_size[0],
            height=self.preview_size[1],
            bg="#e5e7eb",
            highlightthickness=1,
            highlightbackground="#9ca3af"
        )
        self.image_canvas.pack()

        # Placeholder text in canvas
        self.placeholder_text = self.image_canvas.create_text(
            self.preview_size[0] // 2,
            self.preview_size[1] // 2,
            text="No image loaded\n\nClick 'Browse Image...' or 'Sample'",
            fill="#6b7280",
            font=("Segoe UI", 10),
            justify=tk.CENTER
        )

    def _build_action_buttons(self):
        btn_frame = tk.Frame(self.root, bg="#f3f4f6")
        btn_frame.pack(pady=(0, 14))

        self.btn_browse = ttk.Button(
            btn_frame,
            text="Browse Image...",
            command=self.browse_image
        )
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        self.btn_classify = ttk.Button(
            btn_frame,
            text="Classify",
            style="Primary.TButton",
            command=self.classify_image
        )
        self.btn_classify.pack(side=tk.LEFT, padx=5)

        self.btn_sample = ttk.Button(
            btn_frame,
            text="Sample Image",
            command=self.load_random_sample
        )
        self.btn_sample.pack(side=tk.LEFT, padx=5)

        self.btn_clear = ttk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_all
        )
        self.btn_clear.pack(side=tk.LEFT, padx=5)

    def _build_prediction_display(self):
        pred_frame = tk.Frame(self.root, bg="#f3f4f6")
        pred_frame.pack(fill=tk.X, pady=(0, 8), padx=20)

        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.prediction_label = tk.Label(
            pred_frame,
            textvariable=self.prediction_var,
            font=("Segoe UI", 14, "bold"),
            fg="#0b57d0",
            bg="#f3f4f6"
        )
        self.prediction_label.pack()

    def _build_top3_panel(self):
        card_outer = tk.Frame(self.root, bg="#f3f4f6")
        card_outer.pack(fill=tk.X, padx=30, pady=(0, 12))

        # Panel frame matching the screenshot
        panel = tk.Frame(
            card_outer,
            bg="#ffffff",
            bd=1,
            relief=tk.SOLID,
            highlightthickness=0
        )
        panel.pack(fill=tk.X, ipady=8, ipadx=10)

        title = tk.Label(
            panel,
            text="Top 3 predictions",
            font=("Segoe UI", 8, "bold"),
            fg="#4b5563",
            bg="#ffffff"
        )
        title.pack(anchor="w", padx=10, pady=(2, 6))

        self.pred_rows = []
        for _ in range(3):
            row_frame = tk.Frame(panel, bg="#ffffff")
            row_frame.pack(fill=tk.X, padx=10, pady=3)

            name_lbl = tk.Label(
                row_frame,
                text="—",
                font=("Segoe UI", 8, "bold"),
                fg="#1f2937",
                bg="#ffffff",
                width=24,
                anchor="w"
            )
            name_lbl.pack(side=tk.LEFT)

            pbar = CustomProgressBar(
                row_frame,
                width=170,
                height=14,
                bg_color="#e5e7eb",
                fill_color="#00aa3c",
                border_color="#d1d5db"
            )
            pbar.pack(side=tk.LEFT, padx=(4, 10))

            pct_lbl = tk.Label(
                row_frame,
                text="0.0%",
                font=("Segoe UI", 8),
                fg="#374151",
                bg="#ffffff",
                width=6,
                anchor="e"
            )
            pct_lbl.pack(side=tk.RIGHT)

            self.pred_rows.append({
                "name": name_lbl,
                "bar": pbar,
                "pct": pct_lbl
            })

    def _build_status_bar(self):
        status_frame = tk.Frame(self.root, bg="#e5e7eb", bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 8),
            fg="#374151",
            bg="#e5e7eb",
            anchor="w",
            padx=8,
            pady=3
        )
        status_label.pack(fill=tk.X)

    def _init_predictor(self):
        """Asynchronously initialize the BirdPredictor."""
        checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(PROJECT_ROOT, "models", "resnet18_transfer_best.pth")

        if not os.path.exists(checkpoint_path):
            self.root.after(0, lambda: self.status_var.set("Model checkpoint not found in models/"))
            return

        try:
            self.predictor = BirdPredictor(checkpoint_path=checkpoint_path, device_setting="auto")
            device_name = str(self.predictor.device).upper()
            self.root.after(0, lambda: self.status_var.set(f"Ready ({device_name} inference)"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error loading model: {e}"))

    def load_random_sample(self):
        """Pick a sample image from dataset and load it."""
        import glob
        import random
        sample_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "processed", "test", "*", "*.jpg"))
        if not sample_files:
            sample_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "**", "*.jpg"), recursive=True)
        if sample_files:
            chosen = random.choice(sample_files)
            self.load_image(chosen)
        else:
            messagebox.showinfo("No Samples", "No sample images found in data directory.")

    def browse_image(self):
        """Open a file dialog to choose an image."""
        default_dir = os.path.join(PROJECT_ROOT, "data", "processed", "test")
        if not os.path.exists(default_dir):
            default_dir = PROJECT_ROOT

        file_types = [
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        chosen = filedialog.askopenfilename(
            title="Select Bird Image",
            initialdir=default_dir,
            filetypes=file_types
        )
        if chosen:
            self.load_image(chosen)

    def load_image(self, image_path: str):
        """Load and display an image on the preview canvas."""
        try:
            self.current_image_path = image_path
            pil_img = Image.open(image_path).convert("RGB")
            self.current_pil_image = pil_img

            # Resize maintaining aspect ratio to fit inside canvas
            w, h = pil_img.size
            max_w, max_h = self.preview_size
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized)

            self.image_canvas.delete("all")
            # Center in canvas
            cx = self.preview_size[0] // 2
            cy = self.preview_size[1] // 2
            self.image_canvas.create_image(cx, cy, image=self.tk_image)

            fname = os.path.basename(image_path)
            self.status_var.set(f"Loaded: {fname} (Click 'Classify' to run model)")
            self.prediction_var.set("Prediction: -")
            self._reset_top3_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def classify_image(self):
        """Execute CNN model prediction on current image."""
        if not self.current_pil_image:
            messagebox.showinfo("No Image", "Please browse and select a bird image first.")
            return

        if self.predictor is None:
            messagebox.showwarning("Model Loading", "The neural network is still loading. Please wait a moment.")
            return

        self.status_var.set("Running CNN inference...")
        self.btn_classify.state(["disabled"])

        threading.Thread(target=self._run_inference_worker, daemon=True).start()

    def _run_inference_worker(self):
        try:
            results = self.predictor.predict(self.current_pil_image, top_k=3)
            self.root.after(0, lambda: self._apply_prediction_results(results))
        except Exception as e:
            self.root.after(0, lambda: self._handle_prediction_error(str(e)))

    def _apply_prediction_results(self, results: list):
        self.btn_classify.state(["!disabled"])
        if not results:
            self.status_var.set("No prediction returned.")
            return

        top = results[0]
        species = top["species"]
        conf_pct = top["percentage"]

        self.prediction_var.set(f"Prediction: {species}")
        self.status_var.set(f"Predicted {species} ({conf_pct} confidence)")

        for i, row in enumerate(self.pred_rows):
            if i < len(results):
                res = results[i]
                row["name"].config(text=res["species"])
                row["bar"].set_progress(res["probability"])
                row["pct"].config(text=res["percentage"])
            else:
                row["name"].config(text="-")
                row["bar"].set_progress(0.0)
                row["pct"].config(text="0.0%")

    def _handle_prediction_error(self, err_msg: str):
        self.btn_classify.state(["!disabled"])
        self.status_var.set(f"Prediction error: {err_msg}")
        messagebox.showerror("Prediction Error", err_msg)

    def _reset_top3_display(self):
        for row in self.pred_rows:
            row["name"].config(text="-")
            row["bar"].set_progress(0.0)
            row["pct"].config(text="0.0%")

    def clear_all(self):
        """Reset image canvas, predictions, and status."""
        self.current_image_path = None
        self.current_pil_image = None
        self.tk_image = None

        self.image_canvas.delete("all")
        self.image_canvas.create_text(
            self.preview_size[0] // 2,
            self.preview_size[1] // 2,
            text="No image loaded\n\nClick 'Browse Image...' or 'Sample'",
            fill="#6b7280",
            font=("Segoe UI", 10),
            justify=tk.CENTER
        )

        self.prediction_var.set("Prediction: -")
        self._reset_top3_display()
        self.status_var.set("Ready")


def launch_gui():
    root = tk.Tk()
    app = BirdClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
