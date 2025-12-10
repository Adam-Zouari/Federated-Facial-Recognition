"""
Face ID Application - Similar to Phone Face Recognition
Uses trained model to add and verify faces.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
from torchvision import transforms
import config
from models import create_model

# Try to import MTCNN for better face detection
try:
    from facenet_pytorch import MTCNN
    USE_MTCNN = True
except ImportError:
    USE_MTCNN = False
    print("Warning: facenet-pytorch not installed. Using Haar Cascade (less accurate).")
    print("Install MTCNN for better face detection: pip install facenet-pytorch")


def preprocess_image(pil_image, augment=False):
    """
    Preprocess a PIL image for model input.
    
    Args:
        pil_image: PIL Image
        augment: Whether to apply augmentation (not used for inference)
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return transform(pil_image)


class FaceIDApp:
    """Face ID Application for enrollment and verification."""
    
    def __init__(self, model_path, database_path='face_database.json'):
        """
        Initialize Face ID Application.
        
        Args:
            model_path: Path to trained model checkpoint
            database_path: Path to save/load face database
        """
        self.device = config.DEVICE
        self.database_path = database_path
        self.face_database = self.load_database()
        
        # Load trained model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine number of classes from checkpoint
        model_state = checkpoint.get('model_state_dict', checkpoint)
        # Get the classifier weight shape to determine num_classes
        classifier_weight_key = None
        for key in model_state.keys():
            if 'classifier.weight' in key or 'fc.weight' in key:
                classifier_weight_key = key
                break
        
        if classifier_weight_key:
            num_classes = model_state[classifier_weight_key].shape[0]
        else:
            num_classes = 480  # Default for VGGFace2
        
        self.model = create_model('mobilenetv2', num_classes=num_classes)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded successfully (num_classes: {num_classes})")
        
        # Face detection
        if USE_MTCNN:
            # Use MTCNN for robust face detection
            self.face_detector = MTCNN(
                keep_all=False,  # Only return best face
                device=self.device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
                post_process=False  # We'll handle preprocessing ourselves
            )
            print("✓ Using MTCNN face detector (deep learning-based)")
        else:
            # Fallback to Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.face_detector = None
            print("✓ Using Haar Cascade face detector (basic)")
        
        # Camera
        self.camera = None
        
        # Similarity threshold for verification
        self.similarity_threshold = 0.7  # Adjust based on your model's performance
        
        # GUI
        self.root = tk.Tk()
        self.root.title("Face ID Application")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        self.setup_gui()
        
    def load_database(self):
        """Load face database from file."""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                db = json.load(f)
                # Convert embedding lists back to tensors
                for name in db:
                    db[name]['embeddings'] = [torch.tensor(emb) for emb in db[name]['embeddings']]
                print(f"✓ Loaded {len(db)} faces from database")
                return db
        return {}
    
    def save_database(self):
        """Save face database to file."""
        # Convert tensors to lists for JSON serialization
        db_serializable = {}
        for name, data in self.face_database.items():
            db_serializable[name] = {
                'embeddings': [emb.tolist() for emb in data['embeddings']],
                'date_added': data['date_added']
            }
        
        with open(self.database_path, 'w') as f:
            json.dump(db_serializable, f, indent=2)
        print(f"✓ Database saved ({len(self.face_database)} faces)")
    
    def setup_gui(self):
        """Setup GUI components."""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🔐 Face ID Application", 
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text=f"Database: {len(self.face_database)} registered faces",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.status_label.pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=30)
        
        # Add Face button
        add_button = tk.Button(
            button_frame,
            text="➕ Add New Face",
            font=('Arial', 16, 'bold'),
            bg='#27ae60',
            fg='white',
            width=20,
            height=2,
            command=self.add_new_face,
            cursor='hand2'
        )
        add_button.grid(row=0, column=0, padx=20)
        
        # Test Face button
        test_button = tk.Button(
            button_frame,
            text="🔍 Test Face",
            font=('Arial', 16, 'bold'),
            bg='#3498db',
            fg='white',
            width=20,
            height=2,
            command=self.test_face,
            cursor='hand2'
        )
        test_button.grid(row=0, column=1, padx=20)
        
        # View Database button
        view_button = tk.Button(
            button_frame,
            text="📋 View Database",
            font=('Arial', 14),
            bg='#95a5a6',
            fg='white',
            width=20,
            command=self.view_database,
            cursor='hand2'
        )
        view_button.grid(row=1, column=0, pady=20)
        
        # Delete Face button
        delete_button = tk.Button(
            button_frame,
            text="🗑️ Delete Face",
            font=('Arial', 14),
            bg='#e74c3c',
            fg='white',
            width=20,
            command=self.delete_face,
            cursor='hand2'
        )
        delete_button.grid(row=1, column=1, pady=20)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Add your face with different poses for better recognition\n"
                 "Test your face to verify identity",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#bdc3c7',
            justify='center'
        )
        instructions.pack(pady=20)
        
    def extract_embedding(self, image):
        """
        Extract face embedding from image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Embedding tensor (128-dim) or None if no face detected
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect face
        if USE_MTCNN and self.face_detector is not None:
            # Use MTCNN for detection
            pil_image = Image.fromarray(image_rgb)
            boxes, probs = self.face_detector.detect(pil_image)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # Use the most confident detection
            best_idx = np.argmax(probs)
            x1, y1, x2, y2 = boxes[best_idx].astype(int)
            
            # Add padding
            width = x2 - x1
            height = y2 - y1
            padding = int(0.2 * max(width, height))
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_rgb.shape[1], x2 + padding)
            y2 = min(image_rgb.shape[0], y2 + padding)
            
            # Crop face
            face_img = image_rgb[y1:y2, x1:x2]
        else:
            # Use Haar Cascade for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding
            padding = int(0.2 * max(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop face
            face_img = image_rgb[y:y+h, x:x+w]
        
        # Preprocess
        face_pil = Image.fromarray(face_img)
        face_tensor = preprocess_image(face_pil, augment=False)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            # Model returns (logits, embeddings) when return_embedding=True
            _, embedding = self.model(face_tensor, return_embedding=True)
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.squeeze(0).cpu()
    
    def capture_poses(self, name):
        """
        Capture multiple face poses for enrollment.
        
        Args:
            name: Name of the person
            
        Returns:
            List of embeddings or None if failed
        """
        poses = [
            "Look straight at the camera",
            "Turn your head slightly LEFT",
            "Turn your head slightly RIGHT",
            "Tilt your head slightly UP",
            "Tilt your head slightly DOWN"
        ]
        
        embeddings = []
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Cannot access camera!")
            return None
        
        for pose_idx, pose in enumerate(poses):
            # Create capture window
            window_name = f"Enrolling {name} - Pose {pose_idx + 1}/{len(poses)}"
            
            captured = False
            
            while not captured:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Detect faces
                display_frame = frame.copy()
                
                if USE_MTCNN and self.face_detector is not None:
                    # Use MTCNN
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    boxes, probs = self.face_detector.detect(pil_frame)
                    
                    if boxes is not None:
                        for box, prob in zip(boxes, probs):
                            if prob > 0.9:  # High confidence
                                x1, y1, x2, y2 = box.astype(int)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f'{prob:.2f}', (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Use Haar Cascade
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add instructions
                cv2.putText(display_frame, pose, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel", 
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 32:  # Space bar
                    # Check if face detected
                    face_detected = False
                    if USE_MTCNN and self.face_detector is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        boxes, probs = self.face_detector.detect(pil_frame)
                        face_detected = boxes is not None and len(boxes) > 0
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        face_detected = len(faces) > 0
                    
                    if face_detected:
                        embedding = self.extract_embedding(frame)
                        if embedding is not None:
                            embeddings.append(embedding)
                            captured = True
                            print(f"✓ Captured pose {pose_idx + 1}/{len(poses)}")
                        else:
                            messagebox.showwarning("Warning", "Face not detected clearly. Try again.")
                    else:
                        messagebox.showwarning("Warning", "No face detected. Please position your face in the frame.")
                
                elif key == 27:  # ESC
                    cv2.destroyWindow(window_name)
                    self.camera.release()
                    return None
            
            cv2.destroyWindow(window_name)
        
        self.camera.release()
        
        return embeddings
    
    def add_new_face(self):
        """Add a new face to the database."""
        # Ask for name
        name = simpledialog.askstring("Add New Face", "Enter your name:")
        
        if not name:
            return
        
        # Check if name already exists
        if name in self.face_database:
            if not messagebox.askyesno("Name Exists", 
                                       f"{name} already exists. Do you want to update their face data?"):
                return
        
        # Capture multiple poses
        messagebox.showinfo("Instructions", 
                           "We will capture 5 different poses of your face.\n\n"
                           "Position your face in the green rectangle and press SPACE to capture.\n"
                           "Press ESC to cancel.")
        
        embeddings = self.capture_poses(name)
        
        if embeddings is None or len(embeddings) == 0:
            messagebox.showwarning("Cancelled", "Face enrollment cancelled.")
            return
        
        # Save to database
        self.face_database[name] = {
            'embeddings': embeddings,
            'date_added': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.save_database()
        
        self.status_label.config(text=f"Database: {len(self.face_database)} registered faces")
        
        messagebox.showinfo("Success", 
                           f"✓ {name} has been successfully enrolled!\n"
                           f"Captured {len(embeddings)} poses.")
    
    def test_face(self):
        """Test face against database."""
        if len(self.face_database) == 0:
            messagebox.showwarning("Empty Database", "No faces in database. Please add a face first.")
            return
        
        messagebox.showinfo("Instructions", 
                           "Look at the camera and press SPACE to verify.\n"
                           "Press ESC to cancel.")
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Cannot access camera!")
            return
        
        window_name = "Face Verification - Press SPACE to verify"
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Detect faces
            display_frame = frame.copy()
            
            if USE_MTCNN and self.face_detector is not None:
                # Use MTCNN
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                boxes, probs = self.face_detector.detect(pil_frame)
                
                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        if prob > 0.9:  # High confidence
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(display_frame, f'{prob:.2f}', (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Use Haar Cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.putText(display_frame, "Press SPACE to verify, ESC to cancel", 
                       (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # Space bar
                # Check if face detected
                face_detected = False
                if USE_MTCNN and self.face_detector is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    boxes, probs = self.face_detector.detect(pil_frame)
                    face_detected = boxes is not None and len(boxes) > 0
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    face_detected = len(faces) > 0
                
                if face_detected:
                    # Extract embedding
                    test_embedding = self.extract_embedding(frame)
                    
                    if test_embedding is None:
                        messagebox.showwarning("Warning", "Face not detected clearly. Try again.")
                        continue
                    
                    # Compare with database
                    best_match = None
                    best_similarity = -1
                    
                    for name, data in self.face_database.items():
                        # Compare with all stored embeddings for this person
                        similarities = []
                        for stored_embedding in data['embeddings']:
                            similarity = F.cosine_similarity(
                                test_embedding.unsqueeze(0),
                                stored_embedding.unsqueeze(0)
                            ).item()
                            similarities.append(similarity)
                        
                        # Use maximum similarity
                        max_similarity = max(similarities)
                        
                        if max_similarity > best_similarity:
                            best_similarity = max_similarity
                            best_match = name
                    
                    # Verify
                    cv2.destroyWindow(window_name)
                    self.camera.release()
                    
                    if best_similarity >= self.similarity_threshold:
                        messagebox.showinfo("✅ Access Granted", 
                                           f"Welcome, {best_match}!\n\n"
                                           f"Confidence: {best_similarity:.2%}")
                    else:
                        messagebox.showwarning("❌ Unknown Face", 
                                              f"Face not recognized.\n\n"
                                              f"Best match: {best_match} ({best_similarity:.2%})\n"
                                              f"Threshold: {self.similarity_threshold:.2%}")
                    return
                else:
                    messagebox.showwarning("Warning", "No face detected. Please position your face in the frame.")
            
            elif key == 27:  # ESC
                break
        
        cv2.destroyWindow(window_name)
        self.camera.release()
    
    def view_database(self):
        """View all registered faces."""
        if len(self.face_database) == 0:
            messagebox.showinfo("Database", "No faces registered yet.")
            return
        
        info = "Registered Faces:\n\n"
        for name, data in self.face_database.items():
            info += f"👤 {name}\n"
            info += f"   Poses: {len(data['embeddings'])}\n"
            info += f"   Added: {data['date_added']}\n\n"
        
        messagebox.showinfo("Face Database", info)
    
    def delete_face(self):
        """Delete a face from the database with selection dialog."""
        if len(self.face_database) == 0:
            messagebox.showinfo("Database", "No faces registered yet.")
            return
        
        # Create selection window
        delete_window = tk.Toplevel(self.root)
        delete_window.title("Delete Face from Database")
        delete_window.geometry("350x400")
        delete_window.configure(bg='#2c3e50')
        
        # Title
        title = tk.Label(
            delete_window,
            text="Select Face to Delete",
            font=('Arial', 13, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title.pack(pady=10)
        
        # Frame for listbox and scrollbar
        list_frame = tk.Frame(delete_window, bg='#2c3e50')
        list_frame.pack(pady=5, padx=15, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        listbox = tk.Listbox(
            list_frame,
            font=('Arial', 11),
            bg='#ecf0f1',
            selectmode=tk.SINGLE,
            height=10,
            yscrollcommand=scrollbar.set
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        names = list(self.face_database.keys())
        for name in names:
            data = self.face_database[name]
            display_text = f"{name}  ({len(data['embeddings'])} poses, {data['date_added']})"
            listbox.insert(tk.END, display_text)
        
        # Button frame
        button_frame = tk.Frame(delete_window, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        def on_delete():
            """Handle delete button click."""
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a face to delete.")
                return
            
            selected_idx = selection[0]
            selected_name = names[selected_idx]
            
            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", 
                                  f"Are you sure you want to delete '{selected_name}'?"):
                del self.face_database[selected_name]
                self.save_database()
                self.status_label.config(text=f"Database: {len(self.face_database)} registered faces")
                messagebox.showinfo("Deleted", f"'{selected_name}' has been removed from the database.")
                delete_window.destroy()
        
        def on_cancel():
            """Handle cancel button click."""
            delete_window.destroy()
        
        # Delete button
        delete_btn = tk.Button(
            button_frame,
            text="🗑️ Delete",
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            width=15,
            command=on_delete,
            cursor='hand2'
        )
        delete_btn.grid(row=0, column=0, padx=10)
        
        # Cancel button
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            font=('Arial', 12),
            bg='#95a5a6',
            fg='white',
            width=15,
            command=on_cancel,
            cursor='hand2'
        )
        cancel_btn.grid(row=0, column=1, padx=10)
        
        # Make window modal
        delete_window.transient(self.root)
        delete_window.grab_set()
        self.root.wait_window(delete_window)
    
    def run(self):
        """Run the application."""
        self.root.mainloop()
        
        # Cleanup
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run Face ID application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face ID Application')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/local/vggface2_weak/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--database', type=str, 
                       default='face_database.json',
                       help='Path to face database file')
    parser.add_argument('--threshold', type=float, 
                       default=0.6,
                       help='Similarity threshold for verification (0-1)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first or provide correct path.")
        return
    
    # Create and run app
    app = FaceIDApp(
        model_path=args.model,
        database_path=args.database
    )
    
    # Set custom threshold if provided
    if args.threshold != 0.6:
        app.similarity_threshold = args.threshold
        print(f"Using similarity threshold: {args.threshold}")
    
    print("\n" + "="*60)
    print("🔐 Face ID Application Started")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Database: {args.database}")
    print(f"Similarity Threshold: {app.similarity_threshold}")
    print(f"Registered Faces: {len(app.face_database)}")
    print("="*60 + "\n")
    
    app.run()


if __name__ == '__main__':
    main()
