import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Load the trained model
model_path = r"C:\Users\henri\OneDrive\Área de Trabalho\Senior Project\efficientnet_finetuned.pth"
model = EfficientNet.from_pretrained('efficientnet-b0')
num_features = model._fc.in_features
model._fc = torch.nn.Linear(num_features, 2)  # Binary classification
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225]),
])


class_labels = ["AI-GENERATED", "REAL"]

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

#  image upload
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.heic")])
    if file_path:
        result = predict_image(file_path)

        # Load and display the image
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Display the result
        result_label.config(
            text=f"This image is: {result}",
            fg="green" if result == "REAL" else "red",
        )

# Tkinter app
app = tk.Tk()
app.title("AI Image Classifier")
app.geometry("600x700")
app.configure(bg="#1a1a40")  # background color

# Upload button
upload_logo = ImageTk.PhotoImage(file=r"C:\Users\henri\OneDrive\Área de Trabalho\Senior Project\image_logos\upload button.png") 
upload_button = tk.Button(
    app, image=upload_logo, command=upload_image,
    bg="#1a1a40", activebackground="#1a1a40", borderwidth=0
)
upload_button.pack(pady=30)


image_label = tk.Label(app, bg="#1a1a40")
image_label.pack(pady=20)


result_label = tk.Label(
    app, text="", font=("Arial", 20, "bold"),
    bg="#1a1a40", fg="white"
)
result_label.pack(pady=20)

# Run the app
app.mainloop()

