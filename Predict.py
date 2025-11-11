import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 🧠 المسارات
model_path = "/content/drive/MyDrive/animal_model.pth"
trained_folders_file = "/content/drive/MyDrive/trained_folders.txt"

# ✅ قراءة أسماء الفئات من ملف التدريب
if os.path.exists(trained_folders_file):
    with open(trained_folders_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
else:
    raise ValueError("❌ ملف trained_folders.txt غير موجود! درّب الموديل أولاً.")

num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧩 بناء الموديل
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 🔄 تحميل الوزن
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 🧾 تحويل الصورة (زي التدريب)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🖼️ إدخال مسار الصورة
image_path = input("🖼️ أدخل مسار الصورة (من Google Drive أو Colab): ").strip()

if not os.path.exists(image_path):
    print("❌ الصورة غير موجودة. تأكد من المسار الصحيح.")
else:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    print(f"✅ الحيوان المتوقع هو: {predicted_class}")
