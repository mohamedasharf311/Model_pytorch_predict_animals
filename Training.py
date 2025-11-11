# -------------------------------------------
# Incremental training for animal folders (Colab + Drive)
# -------------------------------------------

import os, shutil, errno, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ----------------- الإعداد -----------------
drive_dir = "/content/drive/MyDrive"
data_dir = f"{drive_dir}/animals"       # <-- عدّل لو اسم الفولدر مختلف
model_path = f"{drive_dir}/animal_model.pth"
trained_folders_file = f"{drive_dir}/trained_folders.txt"

temp_dir = "/content/temp_animals"      # مجلد مؤقت في بيئة Colab
batch_size = 32
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# تحويلات الصور
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ----------------- جلب أسماء الفولدرات -----------------
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data dir not found: {data_dir}\nتأكد أن المسار صحيح داخل Drive")

all_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
print("Found folders in Drive:", len(all_folders))

# اقرأ الفولدرات المتدربة سابقًا إن وجدت
if os.path.exists(trained_folders_file):
    with open(trained_folders_file, "r") as f:
        trained_folders = [x for x in f.read().splitlines() if x.strip()]
else:
    trained_folders = []

remaining_folders = [f for f in all_folders if f not in trained_folders]
next_folders = remaining_folders[:10]   # ناخد 10 كل مرة

if not next_folders:
    print("✅ كل الفولدرات تم التدريب عليها بالفعل.")
else:
    print(f"📂 سيتم التدريب على الفولدرات التالية: {next_folders}")

    # -------------------------------------
    # نسخ آمن من Drive -> temp_dir (تجاوز الملفات المكسورة)
    # -------------------------------------
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    for folder in next_folders:
        src_folder = os.path.join(data_dir, folder)
        dst_folder = os.path.join(temp_dir, folder)
        os.makedirs(dst_folder, exist_ok=True)

        for root, _, files in os.walk(src_folder):
            rel_path = os.path.relpath(root, src_folder)
            dst_path = os.path.join(dst_folder, rel_path) if rel_path != "." else dst_folder
            os.makedirs(dst_path, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                try:
                    # نسخة آمنة: copy2 للحفاظ على metadata
                    shutil.copy2(src_file, dst_file)
                except (OSError, IOError) as e:
                    # تجاهل مشاكل الاتصال أو الملفات المفقودة أو ملفات تالفة
                    err_no = getattr(e, "errno", None)
                    print(f"⚠️ تخطي ملف تالف أو غير متاح: {src_file}  ({e})")
                    continue

    # -------------------------------------
    # بناء قائمة الكلاسات التراكمية (القديمة + الجداد)
    # -------------------------------------
    all_trained = trained_folders + next_folders
    print("Total classes after this run:", len(all_trained))

    # -------------------------------------
    # نحمل الداتا من temp_dir (اللي فيه فقط الـ next_folders)
    # ثم نُعيد ترميز الـ labels لتتوافق مع الـ all_trained indices
    # -------------------------------------
    temp_dataset = datasets.ImageFolder(temp_dir, transform=transform)
    # temp_dataset.classes == قائمة الفولدرات في temp_dir (مرتبة أبجديًا)
    # نخلق mapping من اسم الكلاس -> global_index
    global_class_to_idx = {name: idx for idx, name in enumerate(all_trained)}

    # Wrapper dataset ليعيد تعيين العلامات للـ global indices
    class RemappedDataset(Dataset):
        def __init__(self, imagefolder_dataset, global_map):
            self.samples = imagefolder_dataset.samples   # list of (path, local_label)
            self.transform = imagefolder_dataset.transform
            self.local_classes = imagefolder_dataset.classes
            self.global_map = global_map

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, local_label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            local_class_name = self.local_classes[local_label]
            global_label = self.global_map[local_class_name]
            return img, global_label

    train_dataset = RemappedDataset(temp_dataset, global_class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # -------------------------------------
    # نموذج ResNet18 مع توسيع طبقة الـ fc لتناسب عدد all_trained
    # -------------------------------------
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    new_num_classes = len(all_trained)
    model.fc = nn.Linear(num_features, new_num_classes)
    model = model.to(device)

    # -------------------------------------
    # لو في موديل قديم: نحاول نحمله وننقل أوزان الـ fc القديمة
    # -------------------------------------
    if os.path.exists(model_path):
        print("📦 تم العثور على موديل سابق. جارٍ التحميل ومحاولة الاحتفاظ بالأوزان القديمة...")
        old_state = torch.load(model_path, map_location=device)
        # نحاول استخراج أوزان الـ fc القديمة إن وجدت
        old_fc_w = old_state.get('fc.weight', None)
        old_fc_b = old_state.get('fc.bias', None)

        # نحمل باقي الstate_dict (strict=False علشان حجم fc يمكن يختلف)
        try:
            model.load_state_dict(old_state, strict=False)
            print("✅ تم تحميل معظم الأوزان بنجاح (باستثناء اختلافات طبقة fc إن وُجدت).")
        except Exception as e:
            print("⚠️ حصل خطأ أثناء تحميل الstate dict بشكل غير صارم:", e)

        # إذا كان هناك أوزان قديمة للـ fc، ننقلها للصفوف الأولى من fc الجديدة
        if old_fc_w is not None and old_fc_b is not None:
            try:
                old_num = old_fc_w.shape[0]
                copy_num = min(old_num, new_num_classes)
                with torch.no_grad():
                    model.fc.weight.data[:copy_num].copy_(old_fc_w[:copy_num])
                    model.fc.bias.data[:copy_num].copy_(old_fc_b[:copy_num])
                print(f"🔁 تم نقل أوزان الـ fc القديمة للأول {copy_num} كلاسات.")
            except Exception as e:
                print("⚠️ فشل نقل أوزان الـ fc القديمة:", e)

    # -------------------------------------
    # إعداد التدريب
    # -------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------------------
    # تدريب
    # -------------------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / max(1, total_batches)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")

    # -------------------------------------
    # حفظ الموديل والـ trained_folders_file
    # -------------------------------------
    torch.save(model.state_dict(), model_path)
    print("💾 تم حفظ الموديل في:", model_path)

    # نحدّث trained_folders_file (نتأكد من أنه ترتيب ثابت)
    updated_trained = trained_folders + next_folders
    with open(trained_folders_file, "w") as f:
        f.write("\n".join(updated_trained))
    print("✅ تم تحديث ملف trained_folders.txt")

    # تنظيف temp_dir لو حبيت (اختياري)
    # shutil.rmtree(temp_dir)
    print("تم الانتهاء من التدريب على المجموعة الحالية.")
