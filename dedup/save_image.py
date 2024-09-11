import os
from datasets import load_dataset
from PIL import Image

# Tạo thư mục 'images' nếu chưa tồn tại
os.makedirs('frames', exist_ok=True)

# Tải dataset từ Hugging Face
dataset = load_dataset('wanhin/aic2')

# Duyệt qua từng mẫu trong dataset và lưu ảnh
for i, data in enumerate(dataset['train']):
    image = data['image']  # Đây có thể là đối tượng hình ảnh
    desc = data['desc']    # Tên mô tả dùng để đặt tên file
    
    # Đảm bảo tên file có đúng định dạng .jpg mà không dư chữ 'jpg'
    if not desc.lower().endswith('.jpg'):
        desc += '.jpg'
    
    # Đường dẫn lưu ảnh
    output_path = os.path.join('images', desc)
    
    # Nếu 'image' là đối tượng hình ảnh, lưu lại bằng Pillow
    if isinstance(image, Image.Image):
        image.save(output_path)  # Lưu đối tượng hình ảnh
    else:
        # Nếu 'image' là đường dẫn, copy file
        shutil.copy(image, output_path)
