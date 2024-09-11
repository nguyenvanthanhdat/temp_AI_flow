import os
import shutil
import json
from imagededup.methods import PHash
import time  # Thêm thư viện để đo thời gian

# Đo thời gian bắt đầu
start_time = time.time()

# Đọc file scenes.json
with open('scenes.json', 'r') as f:
    scenes = json.load(f)

# Tạo đối tượng PHash
phasher = PHash()

# Thư mục chứa các khung hình ban đầu
image_dir = 'frames'

# Thư mục lưu trữ từng cảnh
output_dir = 'scenes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Thư mục cuối cùng để lưu các ảnh gốc của từng cụm trong tất cả các cảnh
final_output_dir = 'frame_final'
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

# Đọc toàn bộ ảnh trong thư mục frames và lưu theo index
image_files = sorted(os.listdir(image_dir))  # Đọc và sắp xếp danh sách file ảnh
image_index_map = {idx: file for idx, file in enumerate(image_files)}

# Kết quả cuối cùng
scene_clusters_data = []
num_representative_images = 0  # Đếm số lượng ảnh gốc đã lưu

# Xử lý từng cảnh
for scene_idx, (start_frame, end_frame) in enumerate(scenes, 1):
    # Tạo thư mục riêng cho cảnh
    scene_folder = f'scene{scene_idx}'
    scene_folder_path = os.path.join(output_dir, scene_folder)
    if not os.path.exists(scene_folder_path):
        os.makedirs(scene_folder_path)

    # Lấy danh sách các file ảnh trong khoảng khung hình của cảnh dựa trên index từ scenes.json
    scene_images = [image_index_map[i] for i in range(start_frame, end_frame + 1) if i in image_index_map]

    # Sao chép các file ảnh của cảnh vào thư mục scene tương ứng
    for image in scene_images:
        src_image_path = os.path.join(image_dir, image)
        dst_image_path = os.path.join(scene_folder_path, image)
        if not os.path.exists(dst_image_path):
            shutil.copy(src_image_path, dst_image_path)  # Copy ảnh vào thư mục của cảnh

    # Áp dụng PHash cho các ảnh trong thư mục cảnh
    if scene_images:
        # Mã hóa các ảnh của cảnh
        encodings = phasher.encode_images(image_dir=scene_folder_path)

        # Tìm các ảnh trùng lặp trong cảnh
        duplicates = phasher.find_duplicates(encoding_map=encodings, scores=False)

        # Tạo các cụm duy nhất với 1 ảnh gốc đại diện cho mỗi cụm
        unique_clusters = {}
        visited = set()

        # Lặp qua từng ảnh gốc và các ảnh trùng lặp với nó
        for image, dup_list in duplicates.items():
            if image not in visited:
                unique_clusters[image] = dup_list
                visited.add(image)
                visited.update(dup_list)

        # Lưu thông tin cho cảnh hiện tại
        scene_info = {
            'scene_index': scene_idx,
            'num_clusters': len(unique_clusters),
            'clusters': []
        }

        # Xử lý các cụm và lưu ảnh gốc vào thư mục scene và frame_qh
        for i, (image, dup_list) in enumerate(unique_clusters.items(), 1):
            # Di chuyển hoặc sao chép ảnh gốc vào thư mục frame_qh
            src_image_path = os.path.join(scene_folder_path, image)
            dst_image_path = os.path.join(final_output_dir, image)
            if not os.path.exists(dst_image_path):
                shutil.copy(src_image_path, dst_image_path)  # Copy ảnh gốc vào frame_qh
                num_representative_images += 1  # Tăng số lượng ảnh gốc đã lưu

            # Thêm thông tin cụm vào scene_info
            scene_info['clusters'].append({
                'cluster_index': i,
                'representative_image': image,
                'num_duplicates': len(dup_list),
                'duplicate_images': dup_list
            })

        # Thêm thông tin cảnh vào kết quả cuối
        scene_clusters_data.append(scene_info)

# Lưu kết quả vào file JSON
output_json_file = 'scene_clusters.json'
with open(output_json_file, 'w') as f:
    json.dump(scene_clusters_data, f, indent=4)

# Đo thời gian kết thúc
end_time = time.time()

# Tính tổng thời gian đã thực hiện
elapsed_time = end_time - start_time

# In thông tin cuối cùng
print(f"Xử lý hoàn tất! Số lượng ảnh gốc đã lưu: {num_representative_images}")
print(f"Thời gian thực hiện: {elapsed_time:.2f} giây")
