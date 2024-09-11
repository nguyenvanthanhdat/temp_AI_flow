import os
import numpy as np
import json
from PIL import Image
from transnetv2 import TransNetV2  # Giả sử bạn đã tải mô hình TransNetV2
import time  # Thêm thư viện time

# Khởi tạo mô hình TransNetV2
model = TransNetV2()

# Bắt đầu đo thời gian
start_time = time.time()

# Đường dẫn tới thư mục chứa các frames
frames_dir = "frames"

# Kiểm tra thư mục frames có tồn tại hay không
if not os.path.exists(frames_dir):
    print(f"Thư mục {frames_dir} không tồn tại.")
else:
    # Lấy danh sách các tệp frames (sắp xếp theo thứ tự)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    # Kiểm tra xem có tệp nào trong thư mục frames không
    if not frame_files:
        print("Thư mục frames không chứa bất kỳ file ảnh .jpg nào.")
    else:
        # Tạo một danh sách để lưu các frames sau khi resize
        frames = []

        # Đọc từng frame và resize về 48x27
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            # print(f"Đang xử lý: {frame_path}")  # In ra tên file đang xử lý

            # Mở và resize ảnh
            img = Image.open(frame_path)
            img_resized = img.resize((48, 27))  # Resize về kích thước 48x27
            frame_array = np.array(img_resized)  # Chuyển hình ảnh sang numpy array
            frames.append(frame_array)

        # Chuyển frames thành numpy array 4 chiều (4D array) để đưa vào mô hình
        frames_np = np.array(frames)  # Kích thước sẽ là (số lượng frames, 27, 48, 3)
        # print(f"Frames array shape: {frames_np.shape}")

        # Dự đoán với mô hình TransNetV2
        single_frame_pred, all_frame_pred = model.predict_frames(frames_np)

        # Kết quả dự đoán
        # print(single_frame_pred)
        # print(all_frame_pred)

        # Bạn có thể sử dụng single_frame_pred và all_frame_pred để phân đoạn video thành các cảnh
        scenes = model.predictions_to_scenes(single_frame_pred)

        # In ra các cảnh dự đoán
        # print("Các cảnh dự đoán:", scenes)

        # Chuyển đổi numpy array scenes thành danh sách Python
        scenes_list = [scene.tolist() for scene in scenes]

        # Lưu các cảnh vào file JSON
        output_file = "scenes.json"
        with open(output_file, 'w') as f:
            json.dump(scenes_list, f)
        # print(f"Lưu thông tin cảnh vào file {output_file} thành công.")

# Kết thúc đo thời gian
end_time = time.time()

# Tính tổng thời gian đã thực hiện
elapsed_time = end_time - start_time
print(f"Thời gian thực hiện: {elapsed_time:.2f} giây")


# import os
# import numpy as np
# from PIL import Image
# from transnetv2 import TransNetV2  # Giả sử bạn đã tải mô hình TransNetV2

# # Khởi tạo mô hình TransNetV2
# model = TransNetV2()

# # Đường dẫn tới thư mục chứa các frames
# frames_dir = "frames"

# # Kiểm tra thư mục frames có tồn tại hay không
# if not os.path.exists(frames_dir):
#     print(f"Thư mục {frames_dir} không tồn tại.")
# else:
#     # Lấy danh sách các tệp frames (sắp xếp theo thứ tự)
#     frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

#     # Kiểm tra xem có tệp nào trong thư mục frames không
#     if not frame_files:
#         print("Thư mục frames không chứa bất kỳ file ảnh .jpg nào.")
#     else:
#         # Tạo một danh sách để lưu các frames sau khi resize
#         frames = []

#         # Đọc từng frame và resize về 48x27
#         for frame_file in frame_files:
#             frame_path = os.path.join(frames_dir, frame_file)
#             print(f"Đang xử lý: {frame_path}")  # In ra tên file đang xử lý

#             # Mở và resize ảnh
#             img = Image.open(frame_path)
#             img_resized = img.resize((48, 27))  # Resize về kích thước 48x27
#             frame_array = np.array(img_resized)  # Chuyển hình ảnh sang numpy array
#             frames.append(frame_array)

#         # Chuyển frames thành numpy array 4 chiều (4D array) để đưa vào mô hình
#         frames_np = np.array(frames)  # Kích thước sẽ là (số lượng frames, 27, 48, 3)
#         print(f"Frames array shape: {frames_np.shape}")

#         # Dự đoán với mô hình TransNetV2
#         single_frame_pred, all_frame_pred = model.predict_frames(frames_np)

#         # Kết quả dự đoán
#         print(single_frame_pred)
#         print(all_frame_pred)

#         # Điều chỉnh ngưỡng để xác định độ nhạy của phân đoạn cảnh
#         threshold = 0.005  # Thay đổi giá trị ngưỡng để tăng/giảm độ nhạy

#         # Lọc các cảnh dựa trên ngưỡng đã chỉ định
#         scenes = model.predictions_to_scenes(single_frame_pred, threshold=threshold)

#         # In ra số lượng cảnh trước khi bổ sung
#         print(f"Số lượng cảnh trước khi bổ sung: {len(scenes)}")
#         print(f"Các cảnh trước khi bổ sung: {scenes}")

#         # Bổ sung các khoảng thiếu vào giữa các cảnh
#         full_scenes = []
#         previous_end = 0

#         for start, end in scenes:
#             # Nếu có khoảng thiếu giữa cảnh trước và cảnh hiện tại
#             if start > previous_end + 1:
#                 missing_scene = (previous_end + 1, start - 1)
#                 full_scenes.append(missing_scene)
#                 print(f"Khoảng thiếu được thêm vào là cảnh: {missing_scene}")

#             # Thêm cảnh hiện tại
#             full_scenes.append((start, end))
#             previous_end = end

#         # Nếu có các frame cuối cùng chưa được đánh dấu là cảnh
#         if previous_end < len(frame_files) - 1:
#             full_scenes.append((previous_end + 1, len(frame_files) - 1))
#             print(f"Các frame cuối cùng được thêm vào là cảnh: {(previous_end + 1, len(frame_files) - 1)}")

#         # In ra số lượng cảnh sau khi bổ sung
#         print(f"Số lượng cảnh sau khi bổ sung: {len(full_scenes)}")
#         print(f"Các cảnh sau khi bổ sung: {full_scenes}")

