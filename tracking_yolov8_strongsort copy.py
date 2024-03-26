from ultralytics import YOLO
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from boxmot import (OCSORT, BoTSORT, BYTETracker, DeepOCSORT, StrongSORT, create_tracker, get_tracker_config)
from pathlib import Path
# import sys
# import datetime
# import pandas as pd
# from google.colab.patches import cv2_imshow

test_vid = "D:/DANC/testtttt/drama.mp4"
model_weights = "D:/DANC/testtttt/model_yolo/yolov8x.pt"

model_name_dict = {
    "D:/DANC/testtttt/model_yolo/thucquan.pt": '_TQ',
    "D:/DANC/testtttt/model_yolo/daday.pt": '_DD',
    "D:/DANC/testtttt/model_yolo/htt.pt": '_HTT',
    "D:/DANC/testtttt/model_yolo/yolov8x.pt": '_Y8x'
}
# Lấy tên file video từ test_vid
#input_video_name = test_vid.split("/")[-1].split(".")[0] + model_name_dict.get(model_weights, ['polyp', 'esophagael cancer'])
input_video_name = 'CS201'

# Tạo từ điển ánh xạ giữa tên model_weights và model_classes
model_classes_dict = {
    "D:/DANC/testtttt/model_yolo/thucquan.pt": ['2_Viem_thuc_quan', '5_Ung_thu_thuc_quan'],
    "D:/DANC/testtttt/model_yolo/daday.pt": ['3_Viem_da_day_HP_am', '4_Viem_da_day_HP_duong', '6_Ung_thu_da_day'],
    "D:/DANC/testtttt/model_yolo/htt.pt": ['7_Loet_HTT'],
    "D:/DANC/testtttt/model_yolo/yolov8x.pt": ['person',	 'bicycle',	 'car',	 'motorcycle',	 'airplane',	 'bus',	 'train',	 'truck',	 'boat',	 'traffic light',	 'fire hydrant',	 'stop sign',	 'parking meter',	 'bench',	 'bird',	 'cat',	 'dog',	 'horse',	 'sheep',	 'cow',	 'elephant',	 'bear',	 'zebra',	 'giraffe',	 'backpack',	 'umbrella',	 'handbag',	 'tie',	 'suitcase',	 'frisbee',	 'skis',	 'snowboard',	 'sports ball',	 'kite',	 'baseball bat',	 'baseball glove',	 'skateboard',	 'surfboard',	 'tennis racket',	 'bottle',	 'wine glass',	 'cup',	 'fork',	 'knife',	 'spoon',	 'bowl',	 'banana',	 'apple',	 'sandwich',	 'orange',	 'broccoli',	 'carrot',	 'hot dog',	 'pizza',	 'donut',	 'cake',	 'chair',	 'couch',	 'potted plant',	 'bed',	 'dining table',	 'toilet',	 'tv', 'laptop',  'mouse',  'remote',  'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush' ]
}

# Thiết lập model_classes từ từ điển, nếu không khớp thì trả về ['polyp', 'esophagael cancer']
model_classes = model_classes_dict.get(model_weights, ['polyp', 'esophagael cancer'])

print("Input Video Name:", input_video_name)
print("Model Classes:", model_classes)


class Colors:
    def __init__(self, num_colors=80):
        self.num_colors = num_colors
        self.color_palette = self.generate_color_palette()


    def generate_color_palette(self):
        hsv_palette = np.zeros((self.num_colors, 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = np.linspace(0, 180, self.num_colors, endpoint=False)
        hsv_palette[:, :, 1:] = 255
        bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)
        return bgr_palette.reshape(-1, 3)

    def __call__(self, class_id):
        color = tuple(map(int, self.color_palette[class_id]))
        return color


class ObjectDetection:
    def __init__(self, model_weights="yolov8s.pt", capture_index=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model_weights)
        self.classes = self.model.names
        #self.classes[0] = 'gastritis'
        self.classes = model_classes
        self.colors = Colors(len(self.classes))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.capture_index = capture_index
        self.cap = self.load_capture()
        reid_weights = Path("osnet_x0_25_msmt17.pt")
        self.tracker = StrongSORT(reid_weights,
                                  torch.device(self.device),
                                  fp16 = False,
                                  )

    def load_model(self, weights):
        model = YOLO(weights)
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame, stream=True, verbose=False, conf=0.65, line_width=1)
        return results

    def draw_tracks(self, frame, tracks, txt_file, overlap_threshold=0.5):
      # Tạo một danh sách để lưu trữ các bounding box duy nhất sau khi loại bỏ trùng lặp
      unique_boxes = []

      for track in tracks:
          x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
          id = int(track[4])
          conf = round(track[5], 2)
          class_id = int(track[6])

          # Tính diện tích của bounding box hiện tại
          area_current = (x2 - x1) * (y2 - y1)

          # Kiểm tra các bounding box đã tồn tại trong danh sách unique_boxes
          is_unique = True
          for box in unique_boxes:
              x1_prev, y1_prev, x2_prev, y2_prev = box
              # Tính diện tích chồng lấp giữa bounding box hiện tại và bounding box trong danh sách
              x_overlap = max(0, min(x2, x2_prev) - max(x1, x1_prev))
              y_overlap = max(0, min(y2, y2_prev) - max(y1, y1_prev))
              area_overlap = x_overlap * y_overlap
              area_union = area_current + (x2_prev - x1_prev) * (y2_prev - y1_prev) - area_overlap
              overlap_ratio = area_overlap / area_union

              # Nếu diện tích chồng lấp vượt qua ngưỡng trùng lặp, đánh dấu bounding box hiện tại không duy nhất
              if overlap_ratio > overlap_threshold:
                  is_unique = False
                  break

          # Nếu bounding box là duy nhất, thêm vào danh sách unique_boxes và vẽ lên frame
          if is_unique:
              unique_boxes.append((x1, y1, x2, y2))
              class_name = self.classes[class_id]
              cv2.rectangle(frame, (x1,y1), (x2, y2), self.colors(class_id), 5)
              label = f'{class_name} {conf}' # hiển thị
              (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
              cv2.rectangle(frame, (x1, y1-h-15), (x1+w, y1), self.colors(class_id), -1)
              cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255) , 5)
              # Ghi kết quả vào file txt
              txt_file.write(f"{int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))},{id},{x1},{y1},{x2-x1},{y2-y1},{conf},-1,-1,-1\n")

      return frame


    def load_capture(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        video_name = "strongsort_" + input_video_name + ".mp4"
        self.writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        return cap

    def write_seqinfo_ini(self, seq_name, seq_length, frame_rate, im_width, im_height, im_ext, im_dir):
        with open("seqinfo.ini", "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={seq_name}\n")
            f.write(f"imDir={im_dir}\n")  # Thay thế bằng thư mục chứa ảnh nếu cần
            f.write(f"frameRate={frame_rate}\n")
            f.write(f"seqLength={seq_length}\n")
            f.write(f"imWidth={im_width}\n")
            f.write(f"imHeight={im_height}\n")
            f.write(f"imExt={im_ext}\n")


    def __call__(self):
        tracker = self.tracker


        # Lấy thông tin từ video kết quả
        seq_name = "StrongSort"
        im_dir="img1"
        seq_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        im_ext = ".jpg"  # Phần mở rộng của ảnh

        # Ghi thông tin vào file seqinfo.ini
        self.write_seqinfo_ini(seq_name, seq_length, frame_rate, im_width, im_height, im_ext, im_dir)

        # Mở file txt để ghi kết quả
        with open("results.txt", "w") as txt_file:
          while True:
              start_time = perf_counter()
              ret, frame = self.cap.read()
              if not ret:
                  break
              cv2.rectangle(frame, (0,30), (220,80), (255,255,255),-1 )
              detections = self.predict(frame)
              for dets in detections:
                  tracks = tracker.update(dets.boxes.data.to("cpu").numpy(), frame)
                  if len(tracks.shape) == 2 and tracks.shape[1] == 8:
                      frame = self.draw_tracks(frame, tracks, txt_file)
                  #print(tracks)
              end_time = perf_counter()
              fps = 1/np.round(end_time - start_time, 2)
              cv2.putText(frame, f'FPS: {int(fps)}', (20,70), self.font, 1.5, (0,255,0), 5)
              self.writer.write(frame)
              cv2.imshow('frame', frame)
              if cv2.waitKey(5) & 0xFF == 27:
                  break
          self.cap.release()
          self.writer.release()
          cv2.destroyAllWindows()


detector = ObjectDetection(model_weights, test_vid)
#detector = ObjectDetection(model_weights, capture_index=0)
detector()

