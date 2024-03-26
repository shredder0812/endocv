test_vid = "/content/UTTQ/CS101.mp4"


# Lấy tên file video từ test_vid
input_video_name = test_vid.split("/")[-1].split(".")[0] + '_' + test_vid.split("/")[-2].split(".")[0]

vid_utdd_uttq = [['BVK019', 'BVK022', 'BVK024', 'BVK029', 'BVK042', 'BVK066'], ['CS101', 'CS201', 'BVK037', 'BVK040', 'BVK083', 'BVK091']]




model_weights = "/content/thucquan.pt"

# Tạo từ điển ánh xạ giữa tên model_weights và model_classes
model_classes_dict = {
    "/content/thucquan.pt": ['2_Viem_thuc_quan', '5_Ung_thu_thuc_quan'],
    "/content/daday.pt": ['3_Viem_da_day_HP_am', '4_Viem_da_day_HP_duong', '6_Ung_thu_da_day'],
    "/content/htt.pt": ['7_Loet_HTT']
}

# Thiết lập model_classes từ từ điển, nếu không khớp thì trả về ['polyp', 'esophagael cancer']
model_classes = model_classes_dict.get(model_weights, ['polyp', 'esophagael cancer'])

print("Input Video Name:", input_video_name)
print("Model Classes:", model_classes)
print(vid_utdd_uttq[1][2])