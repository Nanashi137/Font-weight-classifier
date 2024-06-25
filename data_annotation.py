import os 

LABELS2ID = {'unbold': 0, 'bold': 1, 'italic': 2, 'bold_italic': 3}

path = "F:/FWC216/data/"

folders = os.listdir(path)

for folder in folders: 
    ds_path = path + folder 

    image_list = os.listdir(ds_path)
    for image_path in image_list: 
        sub = image_path[:-4]
        new_image_path = sub + "," + folder + ".jpg"
        src_path = path + folder + "/" + image_path
        dst_path = path + folder + "/" + new_image_path
        # print(src_path)
        # print(dst_path) 
        os.rename(src=src_path, dst=dst_path)
