import glob
#win下写法，linux同学自己改下分隔符
all_file_name_list = glob.glob(r'D:\DL-GuPao\YOLO7\NEU-DET\valid\images\*.*')

with open(r'D:\DL-GuPao\YOLO7\NEU-DET\val.txt','w') as f:
    for file_name in all_file_name_list:
        file_name = file_name.replace("\\", "\\\\")
        f.write(file_name)
        f.write('\n')

all_file_name_list = glob.glob(r'D:\DL-GuPao\YOLO7\NEU-DET\train\images\*.*')

with open(r'D:\DL-GuPao\YOLO7\NEU-DET\train.txt','w') as f:
    for file_name in all_file_name_list:
        file_name = file_name.replace("\\", "\\\\")
        f.write(file_name)
        f.write('\n')

