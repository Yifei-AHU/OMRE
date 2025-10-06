import json
# 读取JSON文件
import os
import shutil


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
json_data=read_json_file('test.json')
id_count = {}
for item in json_data:
        id_value = item['id']
        caption = item['caption']
        img_path_list = item['predict_img_path']

        if id_value not in id_count:
            id_count[id_value] = 1
        else:
            id_count[id_value] += 1

        # 根据id和计数来命名文件夹
        folder_name ="rstp_test/"+str(id_value)
        if id_count[id_value]%2!=0:
            folder_name ="rstp_test/"+str(id_value) + '_' + str(id_count[id_value]//2+1)+ '_1'
        else:
            folder_name ="rstp_test/"+str(id_value) + '_' + str(id_count[id_value]//2)+ '_2'

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 创建并写入caption到txt文件
        with open(os.path.join(folder_name, folder_name.split("test/")[-1] + '.txt'), 'w') as txt_file:
            txt_file.write(str(caption))

        # 复制图片到文件夹
        for img_path in img_path_list:
            img_name = os.path.basename(img_path.split("/")[-1])
            shutil.copy(img_path.replace("/data/dengyifei/Data/UPP_Pre_RSTP/","./"), os.path.join(folder_name, img_name))