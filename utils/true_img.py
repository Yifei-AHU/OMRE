import json
# 读取JSON文件
import os
import shutil
import cv2


def add_green_box(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(image_path)
    # 获取图片的高度、宽度和通道数
    height, width, _ = image.shape
    # 定义绿框的颜色（BGR格式，绿色为(0, 255, 0)）
    color = (0, 255, 0)
    # 定义绿框的线宽
    thickness = 2
    # 在图片边缘绘制绿框
    result = cv2.rectangle(image, (0, 0), (width - 1, height - 1), color, thickness)
    return result

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
json_data=read_json_file('/data/dengyifei/Data/RSTPReid/data_captions.json')
test=[]
for data in json_data:
    if data["split"]=='test':
        test.append(data)

id_count = {}
for item in test:
        id_value = item['id']
        img_path = item['img_path']

        if id_value not in id_count:
            id_count[id_value] = 1
        else:
            id_count[id_value] += 1

        # 根据id和计数来命名文件夹
        folder_name = "rstp_test/" + str(id_value) + '_' + str(id_count[id_value]) + '_1'
        # 复制图片到文件夹
        img_name = os.path.basename((img_path.split("/")[-1]).split(".")[0]+"true"+".jpg")
        input_path = os.path.join("/data/dengyifei/Data/RSTPReid/imgs/",
                                  img_path.replace("/data/dengyifei/Data/RSTPReid/",
                                                   "./"))
        output_path = os.path.join(folder_name, img_name)
        # 标绿框
        image_with_box = add_green_box(input_path)
        cv2.imwrite(output_path, image_with_box)

        folder_name = "rstp_test/" + str(id_value) + '_' + str(id_count[id_value]) + '_2'
        # 复制图片到文件夹
        img_name = os.path.basename((img_path.split("/")[-1]).split(".")[0]+"true"+".jpg" )
        input_path = os.path.join("/data/dengyifei/Data/RSTPReid/imgs/",
                                  img_path.replace("/data/dengyifei/Data/RSTPReid/",
                                                   "./"))
        output_path = os.path.join(folder_name, img_name)
        # 标绿框
        image_with_box = add_green_box(input_path)
        cv2.imwrite(output_path, image_with_box)


