import pandas as pd
import shutil
import os
from PIL import Image

def save_df_to_storage(df, storage_folder_path= "../storage/"):
    # 将DataFrame保存为CSV文件
    df.to_csv(storage_folder_path , index=False)

    print("DataFrame数据已成功保存为CSV文件。")




def save_image_to_storage(image, file_name):
    storage_folder = "storage_pic"
    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)

    file_path = os.path.join(storage_folder, file_name)
    image.save(file_path)
    print(f"Image saved to {file_path}")

# 示例用法
# 假设 img 是您生成的图片对象
# 调用函数保存图片到 "storage" 文件夹中，文件名为 "example.jpg"
# save_image_to_storage(img, "example.jpg")

def duquyinzi(filename,iname = 'Unnamed: 0'):
    df = pd.read_csv(filename)
    df.set_index(iname,inplace = True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "TRADE_DATE"
    return df