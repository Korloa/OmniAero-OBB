
# Zhang written
from ultralytics.data.dataset import YOLODataset

data_info = {
    "nc": 5,
    "names":{
        0:'bus',
        1:'car',
        2:'feright_car',
        3:'truck',
        4:'van'
    },
    "channels":4
}

dataset = YOLODataset(
    img_path="src/after/train/images/train",
    data=data_info,
    imgsz=640,
    augment=False,
    task="obb",
    batch_size=32,
)

# 3. 测试一下能不能读到数据
print(f"检测到 OBB 模式: {dataset.use_obb}")
print(f"通道数: {dataset.data['channels']}")

# 尝试读第一条数据
sample = dataset[0]
print(sample.keys()) # 看看有没有 'img' (应该是 4 通道) 和 'obb' 标签
print("标签数据示例:", sample['bboxes'])
print(sample['img'].shape)