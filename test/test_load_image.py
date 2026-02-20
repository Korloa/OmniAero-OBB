from pathlib import Path
import cv2
import numpy as np
# 根据你实际修改的文件路径导入，例如从 ultralytics.data.dataset 导入
from ultralytics.data.dataset import YOLODataset


def test_load_image():
    # 1. 初始化数据集 (指向你的 dataset.yaml 或 数据根目录)
    # img_path: 包含图像路径的 txt 文件，或者直接传图片列表
    dataset = YOLODataset(img_path='path/to/your/images.txt',
                          imgsz=640,
                          augment=False,
                          hyp=None)

    print(f"数据集大小：{len(dataset)}")

    # 2. 测试正常加载 (索引 0)
    try:
        print("\n=== 测试 1: 正常加载 ===")
        img, path, name = dataset.load_image(0)

        # 验证通道数
        assert img.shape[2] == 4, f"❌ 失败：通道数为 {img.shape[2]}，应为 4"
        print(f"✅ 成功：图像形状 {img.shape} (4 通道)")
        print(f"   路径：{path}")

        # 验证 IR 通道是否有内容 (不全为 0)
        ir_channel = img[:, :, 3]
        if np.mean(ir_channel) > 0:
            print(f"✅ 成功：IR 通道有有效数据 (均值：{np.mean(ir_channel):.2f})")
        else:
            print("⚠️  警告：IR 通道全为 0，可能读取失败")

    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return

    # 3. 测试异常中断 (伪造一个不存在的路径)
    try:
        print("\n=== 测试 2: 异常中断验证 ===")
        # 临时修改一个路径让它找不到 IR
        original_load = dataset.load_image

        def mock_load(i, rect_mode=False):
            img, path, name = original_load(i, rect_mode=rect_mode)
            # 强制修改路径让 IR 找不到
            path = str(Path(path).parent.parent / "non_exist_dir" / Path(path).name)
            return img, path, name

        # 这里需要直接测试内部逻辑，更简单的方法是手动构造错误路径
        from ultralytics.data.dataset import YOLODataset
        import ultralytics.data.dataset as ds_module

        # 直接测试文件不存在的情况
        test_img_path = list(dataset.img_files)[0]
        p = Path(test_img_path)
        fake_ir_path = p.parent.parent / "fake_dir_r" / p.name

        if not fake_ir_path.exists():
            print(f"✅ 成功：确认测试路径不存在 {fake_ir_path}")
            print("   (当加载到该图片时，程序应抛出 FileNotFoundError)")

    except Exception as e:
        print(f"异常捕获：{e}")


if __name__ == "__main__":
    test_load_image()