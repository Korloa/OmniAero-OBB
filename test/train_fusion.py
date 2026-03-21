import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    # ==========================================
    # 0. 解析命令行参数
    # ==========================================
    parser = argparse.ArgumentParser(description="YOLO Fusion Model Training Script")

    # 定义命令行参数
    parser.add_argument(
        '--yaml',
        type=str,
        default="ultralytics/cfg/models/26/yolo26-v3-v1.yaml",
        help='指定模型结构的YAML文件路径'
    )

    parser.add_argument(
        '--name',
        type=str,
        default="OmniAero_Fusion_v3_V1",
        help='指定本次训练的保存名称'
    )

    # 设置 data_type 的选项
    parser.add_argument(
        '--data_type',
        type=str,
        default="win",
        choices=['win', 'dsw', 'station'],
        help='选择数据集环境:'
    )

    args = parser.parse_args()

    # 根据传入的 data_type 映射真实的 yaml 路径
    dataset_map = {
        "win": "F:/work/OmniAero-OBB/test/dataset.yaml",
        "dsw": "/mnt/workspace/OmniAero-OBB/test/dataset.yaml",
        "station": "/2024011184/zhangkaixiang/OmniAero-OBB/test/dataset.yaml"
    }

    selected_data_path = dataset_map[args.data_type]

    print(f"[-] 模型结构路径: {args.yaml}")
    print(f"[-] 训练保存名称: {args.name}")
    print(f"[-] 当前选择环境: {args.data_type} -> 路径: {selected_data_path}")
    print("==========================================")

    # ==========================================
    # 1. 加载模型结构 (使用命令行参数 args.yaml)
    # ==========================================
    model = YOLO(args.yaml)

    # ==========================================
    # 2. 加载预训练权重 (迁移学习)
    # ==========================================
    # YOLO 会自动跳过形状不匹配的第一层，加载后面匹配的层
    try:
        model.load("yolo26n-obb.pt")
        print("预训练权重加载成功 (部分层)")
    except Exception as e:
        print(f"权重加载提示: {e}")

    # ==========================================
    # 3. 开始训练
    # ==========================================
    model.train(
        data=selected_data_path,  # 【修改】使用映射后得到的数据集路径
        imgsz=800,  # 【提升】从 640 提升到 800，增强小目标识别
        epochs=150,  # 【增加】给大数据集更多学习时间
        batch=56,  # 【提升】20G 显存建议从 64 起步试试
        workers=8,  # 【提升】加快数据加载
        device=0,
        amp=True,
        patience=50,  # 【新增】50次迭代没提升再停止
        # === 关键参数 ===
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=1.0,
        mixup=0.1,  # 【新增】增加 mixup 增强，防止过拟合
        name=args.name  # 【修改】使用命令行指定的训练名称
    )