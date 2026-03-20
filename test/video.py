import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm  # 用于显示进度条


def letterbox_4c(img, new_shape=(800, 800), color=(114, 114, 114, 0)):
    """
    自定义 4通道 LetterBox 缩放
    保证图像等比例缩放到 800x800，多余部分用纯色填充
    """
    shape = img.shape[:2]  # current shape [height, width]

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 计算需要填充的边框大小 (Padding)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # 两边各一半
    dh /= 2

    # 缩放图像
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 添加边框 (Padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (left, top)


def run_fusion_video_inference(rgb_dir, ir_dir, weights_path, output_video_path, fps=25):
    # 1. 加载训练好的模型
    print("正在加载融合模型...")
    model = YOLO(weights_path)
    names = model.names
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]  # 为不同类别分配颜色

    # 2. 获取并排序所有图片文件名 (确保 000001, 000002 顺序正确)
    valid_extensions = ('.jpg', '.png', '.jpeg')
    image_names = [f for f in os.listdir(rgb_dir) if f.lower().endswith(valid_extensions)]
    image_names.sort()  # 按文件名升序排列

    if not image_names:
        raise ValueError(f"在 {rgb_dir} 中没有找到图片文件！")

    print(f"共找到 {len(image_names)} 帧图像，准备开始推理...")

    # 3. 读取第一帧以获取视频分辨率，并初始化 VideoWriter
    first_rgb_path = os.path.join(rgb_dir, image_names[0])
    first_img = cv2.imread(first_rgb_path)
    if first_img is None:
        raise ValueError(f"无法读取第一张图片: {first_rgb_path}")

    height, width = first_img.shape[:2]
    # 使用 mp4v 编码器保存为 mp4 格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 4. 逐帧循环处理
    for img_name in tqdm(image_names, desc="处理进度"):
        rgb_path = os.path.join(rgb_dir, img_name)
        ir_path = os.path.join(ir_dir, img_name)  # 假设红外图和RGB图同名

        if not os.path.exists(ir_path):
            print(f"警告: 找不到对应的红外图像 {ir_path}，跳过此帧。")
            continue

        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or ir_img is None:
            continue

        # 对齐尺寸 (防止原图 RGB 和 IR 尺寸有极微小差异)
        if ir_img.shape[:2] != rgb_img.shape[:2]:
            ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))

        # 必须增加一个维度，将 (H, W) 变成 (H, W, 1)，否则 concatenate 会报错
        # ir_img = np.expand_dims(ir_img, axis=-1)

        # 拼接成 4 通道
        img_4c = np.concatenate([rgb_img, ir_img], axis=-1)  # [H, W, 4]

        # 预处理：LetterBox 缩放到 800x800
        img_padded, ratio, (pad_w, pad_h) = letterbox_4c(img_4c, new_shape=(800, 800))

        # HWC 转换为 CHW，并转换为 Tensor
        tensor = img_padded.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor).float() / 255.0
        tensor = tensor.unsqueeze(0).to(model.device)

        # 模型前向推理
        results = model(tensor, verbose=False)  # verbose=False 关闭每帧推理的打印

        # 解析结果 & 逆向画框
        obb_preds = results[0].obb
        draw_img = rgb_img.copy()

        if obb_preds is not None and len(obb_preds) > 0:
            points = obb_preds.xyxyxyxy.cpu().numpy()
            classes = obb_preds.cls.cpu().numpy()
            confs = obb_preds.conf.cpu().numpy()

            for pts, cls, conf in zip(points, classes, confs):
                # 坐标还原
                pts[..., 0] = (pts[..., 0] - pad_w) / ratio
                pts[..., 1] = (pts[..., 1] - pad_h) / ratio

                pts = np.int32(pts)
                cls_id = int(cls)
                cls_name = names[cls_id]
                color = colors[cls_id % len(colors)]

                # 画旋转多边形 (OBB)
                cv2.polylines(draw_img, [pts], isClosed=True, color=color, thickness=2)

                # 写上类别和置信度
                text_x, text_y = pts[0][0], pts[0][1]
                label_text = f"{cls_name} {conf:.2f}"
                cv2.putText(draw_img, label_text, (text_x, text_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 写入视频帧
        video_writer.write(draw_img)

    # 5. 释放资源
    video_writer.release()
    print(f"视频推理完成！结果已保存至: {output_video_path}")


if __name__ == '__main__':
    # ================= 配置区域 =================
    # 模型路径
    WEIGHTS_PATH = "F:/work/OmniAero-OBB/src/best_4.pt"

    # 图像序列文件夹路径 (包含 000001.jpg, 000002.jpg 等)
    RGB_DIR = "F:/work/OmniAero-OBB/src/bus_002/rgb"
    IR_DIR = "F:/work/OmniAero-OBB/src/bus_002/ir"

    # 输出的视频路径
    OUTPUT_VIDEO = "fusion_inference_output_2.mp4"

    # 视频帧率 (根据你原本数据集的帧率调整，通常为 25 或 30)
    FPS = 25
    # ==========================================

    run_fusion_video_inference(RGB_DIR, IR_DIR, WEIGHTS_PATH, OUTPUT_VIDEO, FPS)