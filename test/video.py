import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ========================================================
# 1. 兼容性导入 NMS (自动适配新老版本及定制版 OBB)
# ========================================================
try:
    from ultralytics.utils.nms import non_max_suppression

    NMS_KWARGS = {'rotated': True}  # 官方版本需要加上 rotated=True 标志
except ImportError:
    try:
        from ultralytics.utils.ops import non_max_suppression

        NMS_KWARGS = {'rotated': True}
    except ImportError:
        try:
            from ultralytics.utils.ops import non_max_suppression_obb as non_max_suppression

            NMS_KWARGS = {}  # 专属的 obb 函数通常不需要额外传 rotated
        except ImportError:
            from ultralytics.models.yolo.obb.predict import non_max_suppression_obb as non_max_suppression

            NMS_KWARGS = {}


def letterbox_4c(img, new_shape=(800, 800), color=(114, 114, 114, 0)):
    """自定义 4通道 LetterBox 缩放"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (left, top)


def process_video_fusion(rgb_dir, ir_dir, weights_path, output_video_path, fps=20.0):
    print(f"正在加载融合模型: {weights_path}")
    model = YOLO(weights_path)
    names = model.names

    frame_names = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
    if not frame_names:
        raise ValueError(f"在 {rgb_dir} 中没有找到图片，请检查路径。")

    print(f"共发现 {len(frame_names)} 帧图片，准备开始推理...")

    out_video = None

    for idx, frame_name in enumerate(frame_names):
        rgb_path = os.path.join(rgb_dir, frame_name)
        ir_path = os.path.join(ir_dir, frame_name)

        if not os.path.exists(ir_path):
            continue

        # --- 1. 读取图像 ---
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or ir_img is None:
            continue

        if out_video is None:
            orig_h, orig_w = rgb_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))
            print(f"视频分辨率已设置为 {orig_w}x{orig_h}, FPS: {fps}")

        # --- 2. 图像预处理 (含维度防崩溃机制) ---
        if len(ir_img.shape) == 3 and ir_img.shape[2] == 3:
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)

        if ir_img.shape[:2] != rgb_img.shape[:2]:
            ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))

        ir_img = ir_img.reshape((ir_img.shape[0], ir_img.shape[1], 1))
        img_4c = np.concatenate([rgb_img, ir_img], axis=-1)

        img_padded, ratio, (pad_w, pad_h) = letterbox_4c(img_4c, new_shape=(800, 800))

        tensor = img_padded.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor).float() / 255.0
        tensor = tensor.unsqueeze(0).to(model.device)  # [1, 4, 800, 800]

        # --- 3. 核心修复：纯净的前向推理 (防 fuse 破坏结构) ---
        # 坚决不使用 model(tensor)！直接调用底层 nn.Module 前向传播！
        with torch.no_grad():
            results = model.model(tensor)

        raw_preds = results[0] if isinstance(results, tuple) else results

        # --- 4. 纯手工 OBB NMS 后处理 ---
        preds = non_max_suppression(
            raw_preds,
            conf_thres=0.25,
            iou_thres=0.45,
            nc=len(names),
            **NMS_KWARGS
        )[0]  # 取出第一张图的预测结果 [N, 7] 即 (x, y, w, h, angle, conf, cls)

        # --- 5. 坐标逆缩放与可视化 ---
        draw_img = rgb_img.copy()  # 在 1080p 的原图上画框

        if preds is not None and len(preds) > 0:
            # 此时的 preds 是基于 800x800 的坐标，我们需要将其数学映射回 1080p 原尺寸
            preds[:, 0] = (preds[:, 0] - pad_w) / ratio  # 中心点 x
            preds[:, 1] = (preds[:, 1] - pad_h) / ratio  # 中心点 y
            preds[:, 2] /= ratio  # 宽度 w
            preds[:, 3] /= ratio  # 高度 h
            # angle 不受缩放影响，无需调整

            # 将映射回真实尺寸的坐标丢入官方 Results 对象，白嫖它的极简画图功能
            res = Results(
                orig_img=draw_img,
                path=rgb_path,
                names=names,
                obb=preds
            )
            draw_img = res.plot(line_width=2)

        # --- 6. 写入视频与显示 ---
        out_video.write(draw_img)

        cv2.imshow('Multimodal OBB Inference', cv2.resize(draw_img, (800, int(800 * orig_h / orig_w))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (idx + 1) % 10 == 0 or (idx + 1) == len(frame_names):
            print(f"进度: {idx + 1}/{len(frame_names)} 帧")

    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()
    print(f"推理完毕！视频已成功保存至: {output_video_path}")


if __name__ == '__main__':
    # 填入你实际的路径
    WEIGHTS_PATH = "../src/best_3.pt"
    RGB_DIR = "F:/work/OmniAero-OBB/src/bus_031/rgb"
    IR_DIR = "F:/work/OmniAero-OBB/src/bus_031/ir"
    OUTPUT_VIDEO = "fusion_inference_video.mp4"
    FPS = 20.0

    process_video_fusion(RGB_DIR, IR_DIR, WEIGHTS_PATH, OUTPUT_VIDEO, fps=FPS)