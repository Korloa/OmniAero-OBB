import cv2
import torch
import numpy as np
from ultralytics import YOLO

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

def run_fusion_inference(rgb_path, ir_path, weights_path):
    # 1. 加载训练好的模型
    print("正在加载融合模型...")
    model = YOLO(weights_path)
    # 获取类别字典
    names = model.names
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)] # 为不同类别分配颜色

    # 2. 读取图像
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    
    if rgb_img is None or ir_img is None:
        raise ValueError("无法读取图像，请检查路径！")
        
    # 对齐尺寸 (防止原图 RGB 和 IR 尺寸有极微小差异)
    if ir_img.shape[:2] != rgb_img.shape[:2]:
        ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))
        
    ir_img = np.expand_dims(ir_img, axis=-1)

    # 3. 拼接成 4 通道
    img_4c = np.concatenate([rgb_img, ir_img], axis=-1)  # [H, W, 4]

    # 4. 预处理：LetterBox 缩放到 800x800
    img_padded, ratio, (pad_w, pad_h) = letterbox_4c(img_4c, new_shape=(800, 800))

    # HWC 转换为 CHW，并转换为 Tensor
    tensor = img_padded.transpose(2, 0, 1)
    tensor = torch.from_numpy(tensor).float() / 255.0  # 归一化到 0~1
    tensor = tensor.unsqueeze(0).to(model.device)      # 增加 Batch 维度: [1, 4, 800, 800]

    # 5. 模型前向推理！(直接喂入 Tensor，绕过 YOLO 自带的 3通道预处理)
    print("正在进行多模态 OBB 推理...")
    results = model(tensor)
    
    # 6. 解析结果 & 逆向画框
    obb_preds = results[0].obb
    
    # 准备一张画板（用原尺寸的 RGB 画框最清晰）
    draw_img = rgb_img.copy()

    if obb_preds is not None and len(obb_preds) > 0:
        # 获取 8个坐标点、类别和置信度
        # xyxyxyxy 形状: [N, 4, 2] 即 N个框，每个框 4 个点，每个点 (x, y)
        points = obb_preds.xyxyxyxy.cpu().numpy() 
        classes = obb_preds.cls.cpu().numpy()
        confs = obb_preds.conf.cpu().numpy()

        for pts, cls, conf in zip(points, classes, confs):
            # 将基于 800x800 的坐标，逆向还原到原始图像的尺寸
            pts[..., 0] = (pts[..., 0] - pad_w) / ratio
            pts[..., 1] = (pts[..., 1] - pad_h) / ratio
            
            pts = np.int32(pts) # 坐标转为整数
            cls_id = int(cls)
            cls_name = names[cls_id]
            color = colors[cls_id % len(colors)]

            # 画旋转多边形 (OBB)
            cv2.polylines(draw_img, [pts], isClosed=True, color=color, thickness=2)
            
            # 写上类别和置信度 (找最上面的一个点作为文本锚点)
            text_x, text_y = pts[0][0], pts[0][1]
            label_text = f"{cls_name} {conf:.2f}"
            cv2.putText(draw_img, label_text, (text_x, text_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
        print(f"检测完成！共发现 {len(points)} 个目标。")
    else:
        print("未检测到任何目标。")

    # 7. 保存并展示结果
    output_path = "fusion_inference_result2.jpg"
    cv2.imwrite(output_path, draw_img)
    print(f"结果已保存至: {output_path}")

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 替换为你实际训练出的 best.pt 路径
    WEIGHTS_PATH = "runs/obb/OmniAero_Fusion_HighRes5/weights/best.pt"
    
    # 挑一张你的测试图片和对应的红外图片
    RGB_TEST_PATH = "/mnt/workspace/OmniAero-OBB/src/dataset/test/images/test/00031.jpg"
    IR_TEST_PATH  = "/mnt/workspace/OmniAero-OBB/src/dataset/test/images/testr/00031.jpg"
    # ==========================================

    run_fusion_inference(RGB_TEST_PATH, IR_TEST_PATH, WEIGHTS_PATH)