import cv2
import torch
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO


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


def run_fusion_inference(rgb_path, ir_path, weights_path):
    print("正在加载融合模型...")
    model = YOLO(weights_path)
    names = model.names
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    rgb_img = cv2.imread(rgb_path)
    ir_img_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if rgb_img is None or ir_img_gray is None:
        raise ValueError("无法读取图像，请检查路径！")

    # 对齐尺寸
    if ir_img_gray.shape[:2] != rgb_img.shape[:2]:
        ir_img_gray = cv2.resize(ir_img_gray, (rgb_img.shape[1], rgb_img.shape[0]))




    # 拼接成 4 通道
    img_4c = np.concatenate([rgb_img, ir_img_gray], axis=-1)

    # 预处理
    img_padded, ratio, (pad_w, pad_h) = letterbox_4c(img_4c, new_shape=(800, 800))
    tensor = img_padded.transpose(2, 0, 1)
    tensor = torch.from_numpy(tensor).float() / 255.0
    tensor = tensor.unsqueeze(0).to(model.device)

    print("正在进行多模态 OBB 推理...")
    results = model(tensor)
    obb_preds = results[0].obb

    detections = []
    if obb_preds is not None and len(obb_preds) > 0:
        points = obb_preds.xyxyxyxy.cpu().numpy()
        classes = obb_preds.cls.cpu().numpy()
        confs = obb_preds.conf.cpu().numpy()

        for pts, cls, conf in zip(points, classes, confs):
            pts[..., 0] = (pts[..., 0] - pad_w) / ratio
            pts[..., 1] = (pts[..., 1] - pad_h) / ratio
            pts = np.int32(pts)
            cls_id = int(cls)

            detections.append({
                'pts': pts,
                'cls_name': names[cls_id],
                'conf': conf,
                'color': colors[cls_id % len(colors)]
            })

        print(f"检测完成！共发现 {len(points)} 个目标。")
    else:
        print("未检测到任何目标。")

    # 将 IR 转为 BGR，方便后续画彩色的框
    ir_img_bgr = cv2.cvtColor(ir_img_gray, cv2.COLOR_GRAY2BGR)
    return rgb_img, ir_img_bgr, detections


def save_results_to_folder(rgb_img, ir_img, detections, output_dir="fusion_results"):
    """把带框的图保存到指定文件夹中"""
    os.makedirs(output_dir, exist_ok=True)

    rgb_draw = rgb_img.copy()
    ir_draw = ir_img.copy()

    for det in detections:
        pts = det['pts']
        color = det['color']
        text = f"{det['cls_name']} {det['conf']:.2f}"

        # 画 RGB
        cv2.polylines(rgb_draw, [pts], True, color, 2)
        cv2.putText(rgb_draw, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 画 IR
        cv2.polylines(ir_draw, [pts], True, color, 2)
        cv2.putText(ir_draw, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    rgb_out_path = os.path.join(output_dir, "RGB_result.jpg")
    ir_out_path = os.path.join(output_dir, "IR_result.jpg")

    cv2.imwrite(rgb_out_path, rgb_draw)
    cv2.imwrite(ir_out_path, ir_draw)
    print(f"结果图片已保存至文件夹: {os.path.abspath(output_dir)}")


# ================= GUI 交互界面类 =================
class ImageViewer(tk.Tk):
    def __init__(self, rgb_img, ir_img, detections):
        super().__init__()
        self.title("多模态 OBB 检测结果查看器")

        # 转换 BGR 到 RGB 给 Tkinter 使用
        self.rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        self.ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
        self.detections = detections

        # 状态变量
        self.current_mode = "RGB"  # "RGB" 或 "IR"
        self.show_boxes = True
        self.hovered_idx = -1

        # 计算缩放以适应屏幕 (假设最大显示尺寸为 1000x800)
        h, w = self.rgb_img.shape[:2]
        self.scale = min(1000 / w, 800 / h) if w > 1000 or h > 800 else 1.0
        self.disp_w, self.disp_h = int(w * self.scale), int(h * self.scale)

        # 处理图片和坐标缩放
        self.rgb_disp = cv2.resize(self.rgb_img, (self.disp_w, self.disp_h))
        self.ir_disp = cv2.resize(self.ir_img, (self.disp_w, self.disp_h))

        # 界面布局
        self.setup_ui()
        self.update_canvas()

    def setup_ui(self):
        # 顶部按钮区
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=5)

        self.btn_switch = ttk.Button(btn_frame, text="切换到 IR", command=self.toggle_image)
        self.btn_switch.pack(side=tk.LEFT, padx=5)

        self.btn_box = ttk.Button(btn_frame, text="隐藏框", command=self.toggle_boxes)
        self.btn_box.pack(side=tk.LEFT, padx=5)

        self.lbl_info = ttk.Label(btn_frame, text="鼠标悬浮到框上可查看填充效果", foreground="blue")
        self.lbl_info.pack(side=tk.LEFT, padx=20)

        # 图像显示区
        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="gray")
        self.canvas.pack(side=tk.TOP)

        # 绑定鼠标移动事件
        self.canvas.bind('<Motion>', self.on_mouse_move)

    def toggle_image(self):
        self.current_mode = "IR" if self.current_mode == "RGB" else "RGB"
        self.btn_switch.config(text="切换到 RGB" if self.current_mode == "IR" else "切换到 IR")
        self.update_canvas()

    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self.btn_box.config(text="显示框" if not self.show_boxes else "隐藏框")
        self.update_canvas()

    def bgr_to_hex(self, bgr_color):
        """将 OpenCV 的 BGR 元组转为 Tkinter 的 Hex 颜色"""
        return f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"

    def update_canvas(self):
        self.canvas.delete("all")

        # 显示底图
        img_array = self.rgb_disp if self.current_mode == "RGB" else self.ir_disp
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_array))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # 画框
        if self.show_boxes:
            for idx, det in enumerate(self.detections):
                # 坐标按比例缩放
                pts_scaled = det['pts'] * self.scale
                # 展平为 [x1, y1, x2, y2, x3, y3, x4, y4]
                flat_pts = pts_scaled.flatten().tolist()

                color_hex = self.bgr_to_hex(det['color'])

                # 判断是否是悬浮高亮状态
                if idx == self.hovered_idx:
                    # 悬浮时填充颜色（高亮）
                    self.canvas.create_polygon(flat_pts, outline=color_hex, fill=color_hex, width=3, stipple="gray50")
                    # 悬浮时字稍微变大
                    self.canvas.create_text(flat_pts[0], flat_pts[1] - 10, text=f"{det['cls_name']} {det['conf']:.2f}",
                                            fill=color_hex, anchor=tk.SW, font=("Arial", 12, "bold"))
                else:
                    # 正常状态不填充
                    self.canvas.create_polygon(flat_pts, outline=color_hex, fill="", width=2)
                    self.canvas.create_text(flat_pts[0], flat_pts[1] - 10, text=f"{det['cls_name']} {det['conf']:.2f}",
                                            fill=color_hex, anchor=tk.SW, font=("Arial", 10))

    def on_mouse_move(self, event):
        if not self.show_boxes:
            return

        x, y = event.x, event.y
        new_hovered = -1

        # 将鼠标坐标转为原图坐标
        orig_x, orig_y = x / self.scale, y / self.scale

        # 检查鼠标在哪个框内 (逆序遍历，优先触发最上层的框)
        for idx in range(len(self.detections) - 1, -1, -1):
            pts = self.detections[idx]['pts']
            # 使用 cv2.pointPolygonTest 判断点是否在多边形内
            if cv2.pointPolygonTest(pts, (orig_x, orig_y), False) >= 0:
                new_hovered = idx
                break

        # 只有当悬浮状态改变时才重绘，节省性能
        if new_hovered != self.hovered_idx:
            self.hovered_idx = new_hovered
            self.update_canvas()


if __name__ == '__main__':
    # ================= 配置区域 =================
    WEIGHTS_PATH = "F:/work/OmniAero-OBB/src/best.pt"
    RGB_TEST_PATH = "F:/work/OmniAero-OBB/src/bus_008/rgb/000000.jpg"
    IR_TEST_PATH = "F:/work/OmniAero-OBB/src/bus_008/ir/000000.jpg"
    # ==========================================

    # 1. 运行推理，获取原始图和检测坐标
    rgb_img, ir_img_bgr, detections = run_fusion_inference(RGB_TEST_PATH, IR_TEST_PATH, WEIGHTS_PATH)

    # 2. 将画好框的图保存到一个文件夹中
    save_results_to_folder(rgb_img, ir_img_bgr, detections, output_dir="fusion_results")

    # 3. 启动交互式界面
    print("正在打开交互窗口...")
    app = ImageViewer(rgb_img, ir_img_bgr, detections)
    app.mainloop()