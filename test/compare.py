import torch
from ultralytics import YOLO
import numpy as np

def run_evaluation(model_path, data_yaml, task_name):
    """通用评估函数"""
    print(f"\n{'='*20} 正在评估: {task_name} {'='*20}")
    try:
        model = YOLO(model_path)
        # 强制使用 test 集进行评估
        metrics = model.val(
            data=data_yaml,
            split='test',
            imgsz=800,
            batch=32,
            device=0,
            name=task_name,
            verbose=False, # 减少刷屏
            plots=False    # 暂时不画图，只取数据
        )
        return model, metrics
    except Exception as e:
        print(f"❌ {task_name} 评估失败: {e}")
        return None, None

def extract_details(model, metrics):
    """从 metrics 对象中“榨干”所有数据"""
    # 1. 基础信息
    class_indices = metrics.box.ap_class_index
    class_names = [model.names[i] for i in class_indices]
    
    # 2. 精度指标 (转为 list)
    # metrics.box.p 等通常是 (nc,) 的数组
    p_per_class = metrics.box.p.tolist()
    r_per_class = metrics.box.r.tolist()
    
    # mAP@0.5 (Per Class)
    ap50_per_class = metrics.box.ap50.tolist()
    
    # 【关键修正】：mAP@0.5:0.95 (Per Class)
    # metrics.box.maps 已经是每个类别的 mAP50-95 值了，不需要 mean(1)
    ap5095_per_class = metrics.box.maps.tolist()

    # 3. 总体指标 (Global)
    mean_p  = metrics.box.mp
    mean_r  = metrics.box.mr
    map50   = metrics.box.map50
    map5095 = metrics.box.map

    # 4. 速度指标 (单位 ms)
    # 累加预处理、推理、后处理时间
    t = metrics.speed
    speed_ms = t['inference'] + t['postprocess'] + t['preprocess']
    # 防止除以0错误
    fps = 1000.0 / speed_ms if speed_ms > 0 else 0.0

    return {
        "names": class_names,
        "p_list": p_per_class,
        "r_list": r_per_class,
        "ap50_list": ap50_per_class,
        "ap5095_list": ap5095_per_class,
        "mean_p": mean_p,
        "mean_r": mean_r,
        "map50": map50,
        "map5095": map5095,
        "fps": fps
    }

def main():
    # ================= 配置区域 =================
    # 1. 官方基准模型 (纯RGB)
    base_pt = "/mnt/workspace/OmniAero-OBB/runs/obb/runs/obb/Baseline_RGB_HighRes/weights/best.pt"
    base_yaml = "/mnt/workspace/OmniAero-OBB/test/baseline.yaml" # 务必确保里面是 ch: 3

    # 2. 你的融合模型 (RGB+IR)
    fusion_pt = "/mnt/workspace/OmniAero-OBB/runs/obb/OmniAero_Fusion_HighRes5/weights/best.pt"
    fusion_yaml = "/mnt/workspace/OmniAero-OBB/test/dataset.yaml" # 务必确保里面是 ch: 4
    # ===========================================

    # --- 运行评估 ---
    model_b, res_b = run_evaluation(base_pt, base_yaml, "Baseline_RGB")
    model_f, res_f = run_evaluation(fusion_pt, fusion_yaml, "Fusion_RGB_IR")

    if not res_b or not res_f:
        print("评估中断，请检查报错。")
        return

    # --- 提取详细数据 ---
    data_b = extract_details(model_b, res_b)
    data_f = extract_details(model_f, res_f)

    # --- 打印终端对比表格 ---
    print("\n" + "🚀"*15 + " 深度性能对比报告 " + "🚀"*15)
    print(f"{'指标':<15} | {'基准模型 (RGB)':<15} | {'融合模型 (RGB+IR)':<15} | {'提升幅度'}")
    print("-" * 70)
    
    # 总体数据
    metrics_list = [
        ("mAP@0.5", data_b['map50'], data_f['map50']),
        ("mAP@0.5:0.95", data_b['map5095'], data_f['map5095']),
        ("Precision", data_b['mean_p'], data_f['mean_p']),
        ("Recall", data_b['mean_r'], data_f['mean_r']),
        ("FPS (速度)", data_b['fps'], data_f['fps'])
    ]

    for title, v_b, v_f in metrics_list:
        diff = (v_f - v_b)
        # 对于FPS，提升计算方式稍微不同，这里只算差值
        color = "✅" if diff > 0 else "🔻"
        print(f"{title:<15} | {v_b:<15.4f} | {v_f:<15.4f} | {color} {diff:+.4f}")

    print("-" * 70)
    print("📊 详细类别 mAP@0.5 对比:")
    for i, name in enumerate(data_b['names']):
        sb = data_b['ap50_list'][i]
        sf = data_f['ap50_list'][i]
        print(f"{name:<15} | {sb:<15.4f} | {sf:<15.4f} | {(sf-sb)*100:+.2f}%")

    # --- 生成绘图代码 ---
    print("\n" + "="*20 + " 复制以下数据到绘图脚本 " + "="*20)
    
    # 构建包含 Overall 的列表
    labels = data_b['names'] + ['Overall']
    
    # mAP50 数据
    map50_b_list = [round(x, 3) for x in data_b['ap50_list']] + [round(data_b['map50'], 3)]
    map50_f_list = [round(x, 3) for x in data_f['ap50_list']] + [round(data_f['map50'], 3)]
    
    # Recall 数据
    rec_b_list = [round(x, 3) for x in data_b['r_list']] + [round(data_b['mean_r'], 3)]
    rec_f_list = [round(x, 3) for x in data_f['r_list']] + [round(data_f['mean_r'], 3)]

    print(f"labels = {labels}")
    print(f"map50_baseline  = {map50_b_list}")
    print(f"map50_fusion    = {map50_f_list}")
    print(f"recall_baseline = {rec_b_list  }")
    print(f"recall_fusion   = {rec_f_list  }")
    print("="*65)

if __name__ == "__main__":
    main()