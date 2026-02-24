from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8n-obb.pt') 

    model.train(
        # --- 变量 (Variable) ---
        data='/mnt/workspace/OmniAero-OBB/test/baseline.yaml', # 必须确保这里面 ch: 3
        

        imgsz=800,           
        epochs=150,          
        batch=56,            
        workers=8,           
        device=0,
        amp=True,            
        patience=50,         
        
        # --- 数据增强参数 (至关重要) ---
        # 融合模型为了兼容4通道关掉了HSV，基准模型也要关掉，以示公平！
        hsv_h=0.0, 
        hsv_s=0.0, 
        hsv_v=0.0, 
        save=True,
        mosaic=1.0,          # 必须一致
        mixup=0.1,           # 必须一致
        
        project='runs/obb',  # 统一保存在 runs/obb 下，方便管理
        name='Baseline_RGB_HighRes' 
    )