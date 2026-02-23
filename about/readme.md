# Readme 


# idea
1. CNN -> Transformer
2. 不确定度具体到每个像素
3. 对比强化学习
4. 小神经头的进入


# kind
0: 'bus'
1: 'car'
2: 'feright_car'
3: 'truck'
4: 'van'


# ignore
"torch>=1.8.0",
"torch>=1.8.0,!=2.4.0; sys_platform == 'win32'",
# Windows CPU errors w/ 2.4.0 https://github.com/ultralytics/ultralytics/issues 15049
"torchvision>=0.9.0",