from graphviz import Digraph


def draw_overall_architecture():
    """图1：整体宏观网络架构 (保持不变，不展开 Fusion)"""
    dot = Digraph('YOLOv8_Overall', format='png')
    dot.attr(rankdir='TB', size='6,8', dpi='300')
    dot.attr('node', shape='box', style='filled,rounded', fontname='sans-serif', margin='0.2')
    dot.attr('edge', fontname='sans-serif', fontsize='10')

    dot.node('Input', 'Input Images\n(RGB + IR = 4 Ch)', fillcolor='#E8DAEF', shape='cylinder')
    dot.node('Fusion', 'Early Fusion Module\n(Cross-Modal Fusion)', fillcolor='#FDEBD0', penwidth='2')

    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label='YOLOv8 Backbone', style='rounded,dashed', color='gray')
        c.node('P3', 'Stage P3 (Small Obj)', fillcolor='#D6EAF8')
        c.node('P4', 'Stage P4 (Medium Obj)', fillcolor='#D6EAF8')
        c.node('P5', 'Stage P5 (Large Obj)', fillcolor='#D6EAF8')

    dot.node('Neck', 'PANet Neck\n(Multi-scale Fusion)', fillcolor='#D5F5E3', shape='invtrapezium')
    dot.node('Head', 'OBB Detection Head\n[cx, cy, w, h, angle, class]', fillcolor='#FCF3CF', penwidth='2',
             shape='box3d')

    dot.edges([('Input', 'Fusion'), ('Fusion', 'P3'), ('P3', 'P4'), ('P4', 'P5')])
    dot.edge('P3', 'Neck')
    dot.edge('P4', 'Neck')
    dot.edge('P5', 'Neck')
    dot.edge('Neck', 'Head')

    dot.render('yolov8_obb_overall_framework', view=True, cleanup=True)
    print("图1 (整体架构) 已生成：yolov8_obb_overall_framework.png")


def draw_simplified_fusion_module():
    """图2：Fusion 模块内部构造 (高度抽象，不罗列底层算子)"""
    dot = Digraph('Fusion_Simplified', format='png')
    dot.attr(rankdir='TB', size='6,8', dpi='300')
    dot.attr('node', shape='box', style='filled,rounded', fontname='sans-serif', margin='0.2')
    dot.attr('edge', fontname='sans-serif', fontsize='10')

    # 1. 输入层
    dot.node('Input', 'Input Image [4 Channels]', fillcolor='#E8DAEF', shape='folder')

    # 2. 独立特征提取 (分为 RGB 和 IR 两支)
    with dot.subgraph(name='cluster_extract') as c:
        c.attr(label='Independent Spatial Extraction', style='dashed', color='gray')
        c.node('RGB_Conv', 'RGB Branch Conv\n(Extract Visible Features)', fillcolor='#FADBD8')  # 浅红
        c.node('IR_Conv', 'IR Branch Conv\n(Extract Thermal Features)', fillcolor='#D6EAF8')  # 浅蓝
        # 强制在同一行
        c.body.append('{rank=same; RGB_Conv; IR_Conv;}')

    # 3. 动态注意力机制 (高度概括，隐藏细节)
    dot.node('Attention', 'Cross-Modal Attention\n(Generate Dynamic Weights: W_rgb, W_ir)', fillcolor='#FCF3CF',
             shape='hexagon')

    # 4. 加权融合
    dot.node('Fusion', 'Complementary Weighted Fusion\n(Feat_rgb × W_rgb) + (Feat_ir × W_ir)', fillcolor='#D5F5E3',
             penwidth='2')

    # 5. 输出
    dot.node('Output', 'Fused Feature Map\n(To YOLO Backbone)', fillcolor='white')

    # === 构建连接 ===
    dot.edge('Input', 'RGB_Conv', label=' 3 Ch (RGB)')
    dot.edge('Input', 'IR_Conv', label=' 1 Ch (IR)')

    # 特征流向注意力机制计算权重
    dot.edge('RGB_Conv', 'Attention', label=' Feat_rgb', color='#E74C3C')
    dot.edge('IR_Conv', 'Attention', label=' Feat_ir', color='#3498DB')

    # 注意力机制输出权重，特征流向融合模块
    dot.edge('Attention', 'Fusion', label=' Weights', style='dashed')
    dot.edge('RGB_Conv', 'Fusion', color='#E74C3C')
    dot.edge('IR_Conv', 'Fusion', color='#3498DB')

    dot.edge('Fusion', 'Output')

    dot.render('fusion_module_simplified', view=True, cleanup=True)
    print("图2 (Fusion简图) 已生成：fusion_module_simplified.png")


if __name__ == '__main__':
    draw_overall_architecture()
    draw_simplified_fusion_module()