from graphviz import Digraph


def draw_perfect_matrix_yolo11():
    # ==================== 美化版核心参数 ====================
    # 更宽松的间距 + 专业标题 + 列分组 cluster + 字体/颜色升级
    dot = Digraph('YOLO11_Perfect_Matrix_Beautified', format='png')
    dot.attr(rankdir='TB',
             splines='ortho',
             nodesep='0.85',      # 横向列间距更大，更“矩阵”感
             ranksep='1.05')      # 纵向层间距更大，避免拥挤

    # 整体标题（置顶）
    dot.attr(label='YOLO11 Perfect Matrix Architecture\n'
                   'RGB + IR 多模态 OBB 检测完美阵列结构',
             labelloc='t',
             fontsize='20',
             fontcolor='#2C3E50',
             fontname='sans-serif')

    # 统一节点/边样式（字体稍大、更清晰）
    dot.attr('node', fontname='Helvetica', fontsize='12',
             shape='box', style='filled,rounded', margin='0.22,0.14')
    dot.attr('edge', fontname='Helvetica', fontsize='11')

    # ==================== 1. 定义所有节点（标签微调更清晰） ====================
    # Input
    dot.node('Input', 'Input Image\n[4 Ch: RGB+IR]', fillcolor='#E8DAEF', shape='cylinder')

    # Stem
    dot.node('RGB_Ext', '{ L0: RGB_Ext (64) | <Gate> 门控 Gate\n(Weather-Adaptive) }',
             shape='Mrecord', fillcolor='#FADBD8')
    dot.node('IR_Ext', 'L1: IR_Extract (64)\n[Structure Extractor]', fillcolor='#D6EAF8')

    # 左柱：RGB Auxiliary
    dot.node('RGB_P3', 'L14: RGB_P3\n(C3k2, 256)', fillcolor='#FDEDEC', color='#E74C3C', penwidth='2.5')
    dot.node('RGB_P4', 'L15: RGB_P4\n(Conv, 512)', fillcolor='#FDEDEC', color='#E74C3C', penwidth='2.5')
    dot.node('RGB_P5', 'L16: RGB_P5\n(Conv, 1024)', fillcolor='#FDEDEC', color='#E74C3C', penwidth='2.5')

    # 中柱：Main Backbone
    dot.node('Main_P3', 'L6: Main_P3\n(C3k2, 256)', fillcolor='#EBF5FB', color='#2980B9', penwidth='2.5')
    dot.node('Main_P4', 'L8: Main_P4\n(C3k2, 512)', fillcolor='#EBF5FB', color='#2980B9', penwidth='2.5')
    dot.node('Main_P5', 'L12: Main_P5\n(SPPF+C2PSA, 1024)', fillcolor='#EBF5FB', color='#2980B9', penwidth='2.5')

    # 融合层（突出）
    dot.node('Mamba', 'L23: Mamba_Fusion\n[Cross-Modal P3]',
             fillcolor='#FEF9E7', shape='hexagon', color='#F39C12', penwidth='3')
    dot.node('CMA_P4', 'L17: CMA_P4\n[Enhanced P4]',
             fillcolor='#FEF9E7', shape='hexagon', color='#F39C12', penwidth='3')
    dot.node('CMA_P5', 'L18: CMA_P5\n[Enhanced P5]',
             fillcolor='#FEF9E7', shape='hexagon', color='#F39C12', penwidth='3')

    # FPN 中间
    dot.node('P4_Inter', 'L21: P4_Inter\n(Concat+C3k2)', fillcolor='#EAFAF1')

    # 右柱：Neck Outputs
    dot.node('P3_OUT', 'L24: P3_Out\n(Concat+C3k2)', fillcolor='#EAFAF1', color='#27AE60', penwidth='2.5')
    dot.node('P4_OUT', 'L27: P4_Out\n(Concat+C3k2)', fillcolor='#EAFAF1', color='#27AE60', penwidth='2.5')
    dot.node('P5_OUT', 'L30: P5_Out\n(Concat+C3k2)', fillcolor='#EAFAF1', color='#27AE60', penwidth='2.5')

    # Head（更醒目）
    dot.node('OBB', 'L31: OBB Head\n(Oriented Bounding Box)',
             fillcolor='#FAD7A1', shape='folder', penwidth='3.5')

    # ==================== 2. 列分组 Cluster（核心美化！让矩阵结构一目了然） ====================
    # 左列：RGB
    with dot.subgraph(name='cluster_rgb') as c:
        c.attr(style='filled', fillcolor='#FFF0E6', label='RGB Auxiliary Backbone\n(Weather-Adaptive)',
               fontsize='11', fontcolor='#E74C3C', color='#E74C3C', penwidth='2')

        c.node('RGB_Ext')
        c.node('RGB_P3')
        c.node('RGB_P4')
        c.node('RGB_P5')

    # 中列：Main IR
    with dot.subgraph(name='cluster_main') as c:
        c.attr(style='filled', fillcolor='#E6F0FA', label='Main IR/Mixed Backbone',
               fontsize='11', fontcolor='#2980B9', color='#2980B9', penwidth='2')

        c.node('IR_Ext')
        c.node('Main_P3')
        c.node('Main_P4')
        c.node('Main_P5')

    # 融合列
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(style='filled', fillcolor='#FFF9E6', label='Cross-Modal Fusion\n(Mamba + CMA)',
               fontsize='11', fontcolor='#F39C12', color='#F39C12', penwidth='2')

        c.node('Mamba')
        c.node('CMA_P4')
        c.node('CMA_P5')

    # 右列：Neck
    with dot.subgraph(name='cluster_neck') as c:
        c.attr(style='filled', fillcolor='#E6F6F0', label='Neck / FPN\n(Feature Pyramid)',
               fontsize='11', fontcolor='#27AE60', color='#27AE60', penwidth='2')

        c.node('P4_Inter')
        c.node('P3_OUT')
        c.node('P4_OUT')
        c.node('P5_OUT')

    # ==================== 3. 【行对齐】强制同一水平面 ====================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('RGB_Ext'); s.node('IR_Ext')

    with dot.subgraph() as s:
        s.attr(rank='same')  # P3 层
        s.node('RGB_P3'); s.node('Main_P3'); s.node('Mamba'); s.node('P3_OUT')

    with dot.subgraph() as s:
        s.attr(rank='same')  # P4 层
        s.node('RGB_P4'); s.node('Main_P4'); s.node('CMA_P4'); s.node('P4_Inter'); s.node('P4_OUT')

    with dot.subgraph() as s:
        s.attr(rank='same')  # P5 层
        s.node('RGB_P5'); s.node('Main_P5'); s.node('CMA_P5'); s.node('P5_OUT')

    # ==================== 4. 【钢筋对齐】隐形连线（防止任何扭曲） ====================
    # 垂直钢筋
    dot.edge('RGB_Ext', 'RGB_P3', style='invis', weight='200')
    dot.edge('RGB_P3', 'RGB_P4', style='invis', weight='200')
    dot.edge('RGB_P4', 'RGB_P5', style='invis', weight='200')

    dot.edge('IR_Ext', 'Main_P3', style='invis', weight='200')
    dot.edge('Main_P3', 'Main_P4', style='invis', weight='200')
    dot.edge('Main_P4', 'Main_P5', style='invis', weight='200')

    dot.edge('P3_OUT', 'P4_OUT', style='invis', weight='200')
    dot.edge('P4_OUT', 'P5_OUT', style='invis', weight='200')

    # 横向占位钢筋（保持列间距）
    dot.edge('RGB_P3', 'Main_P3', style='invis')
    dot.edge('Main_P3', 'Mamba', style='invis')
    dot.edge('Mamba', 'P3_OUT', style='invis')

    dot.edge('RGB_P4', 'Main_P4', style='invis')
    dot.edge('Main_P4', 'CMA_P4', style='invis')
    dot.edge('CMA_P4', 'P4_Inter', style='invis')
    dot.edge('P4_Inter', 'P4_OUT', style='invis')

    dot.edge('RGB_P5', 'Main_P5', style='invis')
    dot.edge('Main_P5', 'CMA_P5', style='invis')
    dot.edge('CMA_P5', 'P5_OUT', style='invis')

    # ==================== 5. 真实业务逻辑连线（保持原逻辑，微调美观） ====================
    # 输入
    dot.edge('Input', 'RGB_Ext', constraint='false', penwidth='1.8')
    dot.edge('Input', 'IR_Ext', constraint='false', penwidth='1.8')

    # 垂直主干（彩色加粗）
    dot.edge('RGB_Ext:Gate', 'RGB_P3', color='#E74C3C', constraint='false', penwidth='2', label=' Weather Gating')
    dot.edge('RGB_P3', 'RGB_P4', color='#E74C3C', constraint='false', penwidth='2')
    dot.edge('RGB_P4', 'RGB_P5', color='#E74C3C', constraint='false', penwidth='2')

    dot.edge('IR_Ext', 'Main_P3', color='#2980B9', constraint='false', penwidth='2')
    dot.edge('Main_P3', 'Main_P4', color='#2980B9', constraint='false', penwidth='2')
    dot.edge('Main_P4', 'Main_P5', color='#2980B9', constraint='false', penwidth='2')

    # Neck 下采样
    dot.edge('P3_OUT', 'P4_OUT', color='#27AE60', constraint='false', penwidth='2', label='Down 2×')
    dot.edge('P4_OUT', 'P5_OUT', color='#27AE60', constraint='false', penwidth='2', label='Down 2×')

    # 横向融合（融合边虚线突出）
    dot.edge('RGB_P3', 'Mamba', constraint='false', color='#E74C3C', penwidth='1.8')
    dot.edge('Main_P3', 'Mamba', constraint='false', color='#2980B9', penwidth='1.8')
    dot.edge('Mamba', 'P3_OUT', constraint='false', penwidth='2')

    dot.edge('RGB_P4', 'CMA_P4', constraint='false', color='#E74C3C', style='dashed', penwidth='1.8')
    dot.edge('Main_P4', 'CMA_P4', constraint='false', color='#2980B9', penwidth='1.8')
    dot.edge('CMA_P4', 'P4_Inter', constraint='false', penwidth='2')
    dot.edge('P4_Inter', 'P4_OUT', constraint='false', penwidth='2')

    dot.edge('RGB_P5', 'CMA_P5', constraint='false', color='#E74C3C', style='dashed', penwidth='1.8')
    dot.edge('Main_P5', 'CMA_P5', constraint='false', color='#2980B9', penwidth='1.8')
    dot.edge('CMA_P5', 'P5_OUT', constraint='false', penwidth='2')

    # FPN 上采样（紫色突出）
    dot.edge('CMA_P5', 'P4_Inter', constraint='false', color='#8E44AD', label='Up 2×', penwidth='2')
    dot.edge('P4_Inter', 'P3_OUT', constraint='false', color='#8E44AD', label='Up 2×', penwidth='2')

    # 指向 Head（加粗保持位置）
    dot.edge('P3_OUT', 'OBB', constraint='false', label='Small', penwidth='2')
    dot.edge('P4_OUT', 'OBB', constraint='false', label='Medium', penwidth='2')
    dot.edge('P5_OUT', 'OBB', constraint='false', label='Large', weight='300', penwidth='2.5')

    # ==================== 生成 ====================
    dot.render('yolo11_perfect_matrix_beautified', view=True, cleanup=True)
    print("✅ 美化完美阵列对齐版架构图已生成：yolo11_perfect_matrix_beautified.png")
    print("   （新增标题 + 4列分组框 + 更大间距 + 更醒目配色，更专业更漂亮！）")


if __name__ == '__main__':
    draw_perfect_matrix_yolo11()