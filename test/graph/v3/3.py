from graphviz import Digraph


def draw_yolo11_vertical_layout():
    # 使用 ortho 保持连线直角，rankdir='TB' 确保从上到下
    dot = Digraph('YOLO11_Vertical_Matrix', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.6')

    # 统一的节点样式
    dot.attr('node', fontname='sans-serif', fontsize='10', shape='box', style='filled,rounded', margin='0.15,0.1')
    dot.attr('edge', fontname='sans-serif', fontsize='9')

    # ==================== 1. 定义所有节点 ====================

    # 顶部 Input (占据独立一行)
    dot.node('Input', 'Input Image\n[4 Ch: RGB+IR]', fillcolor='#E8DAEF', shape='cylinder')

    # 第二行：Stem 层
    # RGB 提取 + 门控
    dot.node('RGB_Ext', '{ L0: RGB_Ext (64) | <Gate> 门控 Gate(Weather-Adaptive) }', shape='Mrecord',
             fillcolor='#FADBD8')
    # IR 提取
    dot.node('IR_Ext', 'L1: IR_Extract (64)\n[Structure Extractor]', fillcolor='#D6EAF8')

    # 左一柱：Auxiliary Backbone (RGB)
    dot.node('RGB_P3', 'L14: RGB_P3\n(C3k2, 256)', fillcolor='#FDEDEC', color='#E74C3C', penwidth='1.5')
    dot.node('RGB_P4', 'L15: RGB_P4\n(Conv, 512)', fillcolor='#FDEDEC', color='#E74C3C', penwidth='1.5')
    dot.node('RGB_P5', 'L16: RGB_P5\n(Conv, 1024)', fillcolor='#FDEDEC', color='#E74C3C', penwidth='1.5')

    # 左二柱：Main Backbone (IR/Mixed)
    dot.node('Main_P3', 'L6: Main_P3\n(C3k2, 256)', fillcolor='#EBF5FB', color='#2980B9', penwidth='1.5')
    dot.node('Main_P4', 'L8: Main_P4\n(C3k2, 512)', fillcolor='#EBF5FB', color='#2980B9', penwidth='1.5')
    dot.node('Main_P5', 'L12: Main_P5\n(SPPF+C2PSA, 1024)', fillcolor='#EBF5FB', color='#2980B9', penwidth='1.5')

    # 右二柱：融合层 (Mamba/CMA)
    dot.node('Mamba', 'L23: Mamba_Fusion\n[P3]', fillcolor='#FEF9E7', shape='hexagon', color='#F39C12')
    dot.node('CMA_P4', 'L17: CMA_P4\n[P4]', fillcolor='#FEF9E7', shape='hexagon', color='#F39C12')
    dot.node('CMA_P5', 'L18: CMA_P5\n[P5]', fillcolor='#FEF9E7', shape='hexagon', color='#F39C12')

    # 右一柱：Neck Outputs
    dot.node('P3_OUT', 'L24: P3_Out\n(Concat+C3k2)', fillcolor='#EAFAF1', color='#27AE60', penwidth='1.5')
    dot.node('P4_OUT', 'L27: P4_Out\n(Concat+C3k2)', fillcolor='#EAFAF1', color='#27AE60', penwidth='1.5')
    dot.node('P5_OUT', 'L30: P5_Out\n(Concat+C3k2)', fillcolor='#EAFAF1', color='#27AE60', penwidth='1.5')

    # 底部 Head
    dot.node('OBB', 'L31: OBB Head', fillcolor='#FAD7A1', shape='folder', penwidth='2')

    # ==================== 2. 布局对齐约束 ====================

    # 强制 Input 在最上方
    with dot.subgraph() as s:
        s.attr(rank='source')
        s.node('Input')

    # 每一层水平对齐
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('RGB_Ext');
        s.node('IR_Ext')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('RGB_P3');
        s.node('Main_P3');
        s.node('Mamba');
        s.node('P3_OUT')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('RGB_P4');
        s.node('Main_P4');
        s.node('CMA_P4');
        s.node('P4_OUT')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('RGB_P5');
        s.node('Main_P5');
        s.node('CMA_P5');
        s.node('P5_OUT')

    # 垂直隐形钢筋（确保列直立）
    dot.edge('RGB_Ext', 'RGB_P3', style='invis')
    dot.edge('RGB_P3', 'RGB_P4', style='invis')
    dot.edge('RGB_P4', 'RGB_P5', style='invis')

    dot.edge('IR_Ext', 'Main_P3', style='invis')
    dot.edge('Main_P3', 'Main_P4', style='invis')
    dot.edge('Main_P4', 'Main_P5', style='invis')

    dot.edge('P3_OUT', 'P4_OUT', style='invis')
    dot.edge('P4_OUT', 'P5_OUT', style='invis')

    # ==================== 3. 逻辑连线 ====================

    # 1. Input 到第一层 (由 constraint=true 保证其在上方)
    dot.edge('Input', 'RGB_Ext')
    dot.edge('Input', 'IR_Ext')

    # 2. 天气门控 (横向)
    dot.edge('RGB_Ext:Gate', 'IR_Ext', label='Weather Gating', constraint='false')

    # 3. 垂直主干连线
    dot.edge('RGB_Ext', 'RGB_P3', color='#E74C3C')
    dot.edge('RGB_P3', 'RGB_P4', color='#E74C3C')
    dot.edge('RGB_P4', 'RGB_P5', color='#E74C3C')

    dot.edge('IR_Ext', 'Main_P3', color='#2980B9')
    dot.edge('Main_P3', 'Main_P4', color='#2980B9')
    dot.edge('Main_P4', 'Main_P5', color='#2980B9')

    # 4. 横向融合连线
    # P3
    dot.edge('RGB_P3', 'Mamba', color='#E74C3C')
    dot.edge('Main_P3', 'Mamba', color='#2980B9')
    dot.edge('Mamba', 'P3_OUT')
    # P4
    dot.edge('RGB_P4', 'CMA_P4', color='#E74C3C', style='dashed')
    dot.edge('Main_P4', 'CMA_P4', color='#2980B9')
    dot.edge('CMA_P4', 'P4_OUT')
    # P5
    dot.edge('RGB_P5', 'CMA_P5', color='#E74C3C', style='dashed')
    dot.edge('Main_P5', 'CMA_P5', color='#2980B9')
    dot.edge('CMA_P5', 'P5_OUT')

    # 5. Neck 路径 (上采样和下采样)
    dot.edge('P3_OUT', 'P4_OUT', color='#27AE60', label=' Down')
    dot.edge('P4_OUT', 'P5_OUT', color='#27AE60', label=' Down')
    dot.edge('CMA_P5', 'P4_OUT', color='#8E44AD', label=' Up', constraint='false')
    dot.edge('P4_OUT', 'P3_OUT', color='#8E44AD', label=' Up', constraint='false')

    # 6. 指向 Head
    dot.edge('P3_OUT', 'OBB')
    dot.edge('P4_OUT', 'OBB')
    dot.edge('P5_OUT', 'OBB')

    dot.render('yolo11_vertical_layout', view=True, cleanup=True)
    print("架构图已更新：Input 已移至上方，P4_Inter 已删除。")


if __name__ == '__main__':
    draw_yolo11_vertical_layout()