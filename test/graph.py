from graphviz import Digraph


def draw_rgbt_yolo_simplified():
    # 设置图表参数
    # ratio='fill' 尝试填充空间，splines='ortho' 使用折线
    # ranksep 减小层级间距，nodesep 减小节点间距
    dot = Digraph(comment='RGBT YOLO Simplified', format='png')
    dot.attr(rankdir='TB', splines='ortho', dpi='300',
             nodesep='0.6', ranksep='0.5')

    # 全局节点样式：字号适中，高度压缩
    dot.attr('node', shape='box', style='filled, rounded',
             fontname='Arial', fontsize='11', height='0.6', width='1.8')
    dot.attr('edge', arrowsize='0.8', penwidth='1.2')

    # --- 1. 输入层 (并排) ---
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('rgb_in', 'RGB Image\n(Input)', fillcolor='#E1F5FE')
        s.node('t_in', 'Thermal Image\n(Input)', fillcolor='#FFEBEE')

    # --- 2. 骨干网络与融合层 (三个主要层级 P3, P4, P5) ---
    # 使用 Group 概念，将同一层级的节点强制对齐

    # P3 层级 (Small Object Features)
    dot.node('rgb_p3', 'RGB Backbone\n(Stage P3)', fillcolor='#BBDEFB')
    dot.node('fusion_p3', 'Fusion Module\n(P3)', fillcolor='#C8E6C9', shape='hexagon', width='2.0')
    dot.node('t_p3', 'Thermal Backbone\n(Stage P3)', fillcolor='#FFCDD2')

    # P4 层级 (Medium Object Features)
    dot.node('rgb_p4', 'RGB Backbone\n(Stage P4)', fillcolor='#BBDEFB')
    dot.node('fusion_p4', 'Fusion Module\n(P4)', fillcolor='#C8E6C9', shape='hexagon', width='2.0')
    dot.node('t_p4', 'Thermal Backbone\n(Stage P4)', fillcolor='#FFCDD2')

    # P5 层级 (Large Object Features)
    dot.node('rgb_p5', 'RGB Backbone\n(Stage P5)', fillcolor='#BBDEFB')
    dot.node('fusion_p5', 'Fusion Module\n(P5)', fillcolor='#C8E6C9', shape='hexagon', width='2.0')
    dot.node('t_p5', 'Thermal Backbone\n(Stage P5)', fillcolor='#FFCDD2')

    # --- 3. 强制对齐 (关键步骤：解决图太宽/太乱) ---
    # 每一行的 RGB、Fusion、Thermal 必须在同一高度
    with dot.subgraph() as s:
        s.attr(rank='same');
        s.node('rgb_p3');
        s.node('fusion_p3');
        s.node('t_p3')
    with dot.subgraph() as s:
        s.attr(rank='same');
        s.node('rgb_p4');
        s.node('fusion_p4');
        s.node('t_p4')
    with dot.subgraph() as s:
        s.attr(rank='same');
        s.node('rgb_p5');
        s.node('fusion_p5');
        s.node('t_p5')

    # --- 4. 聚合的 Head (简化金字塔) ---
    # 用一个大的块代表 FPN+PAN+Head，不再画细节
    dot.node('yolo_head', 'Feature Pyramid (Neck)\n&\nDetection Heads',
             fillcolor='#FFF9C4', shape='component', width='5', height='1.2')

    # --- 5. 连线逻辑 ---

    # 纵向：骨干网络的信息流
    dot.edge('rgb_in', 'rgb_p3')
    dot.edge('rgb_p3', 'rgb_p4')
    dot.edge('rgb_p4', 'rgb_p5')

    dot.edge('t_in', 't_p3')
    dot.edge('t_p3', 't_p4')
    dot.edge('t_p4', 't_p5')

    # 横向：双流融合 (双向箭头或指向中间)
    # 这里示意：RGB 和 Thermal 都输入给 Fusion 模块
    dot.edge('rgb_p3', 'fusion_p3', dir='forward')  # 指向融合
    dot.edge('t_p3', 'fusion_p3', dir='forward')

    dot.edge('rgb_p4', 'fusion_p4')
    dot.edge('t_p4', 'fusion_p4')

    dot.edge('rgb_p5', 'fusion_p5')
    dot.edge('t_p5', 'fusion_p5')

    # 融合后的特征进入 Head
    # 使用 xlabel 避免连线警告，同时展示不同层级
    dot.edge('fusion_p3', 'yolo_head', xlabel=' P3 Feature ')
    dot.edge('fusion_p4', 'yolo_head', xlabel=' P4 Feature ')
    dot.edge('fusion_p5', 'yolo_head', xlabel=' P5 Feature ')

    # 渲染
    dot.render('rgbt_yolo_simple', view=True, cleanup=True)


if __name__ == '__main__':
    draw_rgbt_yolo_simplified()