from manim import *
import numpy as np


class AdvancedFusion3D(ThreeDScene):
    def construct(self):
        # --- 1. 高级配色方案 ---
        COLOR_RGB = BLUE_D
        COLOR_THM = RED_D
        COLOR_FUSE = GREEN_E
        COLOR_NECK = YELLOW_E
        GLOW_OPACITY = 0.85
        IDLE_OPACITY = 0.15

        # --- 2. 设置 3D 摄像机与赛博朋克网格 ---
        self.set_camera_orientation(phi=65 * DEGREES, theta=-55 * DEGREES, focal_point=[0, -1, 0])
        self.begin_ambient_camera_rotation(rate=0.08)  # 开启缓慢环绕运镜

        # 添加底部科技网格
        grid = NumberPlane(
            x_range=[-8, 8, 1], y_range=[-8, 8, 1],
            background_line_style={"stroke_color": TEAL, "stroke_width": 1, "stroke_opacity": 0.2}
        ).set_z(-1.2)
        self.add(grid)

        # --- 3. 核心节点配置 (X, Y, Z, 宽, 高, 深, 颜色, 标签) ---
        # 尺寸变化体现了 Feature Pyramid 的降采样过程
        nodes_config = {
            "rgb_in": [-3.8, 3.5, 0, 1.8, 1.0, 0.1, COLOR_RGB, "RGB Input"],
            "thm_in": [0.0, 3.5, 0, 1.8, 1.0, 0.1, COLOR_THM, "Thermal Input"],

            "rgb_p3": [-3.8, 1.5, 0, 1.6, 1.2, 0.2, COLOR_RGB, "RGB P3"],
            "thm_p3": [0.0, 1.5, 0, 1.6, 1.2, 0.2, COLOR_THM, "Thermal P3"],
            "fuse_p3": [3.8, 1.5, 0, 1.4, 1.4, 0.3, COLOR_FUSE, "Fusion P3"],  # Fusion 使用圆柱体

            "rgb_p4": [-3.8, -0.5, 0, 1.2, 1.0, 0.4, COLOR_RGB, "RGB P4"],
            "thm_p4": [0.0, -0.5, 0, 1.2, 1.0, 0.4, COLOR_THM, "Thermal P4"],
            "fuse_p4": [3.8, -0.5, 0, 1.2, 1.2, 0.5, COLOR_FUSE, "Fusion P4"],

            "rgb_p5": [-3.8, -2.5, 0, 0.8, 0.8, 0.7, COLOR_RGB, "RGB P5"],
            "thm_p5": [0.0, -2.5, 0, 0.8, 0.8, 0.7, COLOR_THM, "Thermal P5"],
            "fuse_p5": [3.8, -2.5, 0, 0.9, 0.9, 0.8, COLOR_FUSE, "Fusion P5"],

            "neck": [0.0, -4.8, 0, 9.0, 1.2, 0.5, COLOR_NECK, "Feature Pyramid & Detection Heads"]
        }

        nodes = {}

        # --- 4. 生成 3D 实体模型 ---
        for key, (x, y, z, w, h, d, color, text) in nodes_config.items():
            if "fuse" in key:
                # 融合模块使用圆柱体
                obj = Cylinder(radius=w / 2, height=d).move_to([x, y, z])
            else:
                # 其他网络层使用方块
                obj = Prism(dimensions=[w, h, d]).move_to([x, y, z])

            obj.set_fill(color, opacity=IDLE_OPACITY).set_stroke(color, width=2, opacity=0.8)

            # 标签悬浮在上方
            label = Text(text, font_size=18, weight=BOLD).move_to([x, y, z + d / 2 + 0.15])

            nodes[key] = {"obj": obj, "label": label, "center": np.array([x, y, z]), "color": color}
            self.add(obj, label)

        # --- 5. 绘制基础管道 (主板物理连线) ---
        connections = [
            ("rgb_in", "rgb_p3"), ("thm_in", "thm_p3"),
            ("rgb_p3", "rgb_p4"), ("rgb_p3", "fuse_p3"),
            ("thm_p3", "thm_p4"), ("thm_p3", "fuse_p3"),
            ("fuse_p3", "neck"),
            ("rgb_p4", "rgb_p5"), ("rgb_p4", "fuse_p4"),
            ("thm_p4", "thm_p5"), ("thm_p4", "fuse_p4"),
            ("fuse_p4", "neck"),
            ("rgb_p5", "fuse_p5"), ("thm_p5", "fuse_p5"),
            ("fuse_p5", "neck")
        ]

        for src, dst in connections:
            start_pos = nodes[src]["center"]
            end_pos = nodes[dst]["center"]
            # 绘制灰色的 3D 背景管道
            pipe = Line3D(start_pos, end_pos, thickness=0.03, color=GRAY).set_opacity(0.3)
            self.add(pipe)

        # --- 6. 核心逻辑：定义“能量脉冲”流动的动画函数 ---
        def energy_flow(conns):
            sources = set([c[0] for c in conns])
            targets = set([c[1] for c in conns])

            # 1. 源模块充能变亮
            self.play(*[nodes[s]["obj"].animate.set_opacity(GLOW_OPACITY) for s in sources], run_time=0.4)

            # 2. 发射高能光束 (拉高 Z 轴避免穿模)
            lasers = []
            for src, dst, color in conns:
                start = nodes[src]["center"] + np.array([0, 0, 0.1])
                end = nodes[dst]["center"] + np.array([0, 0, 0.1])
                laser = Line(start, end, color=color, stroke_width=8).set_stroke(opacity=0.9)
                self.add(laser)
                lasers.append(laser)

            # 光束生长动画
            self.play(*[Create(laser) for laser in lasers], run_time=0.4)

            # 3. 目标模块被点亮，光束消散，源模块冷却
            self.play(
                *[nodes[t]["obj"].animate.set_opacity(GLOW_OPACITY) for t in targets],
                *[nodes[s]["obj"].animate.set_opacity(IDLE_OPACITY) for s in sources],
                *[laser.animate.set_stroke(opacity=0) for laser in lasers],
                run_time=0.6
            )
            # 清理光束对象
            for laser in lasers:
                self.remove(laser)

        self.wait(2)  # 初始展示 2 秒

        # --- 7. 执行层级执行的流水线动画 ---

        # 阶段 A: 获取输入
        energy_flow([("rgb_in", "rgb_p3", BLUE_B), ("thm_in", "thm_p3", RED_B)])

        # 阶段 B: P3 层分发与融合
        energy_flow([
            ("rgb_p3", "rgb_p4", BLUE_B), ("rgb_p3", "fuse_p3", BLUE_B),
            ("thm_p3", "thm_p4", RED_B), ("thm_p3", "fuse_p3", RED_B)
        ])

        # 阶段 C: P3 特征进入 Neck，同时 P4 层处理
        energy_flow([
            ("fuse_p3", "neck", GREEN_B),
            ("rgb_p4", "rgb_p5", BLUE_B), ("rgb_p4", "fuse_p4", BLUE_B),
            ("thm_p4", "thm_p5", RED_B), ("thm_p4", "fuse_p4", RED_B)
        ])

        # 阶段 D: P4 特征进入 Neck，同时 P5 层处理
        energy_flow([
            ("fuse_p4", "neck", GREEN_B),
            ("rgb_p5", "fuse_p5", BLUE_B), ("thm_p5", "fuse_p5", RED_B)
        ])

        # 阶段 E: P5 最终特征进入 Neck
        energy_flow([("fuse_p5", "neck", GREEN_B)])

        # --- 8. 最终大满贯爆发 (Detection Head) ---
        neck_obj = nodes["neck"]["obj"]
        self.play(neck_obj.animate.set_color(WHITE).set_opacity(1), run_time=0.3)
        self.play(neck_obj.animate.set_color(COLOR_NECK).set_opacity(GLOW_OPACITY), run_time=0.3)

        # 全部冷却，留下高亮底座
        all_keys_except_neck = [k for k in nodes.keys() if k != "neck"]
        self.play(
            *[nodes[k]["obj"].animate.set_opacity(IDLE_OPACITY) for k in all_keys_except_neck],
            run_time=1.5
        )

        self.wait(4)  # 留给观众欣赏 3D 旋转的时间