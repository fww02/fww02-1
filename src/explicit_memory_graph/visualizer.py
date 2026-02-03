"""
Academic-Grade Scene Graph Visualization for 3D-MEM
Dual-Mode Visualizer: Textured Mode + Topology Mode

Features:
- Textured Mode: Habitat 渲染纹理 + 实时位置标注
- Topology Mode: 层级拓扑图（带决策标注）
- 强制 2D 投影（XZ 平面），修复 shapes (2,) (3,) 错误
- 精确坐标对齐（world_to_pixel）
- 无头环境适配（Matplotlib Agg 后端）
- 等比例坐标（axis('equal')）
- 支持外部高质量 top-down map 加载
"""

import os
import logging
from typing import Any, Dict, Optional, List, Tuple, Literal, Union
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制无头渲染，必须在 import pyplot 之前
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    """可视化模式枚举"""
    TEXTURED = "textured"      # 真实纹理俯视图（Habitat 渲染）
    TOPOLOGY = "topology"      # 层级拓扑图（带决策标注）


def load_external_topdown_map(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """
    加载外部高质量 top-down map 图像
    
    支持格式: PNG, JPG, JPEG, BMP, TIFF
    
    Args:
        image_path: 外部图像文件路径
        target_size: 可选的目标尺寸 (width, height)，如果提供则会调整大小
        
    Returns:
        RGB 图像数组 (H, W, 3) uint8，如果加载失败返回 None
    """
    try:
        import cv2
        
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"External top-down map not found: {image_path}")
            return None
        
        # 读取图像 (OpenCV 使用 BGR)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小（如果需要）
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        logger.info(f"Loaded external top-down map: {image_path}, shape={img.shape}")
        return img
        
    except Exception as e:
        logger.warning(f"Failed to load external top-down map: {e}")
        return None


class SceneGraphVisualizer:
    """
    学术级场景图可视化器
    
    双模式独立渲染：
    - Textured Mode: 外部注入 Habitat top_down_map，实时标注位置
    - Topology Mode: 层级拓扑图，带决策冲突标注
    
    支持外部高质量底图：
    - 可通过 load_external_topdown_map() 函数加载外部图像
    - 支持自定义底图边界用于坐标转换
    """

    # ==================== 论文级配色方案 ====================
    # 基于学术论文常用配色
    COLORS = {
        # 节点颜色
        'snapshot': '#3498DB',          # 浅蓝色 - 图像节点 (Snapshot)
        'snapshot_edge': '#2980B9',     # 深蓝色 - 图像节点边框
        'object': '#E74C3C',            # 红色 - 物体节点
        'object_edge': '#C0392B',       # 深红色 - 物体节点边框
        'agent': '#2ECC71',             # 绿色 - 当前位置
        'agent_edge': '#27AE60',        # 深绿色 - 当前位置边框
        'target': '#9B59B6',            # 紫色 - 目标节点
        'target_halo': '#9B59B640',     # 半透明紫色光晕
        
        # 连线颜色
        'trajectory': '#3498DB80',      # 半透明蓝色 - 轨迹线
        'association': '#E74C3C40',     # 半透明红色 - 物体关联线
        
        # 背景/辅助颜色
        'background': '#FAFAFA',        # 浅灰色背景
        'grid': '#E0E0E0',              # 网格线
        'text': '#2C3E50',              # 深灰色文字
        'label_bg': '#00000080',        # 半透明黑色标签背景
    }

    def __init__(
        self, 
        voxel_size: float = 0.1,
        map_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        style: str = "academic",  # "academic" or "default"
    ):
        """
        Args:
            voxel_size: 体素大小（米）
            map_bounds: 地图边界 (min_bound, max_bound)，每个为 (3,) 数组 [x_min, y_min, z_min]
            style: 渲染风格 ("academic" 用于论文, "default" 用于调试)
        """
        self.voxel_size = voxel_size
        self.map_bounds = map_bounds
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # 冲突决策颜色映射（仅 Topology 模式使用）
        self.decision_colors = {
            "MERGE": "#95A5A6",   # 灰色
            "SPLIT": "#E74C3C",   # 红色
            "REPLACE": "#F39C12", # 橙色
            "KEEP": "#F1C40F",    # 黄色
        }
        
        # 设置 matplotlib 全局样式
        if style == "academic":
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                'font.size': 10,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'legend.fontsize': 9,
                'figure.titlesize': 16,
            })

    def set_map_bounds(
        self, 
        min_bound: np.ndarray, 
        max_bound: np.ndarray
    ) -> None:
        """
        设置地图边界（用于 world_to_pixel 坐标转换）
        
        Args:
            min_bound: 最小边界 (3,) [x_min, y_min, z_min]
            max_bound: 最大边界 (3,) [x_max, y_max, z_max]
        """
        self.map_bounds = (np.asarray(min_bound), np.asarray(max_bound))

    def world_to_pixel(
        self,
        world_pos: np.ndarray,
        map_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        将 3D 世界坐标转换为 2D 纹理图像素坐标
        
        数学公式：
            pixel_x = (world_x - x_min) / (x_max - x_min) * map_width
            pixel_y = (world_z - z_min) / (z_max - z_min) * map_height
        
        Args:
            world_pos: 3D 世界坐标 (3,) [x, y, z] 或 2D (2,) [x, z]
            map_shape: 纹理图尺寸 (height, width)
            
        Returns:
            (pixel_x, pixel_y) 像素坐标
        """
        if self.map_bounds is None:
            # 无边界时使用 voxel_size 进行简单转换
            if len(world_pos) == 3:
                px = int(world_pos[0] / self.voxel_size)
                py = int(world_pos[2] / self.voxel_size)
            else:
                px = int(world_pos[0] / self.voxel_size)
                py = int(world_pos[1] / self.voxel_size)
            return (px, py)
        
        min_bound, max_bound = self.map_bounds
        map_height, map_width = map_shape
        
        # 提取 X 和 Z 坐标（Habitat 使用 Y-up 坐标系）
        if len(world_pos) == 3:
            world_x, world_z = world_pos[0], world_pos[2]
        else:
            world_x, world_z = world_pos[0], world_pos[1]
        
        # 归一化到 [0, 1]
        x_range = max_bound[0] - min_bound[0]
        z_range = max_bound[2] - min_bound[2]
        
        if x_range < 1e-6:
            x_range = 1.0
        if z_range < 1e-6:
            z_range = 1.0
        
        norm_x = (world_x - min_bound[0]) / x_range
        norm_z = (world_z - min_bound[2]) / z_range
        
        # 转换为像素坐标
        pixel_x = int(np.clip(norm_x * map_width, 0, map_width - 1))
        pixel_y = int(np.clip(norm_z * map_height, 0, map_height - 1))
        
        return (pixel_x, pixel_y)

    def pixel_to_world(
        self,
        pixel_pos: Tuple[int, int],
        map_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        将 2D 像素坐标转换为 3D 世界坐标（逆变换）
        
        Args:
            pixel_pos: (pixel_x, pixel_y)
            map_shape: (height, width)
            
        Returns:
            (3,) [world_x, 0.0, world_z]
        """
        if self.map_bounds is None:
            world_x = pixel_pos[0] * self.voxel_size
            world_z = pixel_pos[1] * self.voxel_size
            return np.array([world_x, 0.0, world_z])
        
        min_bound, max_bound = self.map_bounds
        map_height, map_width = map_shape
        
        norm_x = pixel_pos[0] / map_width
        norm_z = pixel_pos[1] / map_height
        
        world_x = norm_x * (max_bound[0] - min_bound[0]) + min_bound[0]
        world_z = norm_z * (max_bound[2] - min_bound[2]) + min_bound[2]
        
        return np.array([world_x, 0.0, world_z])

    # ==================== Textured Mode ====================

    def visualize_textured(
        self,
        top_down_map: np.ndarray,
        output_path: str,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        object_positions: Optional[Dict[int, np.ndarray]] = None,
        object_classes: Optional[Dict[int, str]] = None,
        trajectory: Optional[np.ndarray] = None,
        snapshot_positions: Optional[List[np.ndarray]] = None,
        snapshot_connections: Optional[List[Tuple[int, int]]] = None,
        object_to_snapshot: Optional[Dict[int, int]] = None,
        target_object_id: Optional[int] = None,
        title: str = "Scene Graph Visualization",
        show_object_labels: bool = True,
        show_legend: bool = True,
        figsize: Tuple[float, float] = (20, 16),
        dpi: int = 300,
    ):
        """
        Textured Mode: 高质量学术级场景图可视化
        
        适用于论文插图，支持外部高质量底图和精细化渲染控制。
        
        节点样式：
        - 图像节点（Snapshot）: 蓝色实心圆圈，白色边框
        - 物体节点: 红色等边三角形，白色边框
        - 目标节点: 紫色大圆圈 + 半透明光晕
        - 当前位置: 绿色实心圆 + 朝向箭头
        
        连线样式：
        - 轨迹线: 半透明蓝色粗带，带渐变效果
        - 关联线: 半透明红色细线
        
        Args:
            top_down_map: 外部提供的高质量 RGB 俯视图 (H, W, 3)
                          可通过 load_external_topdown_map() 函数加载
            output_path: 输出路径（.png）
            agent_position: 机器人当前 3D 位置 (3,) [x, y, z]
            agent_heading: 机器人朝向角度（弧度）
            object_positions: 物体位置 {obj_id: (3,) position}
            object_classes: 物体类别 {obj_id: class_name}
            trajectory: 机器人轨迹 (N, 3)
            snapshot_positions: Snapshot 位置列表 [(3,), ...]
            snapshot_connections: Snapshot 之间的连接 [(i, j), ...]
            object_to_snapshot: 物体到 Snapshot 的关联 {obj_id: snapshot_idx}
            target_object_id: 目标物体 ID（用于高亮显示）
            title: 图像标题
            show_object_labels: 是否显示物体类别标签（默认 True）
            show_legend: 是否显示图例（默认 True）
            figsize: 图像尺寸 (width, height) in inches
            dpi: 输出 DPI（默认 300）
        """
        if top_down_map is None or top_down_map.size == 0:
            self.logger.warning("No top_down_map provided for textured mode")
            return

        # 使用类属性中定义的颜色（如果在 academic 模式下）
        if self.style == "academic":
            COLOR_SNAPSHOT = self.COLORS['snapshot']
            COLOR_SNAPSHOT_EDGE = self.COLORS['snapshot_edge']
            COLOR_TRAJECTORY = self.COLORS['trajectory']
            COLOR_OBJECT = self.COLORS['object']
            COLOR_OBJECT_EDGE = self.COLORS['object_edge']
            COLOR_ASSOCIATION = self.COLORS['association']
            COLOR_AGENT = self.COLORS['agent']
            COLOR_AGENT_EDGE = self.COLORS['agent_edge']
            COLOR_TARGET = self.COLORS['target']
            COLOR_TARGET_HALO = self.COLORS['target_halo']
        else:
            # 默认颜色
            COLOR_SNAPSHOT = '#4DA6FF'
            COLOR_SNAPSHOT_EDGE = '#2980B9'
            COLOR_TRAJECTORY = '#3498DB80'
            COLOR_OBJECT = '#E74C3C'
            COLOR_OBJECT_EDGE = '#C0392B'
            COLOR_ASSOCIATION = '#E74C3C40'
            COLOR_AGENT = '#2ECC71'
            COLOR_AGENT_EDGE = '#27AE60'
            COLOR_TARGET = '#9B59B6'
            COLOR_TARGET_HALO = '#9B59B640'

        # 高清输出设置
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 1) 绘制高质量底图
        ax.imshow(top_down_map, origin="upper", interpolation='bilinear')
        map_shape = top_down_map.shape[:2]  # (H, W)
        
        # 计算合适的节点大小（基于地图尺寸）
        map_diag = np.sqrt(map_shape[0]**2 + map_shape[1]**2)
        base_radius = max(8, map_diag / 100)  # 动态调整节点大小
        
        # 2) 绘制轨迹连线（作为底层）
        if snapshot_positions is not None and len(snapshot_positions) > 1:
            snapshot_pixels = []
            for pos in snapshot_positions:
                px, py = self.world_to_pixel(pos, map_shape)
                snapshot_pixels.append((px, py))
            snapshot_pixels = np.array(snapshot_pixels)
            
            # 使用 LineCollection 绘制带渐变的轨迹
            if snapshot_connections is not None:
                segments = []
                for (i, j) in snapshot_connections:
                    if i < len(snapshot_pixels) and j < len(snapshot_pixels):
                        segments.append([snapshot_pixels[i], snapshot_pixels[j]])
                if segments:
                    lc = LineCollection(
                        segments,
                        colors=COLOR_TRAJECTORY,
                        linewidths=base_radius * 0.8,
                        capstyle='round',
                        joinstyle='round',
                        zorder=2
                    )
                    ax.add_collection(lc)
            else:
                # 默认顺序连接（渐变效果）
                segments = [[snapshot_pixels[i], snapshot_pixels[i+1]] 
                           for i in range(len(snapshot_pixels) - 1)]
                alphas = np.linspace(0.3, 0.8, len(segments))
                colors = [(*matplotlib.colors.to_rgb(COLOR_SNAPSHOT), a) for a in alphas]
                lc = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=base_radius * 0.8,
                    capstyle='round',
                    joinstyle='round',
                    zorder=2
                )
                ax.add_collection(lc)

        # 3) 绘制物体关联线（从 Snapshot 到物体）
        if object_positions and object_to_snapshot and snapshot_positions:
            for obj_id, snapshot_idx in object_to_snapshot.items():
                if obj_id in object_positions and snapshot_idx < len(snapshot_positions):
                    obj_pos = object_positions[obj_id]
                    snap_pos = snapshot_positions[snapshot_idx]
                    
                    obj_px, obj_py = self.world_to_pixel(obj_pos, map_shape)
                    snap_px, snap_py = self.world_to_pixel(snap_pos, map_shape)
                    
                    ax.plot(
                        [snap_px, obj_px], [snap_py, obj_py],
                        color=COLOR_ASSOCIATION,
                        linewidth=1.5,
                        linestyle='--',
                        alpha=0.6,
                        zorder=3
                    )

        # 4) 绘制 Snapshot 图像节点（蓝色实心圆圈）
        if snapshot_positions is not None:
            for i, pos in enumerate(snapshot_positions):
                px, py = self.world_to_pixel(pos, map_shape)
                circle = plt.Circle(
                    (px, py), 
                    radius=base_radius,
                    facecolor=COLOR_SNAPSHOT,
                    edgecolor=COLOR_SNAPSHOT_EDGE,
                    linewidth=2,
                    alpha=0.9,
                    zorder=10
                )
                ax.add_patch(circle)

        # 5) 绘制物体节点（等边三角形）
        # 过滤掉背景类
        background_classes = {'wall', 'floor', 'ceiling', 'paneling', 'banner', 'misc'}
        
        if object_positions:
            for obj_id, pos in object_positions.items():
                # 跳过目标物体（单独绘制）
                if target_object_id is not None and obj_id == target_object_id:
                    continue
                
                # 过滤背景类
                class_name = object_classes.get(obj_id, '') if object_classes else ''
                if class_name.lower() in background_classes:
                    continue
                
                px, py = self.world_to_pixel(pos, map_shape)
                
                # 等边三角形（顶点朝上）
                triangle_size = base_radius * 0.9
                triangle = plt.Polygon(
                    [
                        (px, py - triangle_size),  # 顶点
                        (px - triangle_size * 0.866, py + triangle_size * 0.5),  # 左下
                        (px + triangle_size * 0.866, py + triangle_size * 0.5),  # 右下
                    ],
                    facecolor=COLOR_OBJECT,
                    edgecolor=COLOR_OBJECT_EDGE,
                    linewidth=1.5,
                    alpha=0.85,
                    zorder=15
                )
                ax.add_patch(triangle)
                
                # 添加类别标签（如果需要）
                if show_object_labels and class_name:
                    ax.text(
                        px, py + triangle_size + base_radius * 0.6,
                        class_name,
                        fontsize=max(5, base_radius * 0.5),
                        color='white',
                        fontweight='bold',
                        ha='center',
                        va='top',
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='black',
                            alpha=0.6,
                            edgecolor='none'
                        ),
                        zorder=16
                    )

        # 6) 绘制目标节点（紫色大圆圈 + 光晕）
        if target_object_id is not None and object_positions and target_object_id in object_positions:
            pos = object_positions[target_object_id]
            px, py = self.world_to_pixel(pos, map_shape)
            
            # 半透明光晕（多层渐变效果）
            for r_mult, alpha in [(3.0, 0.15), (2.0, 0.25), (1.5, 0.35)]:
                halo = plt.Circle(
                    (px, py),
                    radius=base_radius * r_mult,
                    facecolor=COLOR_TARGET,
                    edgecolor='none',
                    alpha=alpha,
                    zorder=17
                )
                ax.add_patch(halo)
            
            # 目标圆圈
            target_circle = plt.Circle(
                (px, py),
                radius=base_radius * 1.2,
                facecolor=COLOR_TARGET,
                edgecolor='white',
                linewidth=3,
                alpha=0.95,
                zorder=19
            )
            ax.add_patch(target_circle)
            
            # 目标标签
            if object_classes and target_object_id in object_classes:
                ax.text(
                    px, py - base_radius * 2.5,
                    f"Target: {object_classes[target_object_id]}",
                    fontsize=max(7, base_radius * 0.6),
                    color='white',
                    fontweight='bold',
                    ha='center',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=COLOR_TARGET,
                        alpha=0.9,
                        edgecolor='white',
                        linewidth=1.5
                    ),
                    zorder=20
                )

        # 7) 绘制机器人轨迹（如果没有 Snapshot，使用原始轨迹）
        if trajectory is not None and len(trajectory) > 0 and snapshot_positions is None:
            traj_pixels = []
            for pos in trajectory:
                px, py = self.world_to_pixel(pos, map_shape)
                traj_pixels.append((px, py))
            traj_pixels = np.array(traj_pixels)
            ax.plot(
                traj_pixels[:, 0], traj_pixels[:, 1],
                color=COLOR_TRAJECTORY,
                linewidth=base_radius * 0.6,
                alpha=0.7,
                solid_capstyle='round',
                zorder=2
            )

        # 8) 绘制当前位置（绿色实心圆）
        if agent_position is not None:
            px, py = self.world_to_pixel(agent_position, map_shape)
            
            # 绿色圆圈
            agent_circle = plt.Circle(
                (px, py),
                radius=base_radius * 1.2,
                facecolor=COLOR_AGENT,
                edgecolor=COLOR_AGENT_EDGE,
                linewidth=2.5,
                alpha=0.95,
                zorder=25
            )
            ax.add_patch(agent_circle)
            
            # 朝向箭头
            if agent_heading is not None:
                arrow_len = base_radius * 2.5
                dx = arrow_len * np.cos(agent_heading)
                dy = arrow_len * np.sin(agent_heading)
                ax.arrow(
                    px, py, dx, dy,
                    head_width=base_radius * 0.8,
                    head_length=base_radius * 0.5,
                    fc='#F1C40F',  # 黄色箭头
                    ec='#34495E',   # 深灰边框
                    linewidth=1.5,
                    zorder=26
                )

        # 9) 设置图像属性
        ax.set_xlim(0, map_shape[1])
        ax.set_ylim(map_shape[0], 0)  # 翻转 Y 轴
        ax.set_aspect('equal')  # 强制等比例
        ax.axis('off')  # 隐藏坐标轴
        
        # 10) 添加图例（如果需要）
        if show_legend:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SNAPSHOT,
                           markersize=12, label='Image Node', markeredgecolor=COLOR_SNAPSHOT_EDGE, markeredgewidth=1.5),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLOR_OBJECT,
                           markersize=12, label='Object Node', markeredgecolor=COLOR_OBJECT_EDGE, markeredgewidth=1.5),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_AGENT,
                           markersize=12, label='Agent', markeredgecolor=COLOR_AGENT_EDGE, markeredgewidth=1.5),
                plt.Line2D([0], [0], color=COLOR_SNAPSHOT, linewidth=6, alpha=0.6, label='Trajectory'),
            ]
            if target_object_id is not None:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_TARGET,
                               markersize=14, label='Target', markeredgecolor='white', markeredgewidth=2)
                )
            
            ax.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=max(8, base_radius * 0.8),
                framealpha=0.9,
                fancybox=True,
                shadow=True,
                frameon=True,
                edgecolor='#CCCCCC'
            )

        # 11) 保存高清图像
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout(pad=0.5)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        self.logger.info(f"Saved high-quality textured map to: {output_path}")

    # ==================== Topology Mode (原有功能) ====================

    def visualize_hierarchical_graph(
        self,
        regions: Dict[int, Any],
        scene_objects: Dict[int, Dict[str, Any]],
        obj_to_region: Dict[int, int],
        floor_id: str,
        output_path: str,
        decision_history: Optional[Dict[int, List[str]]] = None,
        bg_image: Optional[np.ndarray] = None,
    ):
        """
        Topology Mode: 语义拓扑图（2D 俯视图）
        
        特点：
        - 可选 Habitat top_down_map 作为背景
        - 绘制房间→物体虚线连线
        - 根据 decision_history 高亮冲突节点
        - 强制 axis('equal') 等比例坐标
        
        Args:
            regions: 区域字典 {region_id: RegionNode}
            scene_objects: 物体字典 {obj_id: {class_name, pcd, bbox, ...}}
            obj_to_region: 物体到区域的映射 {obj_id: region_id}
            floor_id: 楼层标识
            output_path: 输出路径（.png）
            decision_history: 决策历史 {obj_id: [decision1, decision2, ...]}
            bg_image: 可选的背景图像 (H, W, 3) RGB，从 Habitat 仿真器获取的 top_down_map
        """
        if not scene_objects:
            self.logger.warning("No objects to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        
        # 如果提供了背景图，先绘制背景
        if bg_image is not None and bg_image.size > 0:
            ax.imshow(bg_image, origin='upper', alpha=0.6, interpolation='bilinear')
            # 使用图像坐标系统
            use_pixel_coords = True
            map_shape = bg_image.shape[:2]
        else:
            use_pixel_coords = False
            map_shape = None
        
        # 计算中心点（2D 投影：XZ 平面）
        room_centroids_2d = self._compute_room_centroids_2d(regions, floor_id)
        obj_centroids_2d = self._compute_object_centroids_2d(scene_objects)
        
        if not room_centroids_2d:
            self.logger.warning("No room centroids computed")
            plt.close(fig)
            return
        
        # 如果使用像素坐标，转换所有坐标
        if use_pixel_coords and map_shape is not None:
            room_centroids_2d = {
                k: self.world_to_pixel(np.array([v[0], 0, v[1]]), map_shape)
                for k, v in room_centroids_2d.items()
            }
            obj_centroids_2d = {
                k: self.world_to_pixel(np.array([v[0], 0, v[1]]), map_shape)
                for k, v in obj_centroids_2d.items()
            }
        
        # 1) 绘制房间节点（蓝色圆圈）
        node_radius = 15 if use_pixel_coords else 0.3
        for room_id_str, centroid_2d in room_centroids_2d.items():
            circle = Circle(
                centroid_2d, 
                radius=node_radius, 
                facecolor='blue', 
                edgecolor='darkblue',
                alpha=0.6,
                linewidth=2,
                zorder=10
            )
            ax.add_patch(circle)
            text_offset = node_radius * 1.5 if use_pixel_coords else 0.5
            ax.text(
                centroid_2d[0], centroid_2d[1] + text_offset, 
                f"R{room_id_str.split('_')[-1]}", 
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='darkblue'
            )
        
        # 2) 绘制物体节点 + 房间→物体连线
        for obj_id, obj in scene_objects.items():
            rid = obj_to_region.get(obj_id)
            if rid is None:
                continue
            
            room_id_str = f"{floor_id}_{rid}"
            if room_id_str not in room_centroids_2d:
                continue
            
            obj_name = obj.get("class_name", "unknown")
            # 过滤墙、地板、天花板
            if any(
                substring in obj_name.lower()
                for substring in ["wall", "floor", "ceiling", "paneling", "banner"]
            ):
                continue
            
            obj_centroid_2d = obj_centroids_2d.get(str(obj_id))
            if obj_centroid_2d is None:
                continue
            
            # 判断是否为冲突节点
            obj_color, obj_marker = self._get_object_style(obj_id, decision_history)
            
            # 绘制房间→物体连线（虚线）
            room_center = room_centroids_2d[room_id_str]
            ax.plot(
                [room_center[0], obj_centroid_2d[0]],
                [room_center[1], obj_centroid_2d[1]],
                color='gray',
                linestyle='--',
                linewidth=1.0,
                alpha=0.5,
                zorder=1
            )
            
            # 绘制物体节点
            ax.scatter(
                obj_centroid_2d[0], 
                obj_centroid_2d[1],
                c=obj_color,
                marker=obj_marker,
                s=80,
                edgecolors='black',
                linewidths=1.5,
                alpha=0.8,
                zorder=5
            )
            
            # 标注物体名称
            ax.text(
                obj_centroid_2d[0], 
                obj_centroid_2d[1] - 0.3,
                obj_name[:10],
                ha='center', va='top',
                fontsize=6,
                color=obj_color
            )
        
        # 强制等比例坐标轴
        ax.axis('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(
            f'Hierarchical Scene Graph - {floor_id}\n'
            f'(Objects: {len(scene_objects)}, Rooms: {len(room_centroids_2d)})',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 添加图例
        self._add_legend(ax)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved hierarchical graph to: {output_path}")

    def draw_decision_heatmap(
        self,
        decision_events: List[Tuple[int, str, np.ndarray, Optional[np.ndarray]]],
        scene_objects: Dict[int, Dict[str, Any]],
        output_path: str,
        trajectory: Optional[np.ndarray] = None,
    ):
        """
        决策热力图（离散事件点）
        
        Args:
            decision_events: [(obj_id, decision, obj_pos_3d, robot_pos_3d), ...]
            scene_objects: 物体字典
            output_path: 输出路径
            trajectory: 机器人轨迹 (N, 3)
        """
        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
        
        # 1) 绘制物体位置（灰色背景点）
        for obj_id, obj in scene_objects.items():
            centroid_2d = self._get_object_centroid_2d(obj)
            if centroid_2d is not None:
                ax.scatter(
                    centroid_2d[0], centroid_2d[1],
                    c='lightgray',
                    marker='o',
                    s=30,
                    alpha=0.3,
                    zorder=1
                )
        
        # 2) 绘制机器人轨迹
        if trajectory is not None and len(trajectory) > 0:
            traj_2d = trajectory[:, [0, 2]]  # XZ 平面投影
            ax.plot(
                traj_2d[:, 0], traj_2d[:, 1],
                color='black',
                linestyle='-',
                linewidth=1.5,
                alpha=0.4,
                label='Robot Trajectory',
                zorder=2
            )
        
        # 3) 绘制决策事件点
        decision_counts = defaultdict(int)
        
        for obj_id, decision, obj_pos_3d, robot_pos_3d in decision_events:
            if decision == "KEEP":
                if robot_pos_3d is not None:
                    pos_2d = robot_pos_3d[[0, 2]]
                else:
                    pos_2d = obj_pos_3d[[0, 2]]
                marker, color, size = 'x', 'red', 150
            elif decision == "REPLACE":
                pos_2d = obj_pos_3d[[0, 2]]
                marker, color, size = '^', 'yellow', 120
            elif decision == "SPLIT":
                pos_2d = obj_pos_3d[[0, 2]]
                marker, color, size = 's', 'orange', 100
            else:
                continue
            
            ax.scatter(
                pos_2d[0], pos_2d[1],
                c=color,
                marker=marker,
                s=size,
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9,
                zorder=10
            )
            decision_counts[decision] += 1
        
        # 设置图像属性
        ax.axis('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(
            f'Decision Heatmap\n'
            f'KEEP: {decision_counts["KEEP"]}, REPLACE: {decision_counts["REPLACE"]}, SPLIT: {decision_counts["SPLIT"]}',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 图例
        legend_elements = [
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                   markersize=10, label='KEEP', markeredgecolor='black'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', 
                   markersize=10, label='REPLACE', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                   markersize=8, label='SPLIT', markeredgecolor='black'),
        ]
        if trajectory is not None:
            legend_elements.append(
                Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Trajectory')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved decision heatmap to: {output_path}")

    def visualize_conflict_statistics(
        self,
        decision_history: Dict[int, List[str]],
        output_path: str,
    ):
        """绘制冲突决策统计图（柱状图 + 饼图）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        
        # 统计决策类型
        decision_counts = defaultdict(int)
        for obj_id, decisions in decision_history.items():
            for decision in decisions:
                decision_counts[decision] += 1
        
        if not decision_counts:
            decision_counts["NONE"] = 0
        
        # 1) 柱状图
        decisions = list(decision_counts.keys())
        counts = list(decision_counts.values())
        colors = [self.decision_colors.get(d, 'gray') for d in decisions]
        
        ax1.bar(decisions, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Decision Type Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        
        max_count = max(counts) if counts else 1
        for i, (d, c) in enumerate(zip(decisions, counts)):
            ax1.text(i, c + max_count * 0.02, str(c), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2) 饼图
        conflict_objects = sum(1 for decisions in decision_history.values() 
                              if any(d in ["SPLIT", "REPLACE", "KEEP"] for d in decisions))
        normal_objects = len(decision_history) - conflict_objects
        
        if conflict_objects + normal_objects == 0:
            conflict_objects = 0
            normal_objects = 1
        
        ax2.pie(
            [conflict_objects, normal_objects],
            labels=['Conflict', 'Normal'],
            colors=['red', 'lightgray'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax2.set_title('Object Conflict Ratio', fontsize=14, fontweight='bold')
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved conflict statistics to: {output_path}")

    # ==================== 统一接口 ====================

    def visualize(
        self,
        mode: VisualizationMode,
        output_path: str,
        *,
        # Abstract mode params
        regions: Optional[Dict[int, Any]] = None,
        # Textured mode params
        top_down_map: Optional[np.ndarray] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        object_positions: Optional[Dict[int, np.ndarray]] = None,
        trajectory: Optional[np.ndarray] = None,
        # Topology mode params
        scene_objects: Optional[Dict[int, Dict[str, Any]]] = None,
        obj_to_region: Optional[Dict[int, int]] = None,
        decision_history: Optional[Dict[int, List[str]]] = None,
        floor_id: str = "0",
        title: str = "",
    ):
        """
        统一可视化接口
        
        Args:
            mode: 可视化模式 (ABSTRACT / TEXTURED / TOPOLOGY)
            output_path: 输出路径
            其他参数根据模式选择性提供
        """
        if mode == VisualizationMode.TEXTURED:
            self.visualize_textured(
                top_down_map=top_down_map,
                output_path=output_path,
                agent_position=agent_position,
                agent_heading=agent_heading,
                object_positions=object_positions,
                trajectory=trajectory,
                title=title or "Textured Top-Down View",
            )
        
        elif mode == VisualizationMode.TOPOLOGY:
            self.visualize_hierarchical_graph(
                regions=regions or {},
                scene_objects=scene_objects or {},
                obj_to_region=obj_to_region or {},
                floor_id=floor_id,
                output_path=output_path,
                decision_history=decision_history,
            )
        
        else:
            self.logger.error(f"Unknown visualization mode: {mode}")

    # ==================== 辅助方法 ====================

    def _compute_room_centroids_2d(
        self, regions: Dict[int, Any], floor_id: str
    ) -> Dict[str, np.ndarray]:
        """计算房间中心点（2D 投影 XZ 平面）"""
        centroids = {}
        for rid, rn in regions.items():
            if rn.mask is not None:
                coords = np.argwhere(rn.mask)
                if len(coords) > 0:
                    coords_world = coords * self.voxel_size
                    center_3d = coords_world.mean(axis=0)
                    center_2d = center_3d[[0, 2]] if len(center_3d) >= 3 else center_3d[:2]
                    centroids[f"{floor_id}_{rid}"] = center_2d
        return centroids

    def _compute_object_centroids_2d(
        self, scene_objects: Dict[int, Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """计算物体中心点（2D 投影）"""
        centroids = {}
        for obj_id, obj in scene_objects.items():
            centroid_2d = self._get_object_centroid_2d(obj)
            if centroid_2d is not None:
                centroids[str(obj_id)] = centroid_2d
        return centroids

    def _get_object_centroid_2d(self, obj: Dict[str, Any]) -> Optional[np.ndarray]:
        """获取单个物体的 2D 中心点（XZ 平面）"""
        pcd = obj.get("pcd")
        if pcd is not None and hasattr(pcd, "points") and len(pcd.points) > 0:
            center_3d = np.asarray(pcd.points).mean(axis=0)
            return center_3d[[0, 2]]
        
        bbox = obj.get("bbox")
        if bbox is not None:
            if hasattr(bbox, "center"):
                center_3d = np.asarray(bbox.center)
            elif hasattr(bbox, "get_center"):
                center_3d = np.asarray(bbox.get_center())
            else:
                return None
            return center_3d[[0, 2]]
        
        return None

    def _get_object_style(
        self, 
        obj_id: int, 
        decision_history: Optional[Dict[int, List[str]]]
    ) -> Tuple[str, str]:
        """根据决策历史确定物体的颜色和标记"""
        if decision_history is None or obj_id not in decision_history:
            return 'gray', 'o'
        
        decisions = decision_history[obj_id]
        if "SPLIT" in decisions:
            return 'red', 's'
        elif "REPLACE" in decisions:
            return 'orange', '^'
        elif "KEEP" in decisions:
            return 'yellow', 'D'
        return 'gray', 'o'

    def _add_legend(self, ax):
        """添加图例"""
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Normal Object'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markersize=8, label='SPLIT Conflict'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', 
                   markersize=8, label='REPLACE Conflict'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow', 
                   markersize=8, label='KEEP Conflict'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Room Center'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
