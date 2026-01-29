"""
3D Scene Graph Visualization (HOV-SG style - exact replica)

Uses PyVista (VTK-based) for interactive 3D visualization, matching HOV-SG's style:
  - Floor nodes: orange spheres (radius=0.5)
  - Room nodes: blue spheres (radius=0.25)
  - Object nodes: colored point clouds (random colors)
  - Edges: lines connecting floor→room→object (floor-room: white, line_width=4; room-object: gray, line_width=1.5)
  - Filters: excludes wall/floor/ceiling/paneling/banner/overhang objects
  - Minimum points: objects must have >100 points
"""

import os
from typing import Any, Dict, Optional
from collections import defaultdict
import logging

import numpy as np

# Configure PyVista for headless/off-screen rendering (fix GLX BadAccess error)
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTKLINK'] = '0'
# Disable OpenGL hardware acceleration, use software rendering
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'

try:
    import pyvista as pv
    # Force OSMesa or software rendering backend
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()  # Try to start virtual framebuffer if available
    except Exception:
        pass  # If Xvfb not available, continue with software rendering
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class SceneGraphVisualizer:
    """Generate 3D scene graph visualizations exactly like HOV-SG (using PyVista)."""

    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = voxel_size
        self.logger = logging.getLogger(__name__)

    def visualize_hierarchical_graph(
        self,
        regions: Dict[int, Any],
        scene_objects: Dict[int, Dict[str, Any]],
        obj_to_region: Dict[int, int],
        floor_id: str,
        output_path: str,
    ):
        """Generate a 3D hierarchical scene graph visualization (HOV-SG replica).

        This function produces a static screenshot of the 3D scene graph, matching the
        visual style of HOV-SG's visualize_graph.py output.

        Args:
            regions: region_id -> RegionNode
            scene_objects: obj_id -> object dict with keys: class_name, pcd, bbox, etc.
            obj_to_region: obj_id -> region_id mapping
            floor_id: floor identifier (e.g., "floor_0")
            output_path: where to save the visualization screenshot (e.g., "graph_viz.png")
        """
        if not PYVISTA_AVAILABLE or not OPEN3D_AVAILABLE:
            self.logger.warning("PyVista or Open3D not available, skipping 3D visualization")
            return

        if not scene_objects:
            self.logger.warning("No objects to visualize")
            return

        # Initialize PyVista plotter (off_screen for saving screenshots)
        p = pv.Plotter(off_screen=True, window_size=[1600, 1200])

        # Build hierarchy topology
        hier_topo = defaultdict(dict)
        hier_topo[floor_id] = {}
        for rid in regions.keys():
            hier_topo[floor_id][f"{floor_id}_{rid}"] = []

        for obj_id, obj in scene_objects.items():
            rid = obj_to_region.get(obj_id)
            if rid is not None:
                room_id_str = f"{floor_id}_{rid}"
                hier_topo[floor_id][room_id_str].append(str(obj_id))

        # Compute centroids
        floor_centroid = self._compute_floor_centroid(regions)
        room_centroids = self._compute_room_centroids(regions, floor_id)
        obj_centroids = self._compute_object_centroids(scene_objects)

        # Visualization offsets (HOV-SG style: floors stacked vertically with offset [7.0, 2.5, 4.0])
        # For single floor, use zero offset
        viz_offset = np.array([0.0, 0.0, 0.0])
        floor_centroid_viz = floor_centroid + viz_offset + np.array([0.0, 4.0, 0.0])

        # 1) Visualize floor node (orange sphere, radius=0.5)
        p.add_mesh(pv.Sphere(center=tuple(floor_centroid_viz), radius=0.5), color="orange")

        # 2) Visualize room nodes (blue spheres, radius=0.25) + floor→room edges
        room_centroids_viz = {}
        for room_id_str, room_centroid in room_centroids.items():
            room_centroid_viz = room_centroid + viz_offset + np.array([0.0, 3.5, 0.0])
            room_centroids_viz[room_id_str] = room_centroid_viz
            p.add_mesh(pv.Sphere(center=tuple(room_centroid_viz), radius=0.25), color="blue")
            # floor→room edge (white, line_width=4)
            p.add_mesh(
                pv.Line(tuple(floor_centroid_viz), tuple(room_centroid_viz)),
                line_width=4,
                color="white",
            )

        # 3) Visualize object nodes (point clouds + room→object edges)
        for obj_id, obj in scene_objects.items():
            rid = obj_to_region.get(obj_id)
            if rid is None:
                continue  # object not assigned to any region
            room_id_str = f"{floor_id}_{rid}"
            if room_id_str not in room_centroids_viz:
                continue

            obj_name = obj.get("class_name", "unknown")
            # Filter out wall/floor/ceiling (like HOV-SG does)
            if any(
                substring in obj_name.lower()
                for substring in ["wall", "floor", "ceiling", "paneling", "banner", "overhang"]
            ):
                continue

            pcd = obj.get("pcd")
            if pcd is None or not hasattr(pcd, "points") or len(pcd.points) < 100:
                continue

            # Apply offset to point cloud
            cloud_xyz = np.asarray(pcd.points) + viz_offset
            obj_centroid_viz = obj_centroids.get(str(obj_id))
            if obj_centroid_viz is None:
                obj_centroid_viz = cloud_xyz.mean(axis=0)
            else:
                obj_centroid_viz = obj_centroid_viz + viz_offset

            # room→object edge (gray, line_width=1.5, opacity=0.5)
            p.add_mesh(
                pv.Line(tuple(room_centroids_viz[room_id_str]), tuple(obj_centroid_viz)),
                line_width=1.5,
                opacity=0.5,
                color="gray",
            )

            # Object point cloud (random color like HOV-SG)
            random_color = np.random.rand(3)
            cloud = pv.PolyData(cloud_xyz)
            colors_array = np.tile(random_color, (cloud_xyz.shape[0], 1))
            p.add_mesh(
                cloud,
                scalars=colors_array,
                rgb=True,
                point_size=5,
                show_vertices=True,
            )
            self.logger.info(f"Included object of category: {obj_name} (id={obj_id})")

        # Set camera and save screenshot
        p.camera_position = "iso"
        p.screenshot(output_path, transparent_background=False)
        p.close()
        self.logger.info(f"Saved 3D scene graph visualization to: {output_path}")

    def generate_graph_visualization_video(
        self,
        regions: Dict[int, Any],
        scene_objects: Dict[int, Dict[str, Any]],
        obj_to_region: Dict[int, int],
        floor_id: str,
        output_video_path: str,
        n_frames: int = 60,
        fps: int = 10,
    ):
        """Generate a rotating 3D scene graph video (HOV-SG style).

        This function produces a rotating animation of the 3D scene graph, matching the
        visual style of HOV-SG's demo GIFs/videos.

        Args:
            regions: region_id -> RegionNode
            scene_objects: obj_id -> object dict
            obj_to_region: obj_id -> region_id mapping
            floor_id: floor identifier
            output_video_path: where to save the video (e.g., "graph_viz.gif" or "graph_viz.mp4")
            n_frames: number of rotation frames (default: 60)
            fps: frames per second (default: 10)
        """
        if not PYVISTA_AVAILABLE or not OPEN3D_AVAILABLE:
            self.logger.warning("PyVista or Open3D not available, skipping video generation")
            return

        if not scene_objects:
            self.logger.warning("No objects to visualize")
            return

        # Initialize PyVista plotter (off_screen)
        p = pv.Plotter(off_screen=True, window_size=[1600, 1200])

        # Build hierarchy topology (same as visualize_hierarchical_graph)
        hier_topo = defaultdict(dict)
        hier_topo[floor_id] = {}
        for rid in regions.keys():
            hier_topo[floor_id][f"{floor_id}_{rid}"] = []
        for obj_id, obj in scene_objects.items():
            rid = obj_to_region.get(obj_id)
            if rid is not None:
                room_id_str = f"{floor_id}_{rid}"
                hier_topo[floor_id][room_id_str].append(str(obj_id))

        # Compute centroids
        floor_centroid = self._compute_floor_centroid(regions)
        room_centroids = self._compute_room_centroids(regions, floor_id)
        obj_centroids = self._compute_object_centroids(scene_objects)

        viz_offset = np.array([0.0, 0.0, 0.0])
        floor_centroid_viz = floor_centroid + viz_offset + np.array([0.0, 4.0, 0.0])

        # Add meshes (floor, rooms, objects)
        p.add_mesh(pv.Sphere(center=tuple(floor_centroid_viz), radius=0.5), color="orange")

        room_centroids_viz = {}
        for room_id_str, room_centroid in room_centroids.items():
            room_centroid_viz = room_centroid + viz_offset + np.array([0.0, 3.5, 0.0])
            room_centroids_viz[room_id_str] = room_centroid_viz
            p.add_mesh(pv.Sphere(center=tuple(room_centroid_viz), radius=0.25), color="blue")
            p.add_mesh(
                pv.Line(tuple(floor_centroid_viz), tuple(room_centroid_viz)),
                line_width=4,
                color="white",
            )

        for obj_id, obj in scene_objects.items():
            rid = obj_to_region.get(obj_id)
            if rid is None:
                continue
            room_id_str = f"{floor_id}_{rid}"
            if room_id_str not in room_centroids_viz:
                continue
            obj_name = obj.get("class_name", "unknown")
            if any(
                substring in obj_name.lower()
                for substring in ["wall", "floor", "ceiling", "paneling", "banner", "overhang"]
            ):
                continue
            pcd = obj.get("pcd")
            if pcd is None or not hasattr(pcd, "points") or len(pcd.points) < 100:
                continue
            cloud_xyz = np.asarray(pcd.points) + viz_offset
            obj_centroid_viz = obj_centroids.get(str(obj_id))
            if obj_centroid_viz is None:
                obj_centroid_viz = cloud_xyz.mean(axis=0)
            else:
                obj_centroid_viz = obj_centroid_viz + viz_offset
            p.add_mesh(
                pv.Line(tuple(room_centroids_viz[room_id_str]), tuple(obj_centroid_viz)),
                line_width=1.5,
                opacity=0.5,
                color="gray",
            )
            random_color = np.random.rand(3)
            cloud = pv.PolyData(cloud_xyz)
            colors_array = np.tile(random_color, (cloud_xyz.shape[0], 1))
            p.add_mesh(cloud, scalars=colors_array, rgb=True, point_size=5, show_vertices=True)

        # Generate rotating frames
        frames = []
        for frame_idx in range(n_frames):
            azimuth = frame_idx * 360 / n_frames
            p.camera_position = "iso"
            p.camera.azimuth = azimuth
            p.camera.elevation = 20

            # Render frame
            img = p.screenshot(return_img=True, transparent_background=False)
            frames.append(img)

        p.close()

        # Save video/gif
        try:
            import imageio
            if output_video_path.endswith(".gif"):
                imageio.mimsave(output_video_path, frames, fps=fps, loop=0)
            else:
                # mp4
                imageio.mimsave(output_video_path, frames, fps=fps, codec="libx264")
            self.logger.info(f"Saved 3D scene graph video to: {output_video_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save video: {e}")
            # Fallback: save frames
            frame_dir = output_video_path.replace(".mp4", "_frames").replace(".gif", "_frames")
            os.makedirs(frame_dir, exist_ok=True)
            try:
                import matplotlib.pyplot as plt
                for i, frame in enumerate(frames):
                    plt.imsave(os.path.join(frame_dir, f"{i:04d}.png"), frame)
                self.logger.info(f"Saved {len(frames)} frames to: {frame_dir}")
            except Exception:
                pass

    def _compute_floor_centroid(self, regions: Dict[int, Any]) -> np.ndarray:
        """Compute floor centroid from all region masks."""
        all_coords = []
        for rn in regions.values():
            if rn.mask is not None:
                coords = np.argwhere(rn.mask)
                if len(coords) > 0:
                    # Convert grid coords to world coords (rough approximation)
                    coords_world = coords * self.voxel_size
                    all_coords.append(coords_world)
        if not all_coords:
            return np.array([0.0, 0.0, 0.0])
        all_coords = np.vstack(all_coords)
        return all_coords.mean(axis=0)

    def _compute_room_centroids(
        self, regions: Dict[int, Any], floor_id: str
    ) -> Dict[str, np.ndarray]:
        """Compute room centroids from region masks."""
        centroids = {}
        for rid, rn in regions.items():
            if rn.mask is not None:
                coords = np.argwhere(rn.mask)
                if len(coords) > 0:
                    coords_world = coords * self.voxel_size
                    centroids[f"{floor_id}_{rid}"] = coords_world.mean(axis=0)
        return centroids

    def _compute_object_centroids(
        self, scene_objects: Dict[int, Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute object centroids from point clouds."""
        centroids = {}
        for obj_id, obj in scene_objects.items():
            pcd = obj.get("pcd")
            if pcd is not None and hasattr(pcd, "points") and len(pcd.points) > 0:
                centroids[str(obj_id)] = np.asarray(pcd.points).mean(axis=0)
            else:
                bbox = obj.get("bbox")
                if bbox is not None and hasattr(bbox, "center"):
                    centroids[str(obj_id)] = np.asarray(bbox.center)
        return centroids
