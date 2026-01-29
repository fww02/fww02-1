import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _ndimage = None

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

try:
    from .visualizer import SceneGraphVisualizer
except Exception:
    SceneGraphVisualizer = None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_list(x: Any) -> Any:
    """Convert numpy arrays / scalars to python lists recursively (best-effort)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _safe_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_list(v) for v in x]
    return x


@dataclass
class RegionNode:
    """A lightweight 'room-like' region node built from occupancy/island connectivity."""

    region_id: int
    floor_id: str
    created_step: int
    last_seen_step: int
    # boolean 2D mask in voxel grid coordinates, same shape as planner._vol_dim[:2]
    mask: Optional[np.ndarray] = None
    # list of object ids assigned to this region
    objects: List[int] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "room_id": f"{self.floor_id}_{self.region_id}",
            "name": None,
            "floor_id": self.floor_id,
            "objects": self.objects,
            "created_step": self.created_step,
            "last_seen_step": self.last_seen_step,
            # vertices/pcd fields are intentionally omitted in the online lightweight mode
        }


class ExplicitMemoryGraphBuilder:
    """Online, GPU-memory-friendly explicit memory hierarchy builder.

    Goal:
      - build building->floor->region(room)->object hierarchy online
      - save in an HOV-SG-compatible *json layout* (graph/floors, graph/rooms, graph/objects)

    Design constraints:
      - do not run heavy per-point feature fusion
      - keep all data on CPU (numpy / python)
    """

    def __init__(
        self,
        save_root: str,
        voxel_size: float,
        floor_id: str = "0",
        *,
        # region segmentation / tracking
        region_iou_match_threshold: float = 0.6,
        # filter small free-space components (in cells)
        min_region_area_cells: int = 200,
        # object assignment
        assign_dist_m: float = 4.0,
    ):
        self.save_root = save_root
        self.voxel_size = float(voxel_size)
        self.floor_id = str(floor_id)

        self.region_iou_match_threshold = float(region_iou_match_threshold)
        self.min_region_area_cells = int(min_region_area_cells)
        self.assign_dist_m = float(assign_dist_m)

        # region_id -> RegionNode
        self._regions: Dict[int, RegionNode] = {}
        self._next_region_id = 0

        # cached last region mask for fast stable tracking (single best candidate)
        self._last_region_mask: Optional[np.ndarray] = None
        self._last_region_id: Optional[int] = None

        # object_id -> region_id
        self._obj_to_region: Dict[int, int] = {}

        # metadata
        self._created = False
        self._episode_info: Dict[str, Any] = {}

        # trajectory recording for video generation
        self._trajectory_voxels: List[np.ndarray] = []

    def _label_free_space_regions(self, unoccupied: Optional[np.ndarray]) -> List[np.ndarray]:
        """Return a list of connected free-space component masks.

        unoccupied: bool/0-1 array, True indicates free space.
        """
        if unoccupied is None:
            return []
        free = unoccupied.astype(bool)
        if _ndimage is None:
            # scipy not available; fall back to a single region
            return [free]

        # 8-connectivity on the 2D grid
        structure = np.ones((3, 3), dtype=np.int8)
        labeled, n = _ndimage.label(free, structure=structure)
        if n <= 0:
            return []

        regions: List[np.ndarray] = []
        for lab in range(1, n + 1):
            mask = labeled == lab
            if mask.sum() < self.min_region_area_cells:
                continue
            regions.append(mask)
        # sort by area (desc)
        regions.sort(key=lambda m: int(m.sum()), reverse=True)
        return regions

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0

    def _match_or_create_region(self, mask: np.ndarray, step: int) -> int:
        """Match a component mask to existing regions by IoU, else create."""
        best_rid = None
        best_iou = -1.0
        for rid, rn in self._regions.items():
            if rn.mask is None:
                continue
            iou = self._mask_iou(rn.mask, mask)
            if iou > best_iou:
                best_iou = iou
                best_rid = rid

        if best_rid is not None and best_iou >= self.region_iou_match_threshold:
            rn = self._regions[best_rid]
            rn.last_seen_step = step
            rn.mask = mask
            return best_rid

        rid = self._next_region_id
        self._next_region_id += 1
        self._regions[rid] = RegionNode(
            region_id=rid,
            floor_id=self.floor_id,
            created_step=step,
            last_seen_step=step,
            mask=mask,
        )
        return rid

    def start_episode(self, episode_id: str, scene_id: str):
        self._episode_info = {
            "episode_id": str(episode_id),
            "scene_id": str(scene_id),
        }
        self._created = True

    def _assign_agent_region(self, step: int, tsdf_planner: Any) -> int:
        """Compute (or approximate) the region id where the agent currently is."""
        # Prefer free-space connected components, restricted to the current navigable island if available
        unoccupied = getattr(tsdf_planner, "unoccupied", None)
        island = getattr(tsdf_planner, "island", None)
        if unoccupied is None:
            # make sure planner maps were computed at least once
            return self._match_or_create_region(mask=np.ones((1, 1), dtype=bool), step=step)

        free = unoccupied.astype(bool)
        if island is not None:
            free = np.logical_and(free, island.astype(bool))

        # Try stable tracking against previous agent-region mask first
        if self._last_region_mask is not None and self._last_region_id is not None:
            # If last mask still overlaps with current free-space, keep it
            if self._mask_iou(self._last_region_mask, free) >= 0.2:
                # but we still want the precise connected component; fall back if scipy missing
                pass

        comps = self._label_free_space_regions(free)
        if not comps:
            rid = self._match_or_create_region(mask=free, step=step)
            self._last_region_id = rid
            self._last_region_mask = free
            return rid

        # choose the component that best matches last agent region (or the largest as fallback)
        best_mask = comps[0]
        if self._last_region_mask is not None:
            best_iou = -1.0
            for m in comps:
                iou = self._mask_iou(self._last_region_mask, m)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = m

        rid = self._match_or_create_region(mask=best_mask, step=step)
        self._last_region_id = rid
        self._last_region_mask = best_mask
        return rid

    def update(
        self,
        step: int,
        tsdf_planner: Any,
        scene_objects: Dict[int, Dict[str, Any]],
        agent_pts_habitat: np.ndarray,
    ):
        """Update the explicit graph builder.

        Args:
            step: current step index
            tsdf_planner: TSDFPlanner (CPU planner) used by 3D-Mem
            scene_objects: Scene.objects (MapObjectDict-like)
            agent_pts_habitat: (3,) habitat coordinate
        """
        if not self._created:
            raise RuntimeError("ExplicitMemoryGraphBuilder.start_episode() must be called.")

        # Ensure planner maps exist (unoccupied/island computed in update_frontier_map)
        cur_region_id = self._assign_agent_region(step, tsdf_planner)

        # record trajectory in voxel grid coordinates (for video generation)
        try:
            # rough conversion: habitat xyz -> 2D voxel (x, z)
            x_vox = agent_pts_habitat[0] / self.voxel_size
            z_vox = agent_pts_habitat[2] / self.voxel_size
            self._trajectory_voxels.append(np.array([x_vox, z_vox]))
        except Exception:
            pass

        # assign objects to region (simple heuristic: xy distance in habitat to agent within include dist)
        agent_xy = np.asarray(agent_pts_habitat)[[0, 2]]
        for obj_id, obj in scene_objects.items():
            bbox = obj.get("bbox", None)
            if bbox is None or not hasattr(bbox, "center"):
                continue
            center = np.asarray(bbox.center)
            obj_xy = center[[0, 2]]
            dist = float(np.linalg.norm(obj_xy - agent_xy))

            # only add objects we have at least seen once; do not gate by detections count too strictly here
            if obj.get("num_detections", 0) <= 0:
                continue

            # if an object is near current agent position, map it to current region
            # (this prevents drifting assignments when far away)
            if dist <= self.assign_dist_m:
                prev = self._obj_to_region.get(obj_id)
                if prev is not None and prev in self._regions and prev != cur_region_id:
                    # allow reassignment only if it was never finalized; keep latest
                    if obj_id in self._regions[prev].objects:
                        self._regions[prev].objects.remove(obj_id)
                self._obj_to_region[obj_id] = cur_region_id
                if obj_id not in self._regions[cur_region_id].objects:
                    self._regions[cur_region_id].objects.append(obj_id)

    def _floor_json(self) -> Dict[str, Any]:
        # keep it minimal but compatible with HOV-SG Floor.save() schema
        return {
            "floor_id": self.floor_id,
            "name": f"floor_{self.floor_id}",
            "rooms": [f"{self.floor_id}_{rid}" for rid in sorted(self._regions.keys())],
            "vertices": [],
            "floor_height": None,
            "floor_zero_level": None,
        }

    def _room_json(self, rid: int) -> Dict[str, Any]:
        return self._regions[rid].to_json()

    def _object_json(self, obj_id: int, obj: Dict[str, Any]) -> Dict[str, Any]:
        bbox = obj.get("bbox", None)
        center = None
        if bbox is not None and hasattr(bbox, "center"):
            center = np.asarray(bbox.center)

        embedding = obj.get("clip_ft", None)
        if embedding is not None:
            try:
                # move to cpu numpy if it's a torch tensor
                import torch

                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().float().cpu().numpy()
            except Exception:
                pass

        room_id = self._obj_to_region.get(obj_id)
        room_id_str = f"{self.floor_id}_{room_id}" if room_id is not None else None

        vertices = []
        if bbox is not None and hasattr(bbox, "get_box_points"):
            try:
                vertices = np.asarray(bbox.get_box_points())
            except Exception:
                vertices = []

        return {
            "object_id": str(obj_id),
            "room_id": room_id_str,
            "name": obj.get("class_name", None),
            "gt_name": None,
            "vertices": _safe_list(vertices) if len(vertices) else [],
            "bbox_center": _safe_list(center) if center is not None else None,
            "confidence": _safe_float(obj.get("conf", None)),
            "num_detections": int(obj.get("num_detections", 0)),
            "embedding": _safe_list(embedding) if embedding is not None else "",
            "image": obj.get("image", None),
        }

    def save(self, scene_objects: Dict[int, Dict[str, Any]]):
        """Save graph jsons + point clouds + visualizations to disk (HOV-SG-like directory layout)."""
        graph_root = os.path.join(self.save_root, "explicit_graph", "graph")
        floors_dir = os.path.join(graph_root, "floors")
        rooms_dir = os.path.join(graph_root, "rooms")
        objects_dir = os.path.join(graph_root, "objects")
        os.makedirs(floors_dir, exist_ok=True)
        os.makedirs(rooms_dir, exist_ok=True)
        os.makedirs(objects_dir, exist_ok=True)

        # floor
        with open(os.path.join(floors_dir, f"{self.floor_id}.json"), "w", encoding="utf-8") as f:
            json.dump(_safe_list(self._floor_json()), f, ensure_ascii=False, indent=2)

        # rooms
        for rid in sorted(self._regions.keys()):
            with open(
                os.path.join(rooms_dir, f"{self.floor_id}_{rid}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(_safe_list(self._room_json(rid)), f, ensure_ascii=False, indent=2)

        # objects (json + ply if available)
        for obj_id, obj in scene_objects.items():
            obj_json_path = os.path.join(objects_dir, f"{obj_id}.json")
            with open(obj_json_path, "w", encoding="utf-8") as f:
                json.dump(_safe_list(self._object_json(obj_id, obj)), f, ensure_ascii=False, indent=2)
            
            # save object point cloud if open3d available
            if o3d is not None:
                pcd = obj.get("pcd", None)
                if pcd is not None and hasattr(pcd, "points") and len(pcd.points) > 0:
                    try:
                        obj_ply_path = os.path.join(objects_dir, f"{obj_id}.ply")
                        o3d.io.write_point_cloud(obj_ply_path, pcd)
                    except Exception as e:
                        pass  # silently skip if write fails

        # meta
        meta = {
            "episode": self._episode_info,
            "num_regions": len(self._regions),
            "num_objects": len(scene_objects),
            "floor_id": self.floor_id,
            "voxel_size": self.voxel_size,
        }
        meta_path = os.path.join(self.save_root, "explicit_graph", "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_safe_list(meta), f, ensure_ascii=False, indent=2)

        # generate visualizations (region map, object distribution)
        self._save_visualizations(scene_objects, graph_root)

        # --- generate 3D hierarchical scene graph visualization (HOV-SG style) ---
        if SceneGraphVisualizer is not None:
            try:
                visualizer = SceneGraphVisualizer(voxel_size=self.voxel_size)
                
                # Static 3D scene graph image
                graph_3d_path = os.path.join(graph_root, "visualizations", "scene_graph_3d.png")
                visualizer.visualize_hierarchical_graph(
                    regions=self._regions,
                    scene_objects=scene_objects,
                    obj_to_region=self._obj_to_region,
                    floor_id=self.floor_id,
                    output_path=graph_3d_path,
                )
                
                # Rotating 3D scene graph video/gif
                graph_video_path = os.path.join(self.save_root, "explicit_graph", "scene_graph_3d.gif")
                visualizer.generate_graph_visualization_video(
                    regions=self._regions,
                    scene_objects=scene_objects,
                    obj_to_region=self._obj_to_region,
                    floor_id=self.floor_id,
                    output_video_path=graph_video_path,
                    n_frames=60,
                    fps=10,
                )
            except Exception as e:
                import logging
                logging.warning(f"3D scene graph visualization failed: {e}")

        # generate trajectory video if requested
        if len(self._trajectory_voxels) > 1:
            video_path = os.path.join(self.save_root, "explicit_graph", "trajectory.mp4")
            try:
                self.save_trajectory_video(self._trajectory_voxels, video_path, fps=5)
            except Exception as e:
                pass  # skip if video generation fails

    def _save_visualizations(self, scene_objects: Dict[int, Dict[str, Any]], graph_root: str):
        """Generate visualization images: region map, object counts per region, etc."""
        viz_dir = os.path.join(graph_root, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # 1) Draw region masks on a 2D occupancy map
        self._draw_region_map(viz_dir)

        # 2) Draw object distribution (bboxes + labels on topdown)
        self._draw_object_distribution(scene_objects, viz_dir)

    def _draw_region_map(self, viz_dir: str):
        """Draw a topdown view of regions (colored differently)."""
        if not self._regions:
            return
        # find bounding box for all region masks
        all_masks = [rn.mask for rn in self._regions.values() if rn.mask is not None]
        if not all_masks:
            return
        
        # stack masks and get max extent
        h_max = max(m.shape[0] for m in all_masks)
        w_max = max(m.shape[1] for m in all_masks)
        
        # create color image
        img = np.ones((h_max, w_max, 3), dtype=np.uint8) * 255  # white background
        
        cmap = plt.get_cmap("tab10")
        for idx, (rid, rn) in enumerate(self._regions.items()):
            if rn.mask is None:
                continue
            color_rgb = (np.array(cmap(idx % 10)[:3]) * 255).astype(np.uint8)
            mask_coords = np.argwhere(rn.mask)
            for (y, x) in mask_coords:
                if 0 <= y < h_max and 0 <= x < w_max:
                    img[y, x] = color_rgb

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, origin="lower")
        ax.set_title("Region Map (Topdown)", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "region_map.png"), dpi=150)
        plt.close(fig)

    def _draw_object_distribution(self, scene_objects: Dict[int, Dict[str, Any]], viz_dir: str):
        """Draw a topdown view showing object bboxes and counts per region."""
        if not self._regions:
            return
        
        # collect bbox centers in voxel 2D (assume we can project from habitat coords)
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # draw region boundaries
        all_masks = [rn.mask for rn in self._regions.values() if rn.mask is not None]
        if all_masks:
            h_max = max(m.shape[0] for m in all_masks)
            w_max = max(m.shape[1] for m in all_masks)
            base_img = np.ones((h_max, w_max), dtype=np.uint8) * 255
            
            cmap = plt.get_cmap("Pastel1")
            for idx, (rid, rn) in enumerate(self._regions.items()):
                if rn.mask is None:
                    continue
                color_val = int(cmap(idx % 9)[0] * 255)
                mask_coords = np.argwhere(rn.mask)
                for (y, x) in mask_coords:
                    if 0 <= y < h_max and 0 <= x < w_max:
                        base_img[y, x] = color_val
            ax.imshow(base_img, origin="lower", cmap="gray", alpha=0.3)
        
        # draw objects as scatter points (habitat xy -> voxel grid approximation)
        for obj_id, obj in scene_objects.items():
            bbox = obj.get("bbox", None)
            if bbox is None or not hasattr(bbox, "center"):
                continue
            center = np.asarray(bbox.center)
            # convert habitat xyz to 2D voxel grid for visualization (rough heuristic)
            # NOTE: this is approximate; exact conversion requires planner's vol_bnds
            x_vox = int(center[0] / self.voxel_size) + 100  # shift for viz
            z_vox = int(center[2] / self.voxel_size) + 100
            
            class_name = obj.get("class_name", "?")
            ax.scatter(z_vox, x_vox, s=50, c="red", alpha=0.7)
            ax.text(z_vox, x_vox, f"{obj_id}", fontsize=6, color="black")
        
        ax.set_title("Object Distribution (Topdown)", fontsize=14)
        ax.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "object_distribution.png"), dpi=150)
        plt.close(fig)

    def save_trajectory_video(
        self,
        trajectory_points: List[np.ndarray],
        output_path: str,
        fps: int = 5,
    ):
        """Generate a video showing agent trajectory overlaid on region map (similar to HOV-SG demo).

        Args:
            trajectory_points: list of (x, z) in voxel grid coords
            output_path: path to save .mp4 or .gif
            fps: frames per second
        """
        if not self._regions or not trajectory_points:
            return

        # prepare base image (region map)
        all_masks = [rn.mask for rn in self._regions.values() if rn.mask is not None]
        if not all_masks:
            return
        h_max = max(m.shape[0] for m in all_masks)
        w_max = max(m.shape[1] for m in all_masks)
        base_img = np.ones((h_max, w_max, 3), dtype=np.uint8) * 255

        cmap = plt.get_cmap("tab10")
        for idx, (rid, rn) in enumerate(self._regions.items()):
            if rn.mask is None:
                continue
            color_rgb = (np.array(cmap(idx % 10)[:3]) * 255).astype(np.uint8)
            mask_coords = np.argwhere(rn.mask)
            for (y, x) in mask_coords:
                if 0 <= y < h_max and 0 <= x < w_max:
                    base_img[y, x] = color_rgb

        # generate frames
        frames = []
        for step_idx, pt in enumerate(trajectory_points):
            frame = base_img.copy()
            # draw trajectory up to current step
            for i in range(step_idx):
                p0 = trajectory_points[i]
                p1 = trajectory_points[i + 1] if i + 1 < len(trajectory_points) else p0
                x0, z0 = int(p0[0]), int(p0[1])
                x1, z1 = int(p1[0]), int(p1[1])
                # draw line (simple bresenham-like or use cv2.line if available)
                # for simplicity, just mark points
                if 0 <= z0 < h_max and 0 <= x0 < w_max:
                    frame[z0, x0] = [0, 255, 0]  # green trajectory
            # mark current position
            x_cur, z_cur = int(pt[0]), int(pt[1])
            if 0 <= z_cur < h_max and 0 <= x_cur < w_max:
                # draw a circle-like marker (5x5 red square)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        yy, xx = z_cur + dy, x_cur + dx
                        if 0 <= yy < h_max and 0 <= xx < w_max:
                            frame[yy, xx] = [255, 0, 0]  # red agent
            frames.append(frame)

        # save as gif or mp4 (requires imageio or opencv)
        try:
            import imageio
            if output_path.endswith(".gif"):
                imageio.mimsave(output_path, frames, fps=fps)
            else:
                # save as mp4
                imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
        except Exception as e:
            # fallback: save individual frames
            frame_dir = output_path.replace(".mp4", "_frames").replace(".gif", "_frames")
            os.makedirs(frame_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(os.path.join(frame_dir, f"{i:04d}.png"), frame)

