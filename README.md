
# 3D Gaussian Splatting Cinematic Navigation

This project is a small cinematic navigation pipeline on top of a 3D Gaussian Splatting (3DGS) scene:

- Loads an unpacked 3DGS `.ply` scene.
- Plans and renders a camera flight along a configurable path.
- Streams frames through `gsplat` → `ffmpeg` → MP4.
- Optionally runs YOLO detection and builds a second “revisit” pass where the camera returns to selected objects.
- Provides analysis tools for the scene (density maps, histograms, orientation stats, camera path plots).
- Has two main development branches:
  - **main**: flat XZ path + YOLO + revisit.
  - **3d-planner**: scene normalization + 3D path along waypoints + local floor sampling.
 
## Project goals & status

- [x] 1. Render a video from inside the scene (no need to be realistic)
- [x] 2. Detect objects in the rendered video
- [x] 3. 3D object detection
- [x] 4. Path planning
- [x] 5. Obstacle avoidance
- [x] 6. Rendered video that covers most of the scene/area
- [ ] 7. Render a 360° video
- [ ] 8. Interactive demo
- [ ] 9. Real-time preview of the scene or pipeline
- [x] 10. Produce artistic / professional / innovative / realistic result videos (high-quality rendering)


---

## 1. High-Level Architecture

Core ideas:

1. **Scene as Gaussians**  
   A 3DGS scene (PLY) is loaded into a `GaussiansNP` structure (NumPy-based) with:
   - `means` – N×3 Gaussian centers;
   - `quats` – N×4 rotations (WXYZ);
   - `scales`, `opacities`, `colors` – splat parameters.

2. **Camera Path Planner**  
   We generate a list of camera poses:
   ```python
   {
       "eye":    [x, y, z],         # camera position in world
       "center": [x, y, z],         # look-at point in world
       "view":   np.ndarray(4, 4),  # world->camera matrix
   }

Different planners are used in different branches (2D XZ path vs full 3D rail).

3. **Renderer**
   For each pose:

   * Build intrinsics `K` from FOV and resolution.
   * Call `gsplat.rendering.rasterization(...)` to get RGB (and optionally depth).
   * Stream raw frames into `ffmpeg` and encode to MP4.

4. **Detection & Revisit (YOLO)**

   * YOLO runs on rendered RGB frames (first pass).
   * Optionally back-project bounding boxes into world coordinates using depth.
   * Select a few objects, compute a “best observation frame” for each.
   * Build an extra “revisit” path: camera runs the rail again and stops to look at selected objects.

5. **Scene Analysis Tools**

   * Histograms for X/Y/Z distributions.
   * Top-down density maps.
   * Y-slices (floor/ceiling inspection).
   * Orientation histograms from Gaussians’ quaternions.
   * Camera path overlaid on XZ density.

---

## 2. Repository Structure (logical)

Main modules (names as used in code):

* `src/gsplat_scene.py`

  * `GaussiansNP` dataclass.
  * `load_gaussians_from_ply(path, max_points=...)`
  * Camera intrinsics builder: `build_intrinsics(fov_deg, width, height)`.
  * `look_at(eye, center, up)` to build a 4×4 view matrix.
  * 2D path planner with per-segment behaviors (2D branch).
  * 3D path planner with local floor sampling (3D branch).

* `src/render_utils.py`

  * `render_one_frame(...)` – debug single-frame render.
  * `render_frames_gsplat(...)` + `write_video(...)` – legacy in-memory path.
  * `render_gsplat_to_video_streaming(...)` – main streaming renderer with:

    * optional YOLO detection;
    * optional second “revisit” pass;
    * depth-backed 3D object centers.

* `src/analysis/research_scene.py`

  * Scene exploration utilities (histograms, density maps, Y-slices).
  * Quaternion → forward direction → (azimuth, elevation) utilities.
  * `save_camera_path_on_density_xz(...)` – camera path on XZ density map.

* `src/analysis/scene_normalizer.py` (3D branch)

  * Scene normalization:

    * fit floor plane from low Y points;
    * rotate to align floor with XZ and up with +Y;
    * optionally align main horizontal direction with X-axis (yaw);
    * optionally shift floor to Y ≈ 0.

* `src/main.py`

  * All orchestration:

    * load scene;
    * optionally normalize axes;
    * build camera path from config;
    * optionally run analysis plots;
    * stream video + YOLO + revisit;
    * write JSON with camera path and detections.
  * Uses two config dicts:

    * `SCENE_CONFIG_CONF_HALL` – scene + path;
    * `GLOBAL_CONFIG` – render/detection/analysis settings.

---

## 3. Runtime Environment and Stack

The project assumes:

* Python ≥ 3.10.
* CUDA-capable GPU.
* Main libraries:

  * `torch` + CUDA;
  * `gsplat` (Gaussian splatting);
  * `ffmpeg-python` (`ffmpeg` CLI in PATH);
  * `ultralytics` (YOLO);
  * `opencv-python` (drawing, color conversions);
  * `matplotlib` + `numpy` (analysis).

Typical usage is inside a **Docker container** with:

* CUDA runtime and drivers mounted (`--gpus all`).
* All Python dependencies pre-installed.
* `ffmpeg` available inside the container.

The pipeline will fail early if no CUDA device is visible.

---

## 4. Rendering Pipeline

### 4.1 Loading and Normalization

1. `load_gaussians_from_ply(path, max_points)`:

   * Reads PLY into arrays;
   * Optionally downsamples to `max_points` to control memory;
   * Returns `GaussiansNP`.

2. Optional normalization (3D branch):

   * Use `normalize_scene_axes(...)`:

     * Sample bottom part of scene (low Y) to fit a floor plane.
     * Rotate so that floor normal aligns with +Y (no flips > 90°).
     * Estimate main horizontal direction in XZ and optionally align yaw.
     * Optionally shift floor to Y≈0.
   * Updates all `means` and rotations consistently, returns metadata:

     * yaw angle,
     * main horizontal direction before/after normalization, etc.

### 4.2 Streaming Render with YOLO & Revisit

`render_gsplat_to_video_streaming(...)`:

* Inputs:

  * `gauss: GaussiansNP`
  * `poses: List[{"view", "eye", "center"}]`
  * `width`, `height`, `fov_deg`
  * `device` (CUDA)
  * `out_path`, `fps`
  * `detect`, `yolo_model_path`, `yolo_conf`, `draw_boxes`
  * Revisit config: `revisit_top_k`, `revisit_min_world_dist`, `revisit_stop_seconds`, `revisit_interp_seconds`.

* Steps:

  1. Move Gaussian attributes to GPU.

  2. Build intrinsics `K` once via `build_intrinsics`.

  3. Start an `ffmpeg` process in rawvideo RGB mode (pipe).

  4. **First pass** (main rail):

     * For each pose:

       * Call `rasterization(...)` with:

         * `render_mode="RGB"` or `"RGB+ED"` (if depth is needed).
       * Convert RGB float [0,1] → uint8 RGB.
       * If `detect`:

         * Run YOLO on the RGB frame.
         * Collect boxes: `[x1, y1, x2, y2], cls, conf`.
         * If depth available:

           * Back-project bounding boxes to world using `_backproject_box_center_to_world(...)`.
           * Store 3D center `center_world`, approximate radius `radius_world`, frame index, and camera eye.
         * Optionally overlay YOLO’s own annotated frame (`res.plot()`) instead of raw RGB.
       * Write RGB frame bytes into ffmpeg stdin.

  5. **Revisit object selection**:

     * From all detections with valid `center_world`, build a list of objects.
     * `_select_revisit_objects(...)`:

       * Randomly shuffle candidates.
       * Pick up to `revisit_top_k` objects, enforcing minimum 3D distance `revisit_min_world_dist` between centers.

  6. **Attach best observation frames**:

     * `_attach_best_frame_indices(...)`:

       * For each object, compute distance from all `eye` positions on the rail.
       * Store `best_frame_idx` with minimal distance for that object.

  7. **Build revisit path**:

     * `build_revisit_path_with_stops(...)`:

       * Replays the original rail from frame 0 to last.
       * When `i == best_frame_idx` for some object:

         * Insert a sequence of interpolated poses gradually rotating from base look direction to look at the object center.
         * Hold gaze on the object for `stop_seconds` (multiple frames).
       * Orientation is computed via `look_at(eye, obj_center, up=[0,1,0])`.

  8. **Second pass** (revisit rail, no YOLO):

     * For each revisit pose:

       * Render RGB with gsplat.
       * Overlay approximate boxes for selected objects:

         * `_draw_selected_objects_boxes(...)`:

           * Project `center_world` into current camera.
           * Skip if object is behind camera or outside image.
           * Estimate box size from `radius_world` and depth.
           * Draw green square box and caption `cls_name` + `conf`.
       * Write frame RGB into ffmpeg.

  9. Close ffmpeg, check exit code.

  10. Return list of per-frame detections (first pass) if `detect=True`, else `None`.

---

## 5. Camera Path Planning

There are two conceptual phases for this project, reflected in two branches.

### 5.1 2D Branch (flat XZ, with behaviors)

In the 2D branch, the camera follows a polyline in the horizontal XZ plane:

* Up axis: +Y.
* Height: fixed relative to scene diagonal (e.g., `center_y + height_fraction * diag`).
* Path is defined as `path_with_behaviors`:

```python
path_with_behaviors = [
    ([x0, z0], behavior0),  # segment 0: [x0,z0] -> [x1,z1]
    ([x1, z1], behavior1),  # segment 1: [x1,z1] -> [x2,z2]
    ...
]
```

Per-segment behaviors can modify orientation or height while ensuring smooth stitching:

* `mode: "look_at_point"` – camera gradually looks at a target world point, with tunable `strength`.
* `mode: "height_arc"` – adds a smooth vertical bump (arc) over that segment.
* `mode: "extra_yaw"` – adds yaw twist along a segment, e.g., 180–360° spin.

Base behavior: camera looks slightly ahead along the path (look-ahead fraction of total path length).

This branch culminated when YOLO detection and revisit logic were integrated on top of this flat path.

### 5.2 3D Branch (normalized scene + 3D rail + local floor)

In the 3D branch, the focus shifted to:

1. **Scene normalization**:

   * Fit a floor plane from low-Y Gaussians.
   * Rotate so that:

     * floor normal ≈ +Y (no flips > 90°);
     * main horizontal direction in XZ is aligned (yaw limited to some max angle).
   * Optionally shift floor height so that Y=0 corresponds to the floor.

2. **3D waypoints rail**
   Scene config includes full 3D waypoints:

   ```python
   "straight_path_waypoints_xyz": [
       [-3.0, 2.0, -62.0],
       [-3.0, 2.0, -40.0],
       ...
   ]
   ```

   These define the rail in 3D space. The camera planner:

   ```python
   generate_camera_poses_straight_path_3d(
       means=gauss.means,
       waypoints_xyz=...,
       num_frames=...,
       patch_size=...,
       start_cam_y=...,
       floor_percentile=...,
       lookahead_fraction=...
   )
   ```

   Key points:

   * Path is parameterized by arc length along the polyline.
   * For each frame:

     * Interpolate `eye` along the rail.
     * Sample a local K×K patch of Gaussians around the XZ projection of this point.
     * Estimate local floor height from that patch using `floor_percentile` (robust to outliers).
     * Adjust camera Y: keep a stable offset above local floor.
     * Use look-ahead along path to define forward direction and build `view = look_at(eye, center, up)`.

3. **2D vs 3D view**

   * Planning still happens primarily in XZ (for look-ahead and path parameterization), but height is local and scene-aware.
   * This branch reuses the same renderer and YOLO + revisit logic as the 2D branch.

---

## 6. Scene Analysis Tools

`src/analysis/research_scene.py` is a “lab” module for understanding scenes:

* Quaternion utilities:

  * `quat_forward_axis_wxyz(quats)` → per-Gaussian forward direction in world coordinates.
  * `forward_angles_y_up(dirs)` → yaw (azimuth) and elevation distributions.

* Plot helpers:

  * `save_histogram(data, path, title, xlabel, bins)`
  * `save_density_xz(means, path, grid_res, title)`

    * histogram2D of XZ projection (top-down density map).
  * `save_y_slices_xz(means, outdir, grid_res, thickness_frac, num_slices)`

    * horizontal slices along Y; for each slice, density in XZ plane.
  * `save_camera_path_on_density_xz(means, poses, out_path, grid_res, arrow_stride)`

    * overlay camera path and view directions on XZ density map.

These tools are triggered from `main` when `GLOBAL_CONFIG["analyze_scene"]` is `True`.

---

## 7. Configuration

### 7.1 Scene Config

In `src/main.py`, the default example is `SCENE_CONFIG_CONF_HALL`. It controls:

* `scene_path`: path to `.ply` scene.
* `normalize_scene`: whether to run normalization.
* `normalization`: parameters for floor fitting and yaw alignment.
* `path_type`: `"spline"` or `"straight"` (in the final state only `"straight"` + 3D rail are used).
* `straight_path_waypoints_xyz`: full 3D rail.
* `straight_patch_size`: patch size for local floor sampling.
* `straight_floor_percentile`: percentile used to estimate local floor from sample points.
* `straight_start_cam_y`: optional fixed starting camera height (otherwise estimated).
* `spline_base_height_offset`: baseline height above floor (for spline version, if used).

### 7.2 Global Config

`GLOBAL_CONFIG` controls:

* Output and render:

  * `outdir`
  * `seconds`, `fps`
  * `fov_deg`
  * `resolution`
  * `max_splats`
* Detection:

  * `detect` (True/False)
  * `yolo_model`
  * `yolo_conf`
  * `draw_boxes`
* Scene analysis:

  * `analyze_scene`
  * `analysis` dict (grid resolution, slice thickness, num slices, angle histogram bins).
* Camera path plotting:

  * `plot_camera_path`
  * `camera_path_plot`, `camera_path_grid_res`, `camera_path_arrow_stride`.

Revisit-specific parameters (`revisit_top_k`, etc.) are passed directly to `render_gsplat_to_video_streaming` in `main` and can be moved into `GLOBAL_CONFIG` if desired.

---

## 8. How to Run

### 8.1 Setup (high level)

1. Build or pull the Docker image with:

   * Python + CUDA,
   * `torch`, `gsplat`, `ultralytics`, `opencv-python`, `matplotlib`, `ffmpeg-python`,
   * `ffmpeg` installed in the container.

2. Put your 3DGS scene:

   * Example: `scenes/outdoor-street.ply`.

3. Adjust configs in `src/main.py`:

   * `SCENE_CONFIG_CONF_HALL["scene_path"]` to point to your `.ply`.
   * Adjust `straight_path_waypoints_xyz` to define camera rail.
   * Tune normalization and planning parameters if needed.
   * Set detection flags depending on whether you want YOLO + revisit.

### 8.2 Running

From inside the container, at project root:

```bash
python3 -m src.main
```

Optionally, you can import `main` and call it with your own config dicts:

```python
from src.main import main, SCENE_CONFIG_CONF_HALL, GLOBAL_CONFIG

my_scene_cfg = {**SCENE_CONFIG_CONF_HALL, "scene_path": "scenes/MyScene.ply"}
my_global_cfg = {**GLOBAL_CONFIG, "seconds": 30.0, "fps": 30, "detect": True}

main(scene_cfg=my_scene_cfg, global_cfg=my_global_cfg)
```

### 8.3 Outputs

* MP4 video:

  * `<outdir>/panorama_tour.mp4`.

* Camera path JSON:

  * `<outdir>/camera_path.json`
  * Contains per-frame `eye`, `center`, and `view` matrix.

* YOLO detections (if enabled):

  * `<outdir>/detections_yolo.json`
  * List of frames with detections, including optionally `center_world` and `radius_world`.

* Scene analysis (if enabled):

  * `<outdir>/<analysis_subdir>/...` with histograms, density maps, slices.
  * `<outdir>/camera_path_on_density_xz.png` – path overlay.

---

## 9. Problems, Challenges and Lessons Learned

Some of the main issues encountered:

1. **Memory and performance**

   * Initial implementation rendered all frames to memory, then wrote video.
   * This quickly hits GPU/CPU RAM limits for longer sequences.
   * Solution: streaming renderer `render_gsplat_to_video_streaming` that:

     * Keeps Gaussians on GPU;
     * Renders frame-by-frame;
     * Pipes raw RGB to `ffmpeg` without storing the full clip.

2. **GPU-only pipeline**

   * `gsplat` requires CUDA; there is no CPU fallback path here.
   * The code explicitly checks for CUDA and fails early if unavailable.

3. **Coordinate systems and scene orientation**

   * Raw 3DGS scenes can be arbitrarily rotated or translated.
   * Floor might not be aligned with the XZ plane; scene could be tilted or rotated.
   * Normalization needed careful handling to avoid flipping axes, especially:

     * Keeping +Y as “up”;
     * Avoiding rotations > 90° that would invert the scene;
     * Limiting yaw corrections to a reasonable angle.

4. **Local floor estimation**

   * A naive global floor estimate is often insufficient:

     * Scenes can have stairs, slopes, obstacles.
   * The 3D branch introduced a local patch-based approach:

     * For each camera position, sample nearby Gaussians;
     * Estimate a local floor height via a percentile (robust to outliers);
     * Maintain a stable offset above floor for camera height.

5. **Back-projecting YOLO detections into 3D**

   * Requires reliable depth:

     * `gsplat` provides an “expected depth” channel, which is not a perfect geometric depth but is usable.
   * Bounding boxes can include floor/background points:

     * To reduce artifacts, the code samples a small grid of pixels inside the box and averages back-projected points.
     * Still, wrong depth samples can yield 3D centers that sit on floor or behind the actual object.

6. **Revisit path design**

   * The revisit pass has to:

     * Use the existing rail;
     * Not introduce large jumps in camera position;
     * Avoid abrupt changes in orientation.
   * Solution:

     * For each selected object, camera stops at the frame where the object is closest to the rail;
     * Orientation smoothly interpolates from forward path direction to “look at object”;
     * An extra hold interval keeps the object in view.

7. **Branch divergence**

   * One branch stopped at 2D XZ path + YOLO + revisit.
   * Another evolved into full 3D rail with normalization + local floor.
   * Keeping the code reasonably modular (shared renderer, shared analysis) was important to avoid duplication.

---

## 10. Future Work

Potential extensions:

1. **Better object–floor separation**

   * Improve 3D object center estimation:

     * Use robust statistics, outlier rejection, or super-voxel clustering.
     * Use multiple frames and triangulation instead of single-frame depth.

2. **True 3D spline planner**

   * Replace polyline-based 3D path with a spline (Catmull–Rom or B-spline) in full 3D, with:

     * Curvature constraints;
     * Collision avoidance (with Gaussians as obstacles).

3. **Semantic-aware planning**

   * Use YOLO detections and/or semantic labels in the 3DGS to:

     * Plan paths that maximize coverage of specific object categories;
     * Automatically generate narrative tours (“walk through all chairs, then all doors”).

4. **Interactive configuration**

   * Move from in-code dictionaries to:

     * External YAML/JSON configs;
     * Simple CLI for switching scenes and paths;
     * Maybe a small web UI to draw waypoints on top-down density maps.

5. **Multi-camera rigs**

   * Extend poses to support:

     * Stereo pairs or rigs;
     * Alternative projection models (e.g., fisheye, 360°).

6. **Realtime mode**

   * Investigate whether the pipeline can be adapted to a semi-realtime viewer:

     * Preload Gaussians;
     * Render camera paths on-the-fly with YOLO running in a separate thread.

---

## 11. Summary

The project provides a compact but complete stack for:

* Loading 3D Gaussian Splatting scenes;
* Planning cinematic camera paths (2D XZ or 3D with local floor);
* Rendering video via gsplat and ffmpeg, in a GPU-friendly streaming fashion;
* Running YOLO detection on rendered frames;
* Building a second “revisit” tour to look again at selected objects in 3D;
* Analyzing scenes with various plots and statistics.

It is deliberately kept config-driven (scene + global configs), and can be extended with new path planners, detectors, and analysis tools as needed.

