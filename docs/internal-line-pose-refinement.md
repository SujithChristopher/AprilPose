# Internal Line Pose Refinement for a Single AprilTag

## Goal

OpenCV or an AprilTag detector can provide the four outer tag corners, but pose
from only four coplanar points is sensitive to corner noise. The goal is to use
the tag's known internal black/white pattern to extract additional geometric
constraints without using deep learning.

The important signal is not the decoded binary values by themselves. The useful
signal is the image-space location of the internal black/white cell boundaries.
Those boundaries can be detected as subpixel edge points, grouped into grid
lines, and used to refine homography and pose.

## High-Level Pipeline

```text
AprilTag detector
-> initial tag ID and outer corners
-> initial homography
-> reconstruct known tag cell map
-> enumerate visible internal black/white boundaries
-> project expected boundary segments into the image
-> sample perpendicular intensity profiles
-> detect signed subpixel edge points
-> robustly fit grid lines
-> intersect vertical and horizontal lines
-> refine homography and solvePnP
```

The first implementation should target this line-based refinement path before
attempting heavier photometric optimization.

## Coordinate Model

Represent the tag as a square grid in tag-local coordinates:

```text
(0, 0) ---------------- (N, 0)
  |                        |
  |                        |
(0, N) ---------------- (N, N)
```

Each cell occupies one unit square. A vertical grid boundary is at `x = c`; a
horizontal grid boundary is at `y = r`.

For pose, convert tag-local grid points to metric coordinates using the physical
tag size:

```text
metric_x = (grid_x / N - 0.5) * tag_size_m
metric_y = (grid_y / N - 0.5) * tag_size_m
metric_z = 0
```

The exact grid size and cell colors must come from the AprilTag family and tag
ID. The family layout is needed so we know which cell boundaries should contain
real black/white transitions.

## Visible Edge Segments

Do not blindly search every grid line. A boundary is visible only where adjacent
cells differ.

Vertical boundary between columns `c - 1` and `c`:

```text
if cell[row][c - 1] != cell[row][c]:
    segment from (c, row) to (c, row + 1) is a real edge
```

Horizontal boundary between rows `r - 1` and `r`:

```text
if cell[r - 1][col] != cell[r][col]:
    segment from (col, r) to (col + 1, r) is a real edge
```

Store each expected edge segment with:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class EdgeSegment:
    orientation: Literal["vertical", "horizontal"]
    line_index: int
    cell_row: int
    cell_col: int
    first_color: int
    second_color: int
    p0_tag: tuple[float, float]
    p1_tag: tuple[float, float]
```

`first_color` and `second_color` define expected edge polarity. For a vertical
edge, they are left and right cell colors. For a horizontal edge, they are top
and bottom cell colors.

## Edge Sampling

Use the initial homography to project each expected segment into the original
grayscale image. Sample a small number of points along the projected segment.
At each point, sample a short 1D intensity profile perpendicular to the segment.

```text
         profile samples
              |
              |
--------------+-------------- projected edge segment
              |
              |
```

Recommended starting parameters:

```text
samples_per_segment = 3 to 5
profile_radius_px = 3 to 5
profile_width = 2 * profile_radius_px + 1
```

For each profile:

1. Bilinearly sample grayscale intensities.
2. Smooth the profile lightly with a small 1D Gaussian.
3. Compute the derivative along the profile.
4. Search near the expected center for the strongest signed gradient.
5. Reject weak gradients or gradients with the wrong sign.
6. Fit a parabola around the derivative peak for subpixel edge position.

Expected polarity matters. If the model says the edge is black-to-white in the
sampling direction, keep positive gradients. If it says white-to-black, keep
negative gradients.

## Line Fitting

Bucket detected edge points by model grid line:

```text
("vertical", c)   -> points belonging to x = c
("horizontal", r) -> points belonging to y = r
```

Fit a 2D line for each bucket:

```text
ax + by + c = 0
```

Start with weighted least squares, using gradient magnitude as the point weight.
Add RANSAC only if real images show many false edge samples.

Reject a fitted line if:

```text
support point count is too low
average gradient is too weak
fit residual is too high
line is nearly duplicate or geometrically inconsistent
```

Perspective projection preserves straight planar lines, so line fitting is valid
after lens distortion is handled. For a first version, undistort the image before
running refinement. Alternatively, undistort sampled points before line fitting.

## Grid Intersections

Intersect reliable vertical and horizontal fitted lines:

```text
vertical_lines[c] x horizontal_lines[r] -> image point for grid coordinate (c, r)
```

Each intersection gives a known planar correspondence:

```text
object point: (metric_x, metric_y, 0)
image point:  detected line intersection
```

Use only intersections where both source lines are reliable. Optionally weight
each intersection by the support and residual of the two lines.

## Pose Refinement

Use the collected grid intersections to refine homography and pose:

```python
H_refined, inlier_mask = cv2.findHomography(
    object_grid_xy,
    image_points,
    cv2.RANSAC,
)

ok, rvec, tvec = cv2.solvePnP(
    object_points_3d,
    image_points_2d,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_IPPE,
)

rvec, tvec = cv2.solvePnPRefineLM(
    object_points_3d,
    image_points_2d,
    camera_matrix,
    dist_coeffs,
    rvec,
    tvec,
)
```

For the original four-corner baseline, also test `SOLVEPNP_IPPE_SQUARE`. For
the internal-grid correspondence set, `SOLVEPNP_IPPE` is the better fit because
the points are planar but not only the square's four corners.

## Performance Tradeoff

The added cost is mostly edge sampling:

```text
visible_edge_segments * samples_per_segment * profile_width
```

Example:

```text
100 visible segments * 5 samples * 9 profile pixels = 4,500 image samples
```

That is modest for a single tag. The cost becomes significant if we scan full
image regions, run global Hough/Canny pipelines, or do dense photometric
optimization.

Recommended adaptive behavior:

```text
cell width < 4 px:
    skip internal refinement

cell width 5-10 px:
    use sparse internal sampling

cell width > 10 px:
    use line refinement
```

For video, run the internal refinement every few frames if needed and use a
temporal filter between refinement frames.

## Accuracy Risks

Internal line refinement helps only when the internal cells are visible at enough
resolution. It can hurt if the tag is too small, blurred, saturated, or strongly
distorted.

Main risks:

```text
poor camera calibration
uncorrected lens distortion
motion blur
rolling shutter
glare or shadows
low cell resolution
wrong tag family layout
false edges from the background or tag print defects
```

The implementation should be able to fall back to the four-corner pose whenever
internal line confidence is low.

## Implementation Milestones

1. Baseline OpenCV AprilTag detection and four-corner pose.
2. Subpixel refinement for outer corners.
3. Tag family cell-map reconstruction.
4. Expected visible edge-segment generation.
5. Perpendicular profile sampling and signed subpixel edge detection.
6. Weighted line fitting for vertical and horizontal grid boundaries.
7. Grid intersection generation.
8. Refined homography and `solvePnP` integration.
9. Confidence scoring and fallback behavior.
10. Frame-to-frame temporal filtering for video use.

## Out of Scope for First Pass

Deep learning keypoint prediction, direct pose regression, and full photometric
rendering optimization are intentionally out of scope for the first prototype.
They can be evaluated later after the classical line-refinement baseline is
measured.
