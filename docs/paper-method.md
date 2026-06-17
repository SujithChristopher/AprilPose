# Internal Edge-Constrained AprilTag Pose Estimation

## Abstract-Level Summary

This method improves single-AprilTag orientation estimation by using the tag's
internal black-white cell boundaries as geometric constraints. A conventional
AprilTag detector provides the tag identity and four outer image corners. The
known binary cell layout is then reconstructed from the decoded identity, and
only cell boundaries with a predicted black-white transition are searched in the
original grayscale image. Detected subpixel edge samples are grouped into
projective grid lines, intersected, and used as additional coplanar
correspondences for pose estimation.

The approach is fully classical: it uses homography projection, one-dimensional
gradient edge localization, weighted line fitting, and planar PnP. No learned
model is required.

## Problem Setup

Let the AprilTag be a planar square target with physical side length `L`. For a
tag family with `M x M` payload bits and one black border, define an `N x N`
cell grid where `N = M + 2`. For the current `DICT_APRILTAG_36h11`
implementation, `M = 6` and `N = 8`.

The detector gives:

```text
tag id k
outer image corners u_0, ..., u_3
camera calibration K
```

The target-local grid coordinate `(x, y)` maps to metric object coordinates:

```text
X = (x / N - 0.5) L
Y = (y / N - 0.5) L
Z = 0
```

## Algorithm

1. Detect the AprilTag and estimate an initial homography `H` from the four
   outer corners.
2. Reconstruct the tag's binary cell image `B(y, x)` from the tag family and
   decoded id.
3. Enumerate all internal grid-boundary segments where adjacent cells differ.
   Vertical boundaries are kept when `B(r, c - 1) != B(r, c)`. Horizontal
   boundaries are kept when `B(r - 1, c) != B(r, c)`.
4. Project each expected boundary segment into the image using `H`.
5. Sample short grayscale intensity profiles perpendicular to each projected
   segment.
6. Use the expected black-white polarity to select the strongest signed
   gradient near the predicted edge center.
7. Estimate the subpixel edge position with a local parabolic fit around the
   gradient peak.
8. Bucket detected edge points by their model grid line.
9. Fit each grid line with weighted least squares, using edge-gradient magnitude
   as the weight.
10. Intersect reliable vertical and horizontal lines to obtain internal grid
    intersection points.
11. Convert grid intersections to metric object points and solve a planar PnP
    problem.

In the current implementation, OpenCV performs tag detection, fisheye
undistortion, baseline pose estimation, and PnP refinement. The internal
edge-search and line-intersection stage is implemented in Rust through
`maturin`/`pyo3` for realtime performance.

## Edge Localization

For a projected boundary point `p`, the algorithm samples a 1D profile along
the image-space normal direction `n`:

```text
I_i = I(p + i n),  i in [-r, r]
```

The profile is lightly smoothed and differentiated. If the model predicts a
black-to-white transition, positive gradients are accepted; if it predicts a
white-to-black transition, negative gradients are accepted. This polarity check
rejects many false edges caused by background texture, shadows, and nearby tag
structure.

For the selected gradient peak at index `i`, the subpixel offset is estimated by
fitting a parabola to the neighboring derivative values:

```text
delta = 0.5 * (g_{i-1} - g_{i+1}) / (g_{i-1} - 2g_i + g_{i+1})
```

The final edge sample is `p + (i + delta) n`.

## Pose Estimation

The baseline method estimates pose from only the four outer tag corners. The
proposed method estimates pose from up to `(N - 1)^2` internal grid
intersections. For `N = 8`, this gives up to 49 internal correspondences.

The pose is computed from coplanar points using OpenCV's planar PnP path and
then refined by nonlinear reprojection minimization. The refined pose is
accepted only when enough internal lines and intersections are detected.

## Evaluation Protocol

The dataset contains synchronized monochrome camera frames and OptiTrack motion
capture. The webcam timestamp stream contains a hardware sync flag:

```text
0 = mocap inactive
1 = mocap active
```

Frames are clipped to the active interval. For the current recording, the active
camera segment starts at webcam frame `262` and contains `3445` active frames.

OptiTrack rigid-body orientation is exported as quaternions at 100 Hz. Since the
camera and motion-capture streams have different sampling rates, OptiTrack
quaternions are interpolated to camera timestamps using spherical linear
interpolation.

Two pose trajectories are compared:

```text
baseline: four outer AprilTag corners
refined: internal edge-constrained grid intersections
```

The primary error metric is frame-to-frame rotation-change error:

```text
e_t = | angle(R_tag,t-1^-1 R_tag,t) - angle(R_mocap,t-1^-1 R_mocap,t) |
```

This metric is preferred over raw Euler angle differences because it is less
sensitive to fixed coordinate-frame offsets between the AprilTag and the
OptiTrack rigid body.

Euler trajectory errors are also reported after zeroing the first valid frame:

```text
relative_euler_error_t = relative_euler_tag,t - relative_euler_mocap,t
delta_euler_error_t =
    (relative_euler_tag,t - relative_euler_tag,t-1)
  - (relative_euler_mocap,t - relative_euler_mocap,t-1)
```

## Current Results

On the full active clip for tag id `14`:

```text
active frames: 3445
valid pose frames: 3444
```

Rotation-change error:

```text
baseline mean absolute delta-angle error: 4.094 deg
refined  mean absolute delta-angle error: 1.037 deg

baseline median absolute delta-angle error: 1.627 deg
refined  median absolute delta-angle error: 0.353 deg

baseline delta-angle error RMSE: 8.904 deg
refined  delta-angle error RMSE: 2.855 deg
```

Delta Euler RMSE:

```text
baseline x/y/z: 4.167 / 7.963 / 1.888 deg
refined  x/y/z: 1.573 / 2.452 / 0.628 deg
```

Runtime with the Rust internal-refinement backend:

```text
baseline pipeline median: 3.38 ms/frame
internal refinement median: 0.71 ms/frame
refined pipeline median: 4.11 ms/frame
```

The Rust backend makes the internal refinement stage approximately sub-
millisecond on the evaluated machine. AprilTag detection is now the dominant
runtime component.

## Limitations

The method still estimates pose from a single planar target, so it cannot remove
all planar-pose ambiguity or calibration sensitivity. It depends on sufficient
cell resolution, good camera calibration, and visible internal black-white
transitions. Euler-axis comparisons require care because the AprilTag coordinate
frame and OptiTrack rigid-body frame are not inherently identical. For this
reason, geodesic frame-to-frame rotation-change error is the preferred
trajectory metric.
