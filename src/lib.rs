use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone, Copy)]
struct EdgePoint {
    x: f64,
    y: f64,
    weight: f64,
}

#[derive(Clone, Copy)]
struct FittedLine {
    a: f64,
    b: f64,
    c: f64,
    rms: f64,
    count: usize,
}

fn project(h: &[[f64; 3]; 3], x: f64, y: f64) -> Option<(f64, f64)> {
    let px = h[0][0] * x + h[0][1] * y + h[0][2];
    let py = h[1][0] * x + h[1][1] * y + h[1][2];
    let pz = h[2][0] * x + h[2][1] * y + h[2][2];
    if pz.abs() < 1e-12 {
        return None;
    }
    Some((px / pz, py / pz))
}

fn bilinear(gray: &[u8], height: usize, width: usize, x: f64, y: f64) -> Option<f64> {
    if x < 0.0 || y < 0.0 || x >= (width - 1) as f64 || y >= (height - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let p00 = gray[y0 * width + x0] as f64;
    let p01 = gray[y0 * width + x0 + 1] as f64;
    let p10 = gray[(y0 + 1) * width + x0] as f64;
    let p11 = gray[(y0 + 1) * width + x0 + 1] as f64;

    let top = (1.0 - dx) * p00 + dx * p01;
    let bottom = (1.0 - dx) * p10 + dx * p11;
    Some((1.0 - dy) * top + dy * bottom)
}

fn subpixel_peak_offset(left: f64, center: f64, right: f64) -> f64 {
    let denominator = left - 2.0 * center + right;
    if denominator.abs() < 1e-9 {
        0.0
    } else {
        (0.5 * (left - right) / denominator).clamp(-1.0, 1.0)
    }
}

fn detect_segment_points(
    gray: &[u8],
    height: usize,
    width: usize,
    h: &[[f64; 3]; 3],
    orientation: usize,
    first_color: u8,
    second_color: u8,
    p0_tag: (f64, f64),
    p1_tag: (f64, f64),
    samples_per_segment: usize,
    profile_radius: usize,
    min_gradient: f64,
) -> Vec<EdgePoint> {
    let Some((p0x, p0y)) = project(h, p0_tag.0, p0_tag.1) else {
        return Vec::new();
    };
    let Some((p1x, p1y)) = project(h, p1_tag.0, p1_tag.1) else {
        return Vec::new();
    };

    let tx = p1x - p0x;
    let ty = p1y - p0y;
    let length = (tx * tx + ty * ty).sqrt();
    if length < 2.0 {
        return Vec::new();
    }

    let center_tag = ((p0_tag.0 + p1_tag.0) * 0.5, (p0_tag.1 + p1_tag.1) * 0.5);
    let normal_target_tag = if orientation == 0 {
        (center_tag.0 + 0.1, center_tag.1)
    } else {
        (center_tag.0, center_tag.1 + 0.1)
    };
    let Some((cx, cy)) = project(h, center_tag.0, center_tag.1) else {
        return Vec::new();
    };
    let Some((nx_target, ny_target)) = project(h, normal_target_tag.0, normal_target_tag.1) else {
        return Vec::new();
    };
    let mut nx = nx_target - cx;
    let mut ny = ny_target - cy;
    let normal_norm = (nx * nx + ny * ny).sqrt();
    if normal_norm < 1e-9 {
        return Vec::new();
    }
    nx /= normal_norm;
    ny /= normal_norm;

    let polarity = if second_color > first_color {
        1.0
    } else {
        -1.0
    };
    let profile_width = 2 * profile_radius + 1;
    let center_gradient_index = profile_radius.saturating_sub(1);
    let mut edge_points = Vec::new();

    for sample_index in 0..samples_per_segment {
        let alpha = (sample_index as f64 + 0.5) / samples_per_segment as f64;
        let center_x = (1.0 - alpha) * p0x + alpha * p1x;
        let center_y = (1.0 - alpha) * p0y + alpha * p1y;

        let mut profile = Vec::with_capacity(profile_width);
        let mut valid = true;
        for index in 0..profile_width {
            let offset = index as isize - profile_radius as isize;
            let x = center_x + offset as f64 * nx;
            let y = center_y + offset as f64 * ny;
            if let Some(value) = bilinear(gray, height, width, x, y) {
                profile.push(value);
            } else {
                valid = false;
                break;
            }
        }
        if !valid || profile.len() < 3 {
            continue;
        }

        let mut smoothed = profile.clone();
        for index in 1..profile.len() - 1 {
            smoothed[index] =
                0.25 * profile[index - 1] + 0.5 * profile[index] + 0.25 * profile[index + 1];
        }

        let mut signed_gradients = Vec::with_capacity(smoothed.len() - 1);
        for index in 0..smoothed.len() - 1 {
            signed_gradients.push(polarity * (smoothed[index + 1] - smoothed[index]));
        }

        let search_start = center_gradient_index.saturating_sub(2);
        let search_end = (center_gradient_index + 3).min(signed_gradients.len());
        if search_start >= search_end {
            continue;
        }

        let mut gradient_index = search_start;
        let mut strength = signed_gradients[search_start];
        for (relative_index, value) in signed_gradients[search_start..search_end]
            .iter()
            .enumerate()
        {
            if *value > strength {
                gradient_index = search_start + relative_index;
                strength = *value;
            }
        }
        if strength < min_gradient {
            continue;
        }

        let mut offset_position = gradient_index as f64 - profile_radius as f64 + 0.5;
        if gradient_index > 0 && gradient_index + 1 < signed_gradients.len() {
            offset_position += subpixel_peak_offset(
                signed_gradients[gradient_index - 1],
                signed_gradients[gradient_index],
                signed_gradients[gradient_index + 1],
            );
        }

        edge_points.push(EdgePoint {
            x: center_x + offset_position * nx,
            y: center_y + offset_position * ny,
            weight: strength,
        });
    }

    edge_points
}

fn fit_weighted_line(points: &[EdgePoint], min_points: usize) -> Option<FittedLine> {
    if points.len() < min_points {
        return None;
    }

    let mut weight_sum = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for point in points {
        let weight = point.weight.max(1e-6);
        weight_sum += weight;
        cx += weight * point.x;
        cy += weight * point.y;
    }
    if weight_sum <= 0.0 {
        return None;
    }
    cx /= weight_sum;
    cy /= weight_sum;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for point in points {
        let weight = point.weight.max(1e-6);
        let dx = point.x - cx;
        let dy = point.y - cy;
        sxx += weight * dx * dx;
        sxy += weight * dx * dy;
        syy += weight * dy * dy;
    }
    sxx /= weight_sum;
    sxy /= weight_sum;
    syy /= weight_sum;

    let trace = sxx + syy;
    let determinant_term = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let lambda_min = 0.5 * (trace - determinant_term);

    let mut a = sxy;
    let mut b = lambda_min - sxx;
    if (a * a + b * b).sqrt() < 1e-12 {
        a = lambda_min - syy;
        b = sxy;
    }
    let norm = (a * a + b * b).sqrt();
    if norm < 1e-12 {
        return None;
    }
    a /= norm;
    b /= norm;
    let c = -(a * cx + b * cy);

    let mut weighted_residual = 0.0;
    for point in points {
        let residual = (a * point.x + b * point.y + c).abs();
        weighted_residual += point.weight.max(1e-6) * residual * residual;
    }
    let rms = (weighted_residual / weight_sum).sqrt();
    Some(FittedLine {
        a,
        b,
        c,
        rms,
        count: points.len(),
    })
}

fn intersect_lines(vertical: FittedLine, horizontal: FittedLine) -> Option<(f64, f64)> {
    let denominator = vertical.a * horizontal.b - vertical.b * horizontal.a;
    if denominator.abs() < 1e-9 {
        return None;
    }
    let x = (vertical.b * horizontal.c - vertical.c * horizontal.b) / denominator;
    let y = (vertical.c * horizontal.a - vertical.a * horizontal.c) / denominator;
    Some((x, y))
}

#[pyfunction]
#[pyo3(signature = (
    gray,
    cells,
    homography,
    samples_per_segment,
    profile_radius,
    min_gradient,
    min_line_points,
    max_line_rms
))]
fn refine_internal_lines_rust<'py>(
    py: Python<'py>,
    gray: PyReadonlyArray2<'py, u8>,
    cells: PyReadonlyArray2<'py, u8>,
    homography: PyReadonlyArray2<'py, f64>,
    samples_per_segment: usize,
    profile_radius: usize,
    min_gradient: f64,
    min_line_points: usize,
    max_line_rms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let gray_array = gray.as_array();
    let cells_array = cells.as_array();
    let homography_array = homography.as_array();

    let (height, width) = gray_array.dim();
    let (grid_rows, grid_cols) = cells_array.dim();
    if grid_rows != grid_cols {
        return Err(PyValueError::new_err("cells must be a square 2D array"));
    }
    if homography_array.dim() != (3, 3) {
        return Err(PyValueError::new_err("homography must have shape (3, 3)"));
    }

    let gray_slice = gray_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("gray image must be C-contiguous"))?;
    let cells_slice = cells_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cells array must be C-contiguous"))?;
    let h_slice = homography_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("homography must be C-contiguous"))?;
    let h = [
        [h_slice[0], h_slice[1], h_slice[2]],
        [h_slice[3], h_slice[4], h_slice[5]],
        [h_slice[6], h_slice[7], h_slice[8]],
    ];

    let mut buckets: Vec<Vec<EdgePoint>> = vec![Vec::new(); 2 * grid_cols];

    for col in 1..grid_cols {
        for row in 0..grid_rows {
            let left = cells_slice[row * grid_cols + col - 1];
            let right = cells_slice[row * grid_cols + col];
            if left == right {
                continue;
            }
            let points = detect_segment_points(
                gray_slice,
                height,
                width,
                &h,
                0,
                left,
                right,
                (col as f64, row as f64),
                (col as f64, row as f64 + 1.0),
                samples_per_segment,
                profile_radius,
                min_gradient,
            );
            buckets[col].extend(points);
        }
    }

    for row in 1..grid_rows {
        for col in 0..grid_cols {
            let top = cells_slice[(row - 1) * grid_cols + col];
            let bottom = cells_slice[row * grid_cols + col];
            if top == bottom {
                continue;
            }
            let points = detect_segment_points(
                gray_slice,
                height,
                width,
                &h,
                1,
                top,
                bottom,
                (col as f64, row as f64),
                (col as f64 + 1.0, row as f64),
                samples_per_segment,
                profile_radius,
                min_gradient,
            );
            buckets[grid_cols + row].extend(points);
        }
    }

    let mut vertical_lines: Vec<Option<FittedLine>> = vec![None; grid_cols];
    let mut horizontal_lines: Vec<Option<FittedLine>> = vec![None; grid_rows];
    let mut line_records: Vec<f64> = Vec::new();

    for col in 1..grid_cols {
        if let Some(line) = fit_weighted_line(&buckets[col], min_line_points) {
            if line.rms <= max_line_rms {
                vertical_lines[col] = Some(line);
                line_records.extend_from_slice(&[
                    0.0,
                    col as f64,
                    line.a,
                    line.b,
                    line.c,
                    line.rms,
                    line.count as f64,
                ]);
            }
        }
    }

    for row in 1..grid_rows {
        if let Some(line) = fit_weighted_line(&buckets[grid_cols + row], min_line_points) {
            if line.rms <= max_line_rms {
                horizontal_lines[row] = Some(line);
                line_records.extend_from_slice(&[
                    1.0,
                    row as f64,
                    line.a,
                    line.b,
                    line.c,
                    line.rms,
                    line.count as f64,
                ]);
            }
        }
    }

    let mut object_grid: Vec<f64> = Vec::new();
    let mut image_points: Vec<f64> = Vec::new();
    for (col, vertical) in vertical_lines.iter().enumerate().take(grid_cols).skip(1) {
        let Some(vertical) = vertical else {
            continue;
        };
        for (row, horizontal) in horizontal_lines.iter().enumerate().take(grid_rows).skip(1) {
            let Some(horizontal) = horizontal else {
                continue;
            };
            if let Some((x, y)) = intersect_lines(*vertical, *horizontal) {
                object_grid.extend_from_slice(&[col as f64, row as f64]);
                image_points.extend_from_slice(&[x, y]);
            }
        }
    }

    let point_count = object_grid.len() / 2;
    let line_count = line_records.len() / 7;
    let object_grid_array = PyArray2::from_vec2(
        py,
        &(0..point_count)
            .map(|index| vec![object_grid[2 * index], object_grid[2 * index + 1]])
            .collect::<Vec<_>>(),
    )?;
    let image_points_array = PyArray2::from_vec2(
        py,
        &(0..point_count)
            .map(|index| vec![image_points[2 * index], image_points[2 * index + 1]])
            .collect::<Vec<_>>(),
    )?;
    let line_records_array = PyArray2::from_vec2(
        py,
        &(0..line_count)
            .map(|index| {
                let start = 7 * index;
                line_records[start..start + 7].to_vec()
            })
            .collect::<Vec<_>>(),
    )?;
    let bucket_counts: Vec<i64> = buckets.iter().map(|bucket| bucket.len() as i64).collect();

    let result = PyDict::new(py);
    result.set_item("object_grid", object_grid_array)?;
    result.set_item("image_points", image_points_array)?;
    result.set_item("line_records", line_records_array)?;
    result.set_item("bucket_counts", PyArray1::from_vec(py, bucket_counts))?;
    Ok(result)
}

#[pymodule]
fn aprilpose_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(refine_internal_lines_rust, module)?)?;
    module.add(
        "__all__",
        PyList::new(module.py(), ["refine_internal_lines_rust"])?,
    )?;
    Ok(())
}
