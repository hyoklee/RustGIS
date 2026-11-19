//! NetCDF-4 and HDF5 rasters, read natively (no GDAL) with the pure-Rust `hdf5-reader` crate, so
//! the browser reads them too (#396). terrio's HDF5 driver
//! (opengeos/terrio `crates/terrio/src/drivers/hdf5.rs`) reads them the same way on the desktop.
//!
//! A file holds many variables; one is read: the one named (`HDF5:"f.h5"://path` or a bare
//! `/group/var` path), else the first two- or three-dimensional numeric variable that isn't a
//! coordinate (groups walked depth first, children by name, as GDAL lists subdatasets). A
//! three-dimensional variable's first dimension is its bands (the first [`MAX_BANDS`] of them); a
//! four-dimensional one's first two are taken at index 0 (time, level). `_FillValue` (or
//! `missing_value`) is the no-data value and `scale_factor`/`add_offset` are applied.
//!
//! Georeferencing: CF one-dimensional `lat`/`lon` (`latitude`/`longitude`, `y`/`x` in degrees)
//! coordinate variables beside it, or an HDF-EOS5 geographic grid's `StructMetadata.0`. Anything
//! else (swaths with two-dimensional latitude/longitude arrays, projected grids) opens as a scan,
//! with its pixels shown but unplaced, and says so.

use hdf5_reader::messages::datatype::Datatype;
use hdf5_reader::group::Group;
use hdf5_reader::{Dataset, Hdf5File, SliceInfo, SliceInfoElem};

use super::{ColorHint, Crs, GeoTransform, Raster, max_samples, unreferenced};
use crate::{DataError, Result};

/// File extensions read here (NetCDF-4 is HDF5 inside; NetCDF-3 classic files are not).
pub const EXTENSIONS: &[&str] = &["nc", "nc4", "h5", "hdf5", "he5", "hdf"];

/// The most bands taken from a three-dimensional variable.
pub const MAX_BANDS: u64 = 16;

/// Whether `name` has one of [`EXTENSIONS`].
pub fn is_name(name: &str) -> bool {
    let ext = std::path::Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    EXTENSIONS.contains(&ext.as_str())
}

/// Whether `bytes` start with the HDF5 signature (at 0, or 512, 1024, 2048… after a user block).
pub fn is_hdf5(bytes: &[u8]) -> bool {
    const SIG: &[u8] = b"\x89HDF\r\n\x1a\n";
    let mut at = 0usize;
    loop {
        match bytes.get(at..at + 8) {
            Some(b) if b == SIG => return true,
            Some(_) => at = if at == 0 { 512 } else { at * 2 },
            None => return false,
        }
    }
}

fn fail(name: &str, e: impl std::fmt::Display) -> DataError {
    DataError::Format(format!("{name}: {e}"))
}

/// Whether a datatype is a number this reader turns into `f32`.
fn numeric(dt: &Datatype) -> bool {
    matches!(dt, Datatype::FixedPoint { size: 1 | 2 | 4 | 8, .. } | Datatype::FloatingPoint { size: 4 | 8, .. })
}

/// GDAL-style name of a pixel type.
fn type_name(dt: &Datatype) -> &'static str {
    match dt {
        Datatype::FixedPoint { size: 1, signed: false, .. } => "UInt8",
        Datatype::FixedPoint { size: 1, signed: true, .. } => "Int8",
        Datatype::FixedPoint { size: 2, signed: false, .. } => "UInt16",
        Datatype::FixedPoint { size: 2, signed: true, .. } => "Int16",
        Datatype::FixedPoint { size: 4, signed: false, .. } => "UInt32",
        Datatype::FixedPoint { size: 4, signed: true, .. } => "Int32",
        Datatype::FixedPoint { size: 8, signed: false, .. } => "UInt64",
        Datatype::FixedPoint { size: 8, signed: true, .. } => "Int64",
        Datatype::FloatingPoint { size: 4, .. } => "Float32",
        _ => "Float64",
    }
}

/// A selection read as `f64`s, whatever the stored number type.
fn read_f64(ds: &Dataset, sel: &SliceInfo) -> std::result::Result<Vec<f64>, String> {
    macro_rules! as_f64 {
        ($t:ty) => {
            ds.read_slice::<$t>(sel).map(|a| a.iter().map(|v| *v as f64).collect()).map_err(|e| e.to_string())
        };
    }
    match ds.dtype() {
        Datatype::FixedPoint { size: 1, signed: false, .. } => as_f64!(u8),
        Datatype::FixedPoint { size: 1, signed: true, .. } => as_f64!(i8),
        Datatype::FixedPoint { size: 2, signed: false, .. } => as_f64!(u16),
        Datatype::FixedPoint { size: 2, signed: true, .. } => as_f64!(i16),
        Datatype::FixedPoint { size: 4, signed: false, .. } => as_f64!(u32),
        Datatype::FixedPoint { size: 4, signed: true, .. } => as_f64!(i32),
        Datatype::FixedPoint { size: 8, signed: false, .. } => as_f64!(u64),
        Datatype::FixedPoint { size: 8, signed: true, .. } => as_f64!(i64),
        Datatype::FloatingPoint { size: 4, .. } => as_f64!(f32),
        _ => as_f64!(f64),
    }
}

/// A whole one-dimensional variable as `f64`s.
fn read_1d(ds: &Dataset) -> Option<Vec<f64>> {
    (ds.ndim() == 1 && numeric(ds.dtype())).then(|| read_f64(ds, &SliceInfo::all(1)).ok()).flatten()
}

fn attr_f64(ds: &Dataset, name: &str) -> Option<f64> {
    ds.attribute(name).ok().and_then(|a| a.read_as_f64().ok()).filter(|v| v.is_finite())
}

fn attr_string(ds: &Dataset, name: &str) -> Option<String> {
    ds.attribute(name).ok().and_then(|a| a.read_string().ok())
}

/// Names of coordinate variables, never read as the raster itself.
fn is_coordinate_name(n: &str) -> bool {
    matches!(
        n.to_ascii_lowercase().as_str(),
        "lat" | "lon" | "latitude" | "longitude" | "x" | "y" | "time" | "lev" | "level" | "nav_lat" | "nav_lon" | "lat_bnds" | "lon_bnds"
            | "time_bnds" | "xdim" | "ydim"
    )
}

/// A variable found in the file, with the group that holds it and its path.
struct Found {
    path: String,
    ds: Dataset,
    group: Group,
}

/// Every candidate variable, depth first, children by name.
fn walk(group: &Group, prefix: &str, out: &mut Vec<Found>, depth: usize) {
    if depth > 32 {
        return;
    }
    let Ok((mut groups, mut datasets)) = group.members() else { return };
    datasets.sort_by(|a, b| a.name().cmp(b.name()));
    groups.sort_by(|a, b| a.name().cmp(b.name()));
    for ds in datasets {
        let path = format!("{prefix}/{}", ds.name());
        let dims = ds.shape();
        let usable = (2..=4).contains(&ds.ndim())
            && numeric(ds.dtype())
            && dims[dims.len() - 1] > 1
            && dims[dims.len() - 2] > 1
            && !is_coordinate_name(ds.name())
            && attr_string(&ds, "CLASS").is_none_or(|c| !c.starts_with("DIMENSION_SCALE"));
        if usable {
            out.push(Found { path, ds, group: group.clone() });
        }
    }
    for g in groups {
        let p = format!("{prefix}/{}", g.name());
        walk(&g, &p, out, depth + 1);
    }
}

/// The variable path out of GDAL's subdataset syntax (`HDF5:"f.h5"://grp/var`,
/// `NETCDF:"f.nc":var`) or a bare path, with GDAL's leading double slash made one.
fn variable_path(s: &str) -> String {
    let rest = match s.split_once(':') {
        Some((scheme, rest)) if scheme.eq_ignore_ascii_case("hdf5") || scheme.eq_ignore_ascii_case("netcdf") => {
            rest.rsplit_once(':').map_or(rest, |(_, v)| v)
        }
        _ => s,
    };
    let rest = rest.trim_start_matches('/');
    format!("/{rest}")
}

/// The names of every candidate variable in a file (`/group/var`), for a chooser.
pub fn variables(bytes: &[u8]) -> Result<Vec<String>> {
    let f = Hdf5File::from_bytes(bytes).map_err(|e| fail("HDF5", e))?;
    let root = f.root_group().map_err(|e| fail("HDF5", e))?;
    let mut found = Vec::new();
    walk(&root, "", &mut found, 0);
    Ok(found.into_iter().map(|f| f.path).collect())
}

/// Reads a NetCDF-4 or HDF5 file's variable `var` (the first, when `None`) as a raster.
pub fn read_bytes(file_name: &str, bytes: &[u8], var: Option<&str>) -> Result<Raster> {
    let stem = std::path::Path::new(file_name).file_stem().and_then(|s| s.to_str()).unwrap_or("raster");
    if !is_hdf5(bytes) {
        return Err(fail(
            file_name,
            "not an HDF5 or NetCDF-4 file (NetCDF-3 classic files and HDF4 files are read in the desktop app through GDAL)",
        ));
    }
    let f = Hdf5File::from_bytes(bytes).map_err(|e| fail(file_name, e))?;
    let root = f.root_group().map_err(|e| fail(file_name, e))?;
    let mut found = Vec::new();
    walk(&root, "", &mut found, 0);
    let count = found.len();
    let pick = match var {
        Some(v) => {
            let want = variable_path(v);
            let i = found
                .iter()
                .position(|f| f.path == want || f.path.replace(' ', "_") == want)
                .ok_or_else(|| fail(file_name, format!("no two- or three-dimensional numeric variable {want}")))?;
            found.swap_remove(i)
        }
        None => {
            if found.is_empty() {
                return Err(fail(file_name, "holds no two- or three-dimensional numeric variable to show"));
            }
            found.remove(0)
        }
    };
    let mut lost = Vec::new();
    if var.is_none() && count > 1 {
        lost.push(format!("the file holds {count} variables; {} is shown (open another by its path)", pick.path));
    }
    read_variable(&f, stem, &pick, lost)
}

fn read_variable(f: &Hdf5File, stem: &str, v: &Found, mut lost: Vec<String>) -> Result<Raster> {
    let ds = &v.ds;
    let dims = ds.shape().to_vec();
    let n = dims.len();
    let (h, w) = (dims[n - 2], dims[n - 1]);
    let (w32, h32) = (u32::try_from(w).map_err(|_| fail(stem, "too wide"))?, u32::try_from(h).map_err(|_| fail(stem, "too tall"))?);
    let band_count = match n {
        3 => dims[0].min(MAX_BANDS),
        _ => 1,
    };
    if n == 3 && dims[0] > MAX_BANDS {
        lost.push(format!("{} of its {} layers along its first dimension are shown", MAX_BANDS, dims[0]));
    }
    if n == 4 && (dims[0] > 1 || dims[1] > 1) {
        lost.push(format!("only the first step of its first two dimensions ({} × {}) is shown", dims[0], dims[1]));
    }
    let per_band = (w as usize).saturating_mul(h as usize);
    if per_band.saturating_mul(band_count as usize) > max_samples() {
        return Err(fail(stem, format!("{w} × {h} × {band_count} values are more than this device can hold")));
    }
    let fill = attr_f64(ds, "_FillValue").or_else(|| attr_f64(ds, "missing_value"));
    let scale = attr_f64(ds, "scale_factor");
    let offset = attr_f64(ds, "add_offset");
    let scaled = scale.is_some() || offset.is_some();
    let (scale, offset) = (scale.unwrap_or(1.0), offset.unwrap_or(0.0));
    let all = SliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 };
    let mut bands = Vec::with_capacity(band_count as usize);
    let mut rounded = vec![0u64; band_count as usize];
    for b in 0..band_count {
        let mut sel = Vec::with_capacity(n);
        if n >= 3 {
            sel.push(SliceInfoElem::Index(b));
        }
        if n == 4 {
            sel.push(SliceInfoElem::Index(0));
        }
        sel.push(all.clone());
        sel.push(all.clone());
        let raw = read_f64(ds, &SliceInfo { selections: sel }).map_err(|e| fail(&v.path, e))?;
        let data: Vec<f32> = raw
            .iter()
            .map(|&x| {
                if fill.is_some_and(|fv| x == fv) || !x.is_finite() {
                    f32::NAN
                } else {
                    let y = x * scale + offset;
                    let z = y as f32;
                    if f64::from(z) != y {
                        rounded[b as usize] += 1;
                    }
                    z
                }
            })
            .collect();
        bands.push(data);
    }
    let data_type = if scaled { "Float32" } else { type_name(ds.dtype()) };
    let name = format!("{stem} {}", v.path.trim_start_matches('/'));
    let place = cf_grid(&v.group, &dims, w, h).or_else(|| eos_grid(f, &v.path, w, h));
    let mut r = match place {
        Some((t, flip)) => {
            if flip {
                for band in &mut bands {
                    flip_rows(band, w as usize, h as usize);
                }
            }
            Raster::new(name, w32, h32, bands, t, Crs::LonLat, None, ColorHint::Gray, data_type)?
        }
        None => {
            lost.push("it gives no one-dimensional latitude/longitude it can be placed by, so it is shown unplaced".into());
            let mut r = Raster::new(name, w32, h32, bands, unreferenced(w32, h32), Crs::LonLat, None, ColorHint::Gray, data_type)?;
            r.georeferenced = false;
            r
        }
    };
    if n == 3 {
        for (i, band) in r.bands.iter_mut().enumerate() {
            band.name = format!("{} {}", v.path.rsplit('/').next().unwrap_or("band"), i);
        }
    }
    if let Some(units) = attr_string(ds, "units").filter(|u| !u.trim().is_empty()) {
        r.lost.push(format!("values are in {units}"));
    }
    r.lost.extend(lost);
    if rounded.iter().any(|c| *c > 0) {
        r.rounded = rounded;
    }
    r.source = Some(v.path.clone());
    Ok(r)
}

fn flip_rows(band: &mut [f32], w: usize, h: usize) {
    for row in 0..h / 2 {
        let (top, bottom) = band.split_at_mut((h - 1 - row) * w);
        top[row * w..row * w + w].swap_with_slice(&mut bottom[..w]);
    }
}

/// A regular axis's first value and step (`None` unless evenly spaced).
fn regular(axis: &[f64]) -> Option<(f64, f64)> {
    let n = axis.len();
    if n < 2 || axis.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let step = (axis[n - 1] - axis[0]) / (n - 1) as f64;
    let tol = step.abs() * 0.01 + 1e-9;
    (step != 0.0 && axis.windows(2).all(|p| ((p[1] - p[0]) - step).abs() <= tol)).then_some((axis[0], step))
}

/// A CF grid's transform from the one-dimensional latitude and longitude beside the variable,
/// and whether its rows run south to north (and so are flipped).
fn cf_grid(group: &Group, dims: &[u64], w: u64, h: u64) -> Option<(GeoTransform, bool)> {
    let _ = dims;
    let datasets = group.datasets().ok()?;
    let find = |names: &[&str], len: u64| {
        datasets.iter().find(|d| names.iter().any(|n| d.name().eq_ignore_ascii_case(n)) && d.ndim() == 1 && d.shape()[0] == len)
    };
    let lat = find(&["lat", "latitude", "y", "nav_lat", "ydim"], h)?;
    let lon = find(&["lon", "longitude", "x", "nav_lon", "xdim"], w)?;
    // `x`/`y` only when they are in degrees.
    for d in [lat, lon] {
        if matches!(d.name(), "x" | "y" | "X" | "Y") && !attr_string(d, "units").is_some_and(|u| u.to_ascii_lowercase().starts_with("degree")) {
            return None;
        }
    }
    let (lat0, dlat) = regular(&read_1d(lat)?)?;
    let (lon0, dlon) = regular(&read_1d(lon)?)?;
    if !(-91.0..=91.0).contains(&lat0) || !(-361.0..=361.0).contains(&lon0) {
        return None;
    }
    let flip = dlat > 0.0;
    let north = if flip { lat0 + dlat * (h - 1) as f64 } else { lat0 };
    let t = GeoTransform {
        origin_x: lon0 - dlon / 2.0,
        origin_y: north + dlat.abs() / 2.0,
        pixel_width: dlon,
        pixel_height: -dlat.abs(),
    };
    t.is_valid().then_some((t, flip))
}

/// Degrees from HDF-EOS's packed DDDMMMSSS.SS.
fn packed_dms(v: f64) -> f64 {
    let s = v.signum();
    let a = v.abs();
    let d = (a / 1e6).floor();
    let m = ((a - d * 1e6) / 1e3).floor();
    let sec = a - d * 1e6 - m * 1e3;
    s * (d + m / 60.0 + sec / 3600.0)
}

/// An HDF-EOS5 geographic grid's transform, from `/HDFEOS INFORMATION/StructMetadata.0`, for a
/// variable under `/HDFEOS/GRIDS/<grid>/`.
fn eos_grid(f: &Hdf5File, path: &str, w: u64, h: u64) -> Option<(GeoTransform, bool)> {
    let grid = path.strip_prefix("/HDFEOS/GRIDS/")?.split('/').next()?;
    let meta = f.dataset("/HDFEOS INFORMATION/StructMetadata.0").ok()?.read_string().ok()?;
    // The GROUP=GRID_n block whose GridName is this grid.
    let block = meta.split("GROUP=GRID_").find(|b| b.contains(&format!("GridName=\"{grid}\"")))?;
    let value = |key: &str| -> Option<&str> {
        let i = block.find(&format!("{key}="))?;
        let rest = &block[i + key.len() + 1..];
        Some(rest.split(['\n', '\r']).next()?.trim())
    };
    let proj = value("Projection")?;
    if !proj.contains("GCTP_GEO") {
        return None;
    }
    let pair = |key: &str| -> Option<(f64, f64)> {
        let v = value(key)?.trim_matches(|c| c == '(' || c == ')');
        let (a, b) = v.split_once(',')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    };
    let (ulx, uly) = pair("UpperLeftPointMtrs")?;
    let (lrx, lry) = pair("LowerRightMtrs")?;
    let (ulx, uly, lrx, lry) = (packed_dms(ulx), packed_dms(uly), packed_dms(lrx), packed_dms(lry));
    let t = GeoTransform { origin_x: ulx, origin_y: uly, pixel_width: (lrx - ulx) / w as f64, pixel_height: (lry - uly) / h as f64 };
    (t.is_valid() && t.pixel_height < 0.0).then_some((t, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_and_paths() {
        assert!(!is_hdf5(b"CDF\x01"));
        let mut b = vec![0u8; 520];
        b[512..520].copy_from_slice(b"\x89HDF\r\n\x1a\n");
        assert!(is_hdf5(&b));
        assert_eq!(variable_path("HDF5:\"a.h5\"://g/v"), "/g/v");
        assert_eq!(variable_path("NETCDF:\"a.nc\":air"), "/air");
        assert_eq!(variable_path("/HDFEOS/GRIDS/G/Data Fields/T"), "/HDFEOS/GRIDS/G/Data Fields/T");
        assert!((packed_dms(-180000000.0) + 180.0).abs() < 1e-9);
        assert!((packed_dms(90030000.0) - 90.5).abs() < 1e-9);
        let mut band = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        flip_rows(&mut band, 2, 3);
        assert_eq!(band, vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    /// Reads every `.h5`/`.nc`/`.he5` file in `$RUSTGIS_NCHDF_SAMPLES` (skipped when unset).
    #[test]
    fn sample_files() {
        let Ok(dir) = std::env::var("RUSTGIS_NCHDF_SAMPLES") else { return };
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if !is_name(&name) {
                continue;
            }
            let bytes = std::fs::read(&p).unwrap();
            // Every variable the chooser lists opens by its path, as Add does.
            if let Ok(vars) = variables(&bytes) {
                for v in &vars {
                    let r = crate::raster::read_bytes_page(&name, &bytes, Some(v), true);
                    eprintln!("  {name} {v}: {}", r.map_or_else(|e| format!("ERROR {e}"), |r| format!("{}x{} {}", r.width, r.height, r.name)));
                }
            }
            match read_bytes(&name, &bytes, None) {
                Ok(r) => eprintln!(
                    "{name}: {}x{}x{} {} georef={} t={:?} min={} max={} notes={:?}",
                    r.width,
                    r.height,
                    r.bands.len(),
                    r.data_type,
                    r.georeferenced,
                    r.transform,
                    r.bands[0].stats.min,
                    r.bands[0].stats.max,
                    r.lost
                ),
                Err(e) => eprintln!("{name}: ERROR {e}"),
            }
        }
    }
}
