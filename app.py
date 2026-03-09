from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class PipelineDefaults:
    # Defaults aligned with TIAToolbox patch prediction examples
    model_name: str = "resnet18-idars-tumour"
    patch_size: int = 224
    stride: int = 224
    patch_resolution: float = 0.5
    patch_units: str = "mpp"
    overview_resolution: float = 6.0
    overview_units: str = "mpp"
    min_mask_ratio: float = 0.5
    stain_method: str = "Reinhard"
    batch_size: int = 64


DEFAULTS = PipelineDefaults()


def _safe_import_tiatoolbox():
    # Import lazily so the app UI can render even if deps are missing.
    from tiatoolbox import data as tia_data  # noqa: WPS433
    from tiatoolbox.models.engine.patch_predictor import (  # noqa: WPS433
        PatchPredictor,
    )
    from tiatoolbox.tools import stainnorm  # noqa: WPS433
    from tiatoolbox.tools.patchextraction import (  # noqa: WPS433
        SlidingWindowPatchExtractor,
    )
    from tiatoolbox.wsicore.wsireader import WSIReader  # noqa: WPS433

    return tia_data, stainnorm, PatchPredictor, SlidingWindowPatchExtractor, WSIReader


def _normalize_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Expected RGB image, got shape={img.shape}.")
    return img[:, :, :3]


def _to_bool_mask(mask_img: np.ndarray) -> np.ndarray:
    if mask_img.dtype == bool:
        return mask_img
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    return mask_img > 0


def _make_overlay(
    rgb: np.ndarray,
    heatmap01: np.ndarray,
    *,
    alpha_max: float = 0.65,
    cmap_name: str = "inferno",
) -> np.ndarray:
    rgb = _normalize_uint8_rgb(rgb)
    h, w = rgb.shape[:2]
    if heatmap01.shape[:2] != (h, w):
        raise ValueError("Heatmap must match image size.")

    heatmap01 = np.clip(heatmap01.astype(np.float32), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(heatmap01)  # float [0..1], RGBA
    color = (rgba[:, :, :3] * 255.0).astype(np.float32)

    alpha = (heatmap01 * alpha_max).astype(np.float32)[:, :, None]
    base = rgb.astype(np.float32)
    out = base * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _batch_iter(seq: Iterable, batch_size: int):
    batch = []
    for item in seq:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _resolve_units(
    wsi_info: dict,
    requested_units: str,
    requested_resolution: float,
) -> tuple[str, float]:
    """Fallback to baseline if WSI lacks mpp metadata."""
    if requested_units != "mpp":
        return requested_units, requested_resolution
    mpp = wsi_info.get("mpp")
    if mpp is None or any(v is None for v in (mpp if isinstance(mpp, (list, tuple)) else [mpp])):
        # Use baseline if mpp is missing/unknown.
        return "baseline", 1.0
    return requested_units, requested_resolution


@st.cache_resource(show_spinner=False)
def _get_predictor(model_name: str, batch_size: int):
    _, _, PatchPredictor, _, _ = _safe_import_tiatoolbox()
    # TIAToolbox API varies slightly across versions; keep init minimal.
    return PatchPredictor(pretrained_model=model_name, batch_size=batch_size)


@st.cache_resource(show_spinner=False)
def _get_stain_normalizer(method_name: str):
    tia_data, stainnorm, _, _, _ = _safe_import_tiatoolbox()
    target_image = tia_data.stain_norm_target()
    normalizer = stainnorm.get_normalizer(method_name)
    normalizer.fit(_normalize_uint8_rgb(target_image))
    return normalizer


def _read_thumbnail(wsi, resolution: float, units: str) -> np.ndarray:
    thumb = wsi.slide_thumbnail(resolution=resolution, units=units)
    return _normalize_uint8_rgb(thumb)


def _read_tissue_mask_thumbnail(wsi, resolution: float, units: str) -> np.ndarray:
    # Prefer WSIReader.tissue_mask (TIAToolbox internal TissueMasker defaults).
    try:
        mask_reader = wsi.tissue_mask(method="otsu", resolution=resolution, units=units)
        mask = mask_reader.slide_thumbnail(resolution=resolution, units=units)
        return _to_bool_mask(mask)
    except Exception:
        # Fallback: run a coarse tissue mask via thumbnail + OtsuTissueMasker.
        from tiatoolbox.tools.tissuemask import OtsuTissueMasker  # noqa: WPS433

        thumb = _read_thumbnail(wsi, resolution=resolution, units=units)
        masker = OtsuTissueMasker()
        masks = masker.fit_transform([thumb])
        return _to_bool_mask(masks[0])


def _extract_and_predict(
    wsi_path: str,
    *,
    patch_size: int,
    stride: int,
    patch_resolution: float,
    patch_units: str,
    min_mask_ratio: float,
    predictor,
    stain_normalizer,
    device: str,
    apply_stainnorm: bool,
) -> pd.DataFrame:
    """Return a dataframe with patch top-left (x,y) and tumor probability."""
    _, _, _, SlidingWindowPatchExtractor, WSIReader = _safe_import_tiatoolbox()

    wsi = WSIReader.open(wsi_path)

    extractor = SlidingWindowPatchExtractor(
        wsi_path,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        input_mask="otsu",
        min_mask_ratio=min_mask_ratio,
        resolution=patch_resolution,
        units=patch_units,
    )._generate_location_df()

    locs = extractor.locations_df
    if locs is None or len(locs) == 0:
        return pd.DataFrame(columns=["x", "y", "tumor_prob"])

    xs = locs["x"].to_numpy(dtype=np.int64)
    ys = locs["y"].to_numpy(dtype=np.int64)

    tumor_probs: list[float] = []
    out_xs: list[int] = []
    out_ys: list[int] = []

    indices = np.arange(len(locs))
    progress = st.progress(0, text="Running patch extraction + inference…")

    # Conservative: run our own batching so memory usage stays bounded.
    for bi, batch_idx in enumerate(_batch_iter(indices.tolist(), DEFAULTS.batch_size)):
        patches = []
        batch_x = xs[batch_idx]
        batch_y = ys[batch_idx]

        for x, y in zip(batch_x, batch_y, strict=False):
            patch = wsi.read_rect(
                location=(int(x), int(y)),
                size=(patch_size, patch_size),
                resolution=patch_resolution,
                units=patch_units,
                coord_space="resolution",
            )
            patch = _normalize_uint8_rgb(patch)
            if apply_stainnorm:
                patch = stain_normalizer.transform(patch.copy())
                patch = _normalize_uint8_rgb(patch)
            patches.append(patch)

        # TIAToolbox versions differ: prefer `predict`, fallback to `run`.
        if hasattr(predictor, "predict"):
            out = predictor.predict(
                imgs=patches,
                mode="patch",
                return_probabilities=True,
            
            )
            probs = out.get("probabilities", None)
        else:
            out = predictor.run(
                images=np.asarray(patches),
                patch_mode=True,
                return_probabilities=True,
                device=device,
                output_type="dict",
            )
            probs = out.get("probabilities", None)

        if probs is None:
            raise RuntimeError(
                "Could not read probabilities from TIAToolbox PatchPredictor output."
            )

        probs = np.asarray(probs)
        if probs.ndim != 2 or probs.shape[0] != len(patches):
            raise RuntimeError(f"Unexpected probability tensor shape: {probs.shape}")

        # Assume binary [non-tumor, tumor]. If not, use max over non-background as tumor proxy.
        if probs.shape[1] >= 2:
            batch_tumor = probs[:, 1]
        else:
            batch_tumor = probs[:, 0]

        tumor_probs.extend(batch_tumor.astype(float).tolist())
        out_xs.extend(batch_x.astype(int).tolist())
        out_ys.extend(batch_y.astype(int).tolist())

        frac = min(1.0, (len(out_xs) / len(locs)))
        progress.progress(frac, text=f"Running patch extraction + inference… ({len(out_xs)}/{len(locs)})")

    progress.empty()
    return pd.DataFrame({"x": out_xs, "y": out_ys, "tumor_prob": tumor_probs})


def _compute_scores(tumor_probs: np.ndarray) -> tuple[float, float]:
    if tumor_probs.size == 0:
        return float("nan"), float("nan")
    tumor_probs = tumor_probs.astype(np.float32)
    burden = float(np.mean(tumor_probs))
    k = max(1, int(math.ceil(0.10 * tumor_probs.size)))
    topk = float(np.mean(np.sort(tumor_probs)[-k:]))
    return burden, topk


def _build_hotspot_heatmap(
    wsi,
    patch_df: pd.DataFrame,
    *,
    patch_size: int,
    patch_resolution: float,
    patch_units: str,
    overview_resolution: float,
    overview_units: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (overview_rgb, heatmap01) in overview pixel space."""
    overview = _read_thumbnail(wsi, resolution=overview_resolution, units=overview_units)
    oh, ow = overview.shape[:2]

    if len(patch_df) == 0:
        return overview, np.zeros((oh, ow), dtype=np.float32)

    dim_extract = wsi.slide_dimensions(patch_resolution, patch_units)  # (w,h)
    dim_overview = wsi.slide_dimensions(overview_resolution, overview_units)  # (w,h)
    sx = (dim_overview[0] / dim_extract[0]) if dim_extract[0] else 1.0
    sy = (dim_overview[1] / dim_extract[1]) if dim_extract[1] else 1.0

    pw = max(1, int(round(patch_size * sx)))
    ph = max(1, int(round(patch_size * sy)))

    heat = np.zeros((oh, ow), dtype=np.float32)
    xs = patch_df["x"].to_numpy(dtype=np.float32)
    ys = patch_df["y"].to_numpy(dtype=np.float32)
    ps = patch_df["tumor_prob"].to_numpy(dtype=np.float32)

    for x, y, p in zip(xs, ys, ps, strict=False):
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = min(ow, x0 + pw)
        y1 = min(oh, y0 + ph)
        if x1 <= 0 or y1 <= 0 or x0 >= ow or y0 >= oh:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        region = heat[y0:y1, x0:x1]
        np.maximum(region, p, out=region)

    # Smooth slightly for a nicer hotspot appearance.
    try:
        import cv2  # noqa: WPS433

        k = max(3, int(round(min(ow, oh) * 0.005)) | 1)  # odd kernel
        heat = cv2.GaussianBlur(heat, (k, k), 0)
    except Exception:
        pass

    heat = np.clip(heat, 0.0, 1.0)
    return overview, heat


def main():
    st.set_page_config(page_title="WSI Testing Dashboard", layout="wide")
    st.title("WSI Testing Dashboard (TIAToolbox)")

    with st.sidebar:
        st.subheader("Input")
        uploaded = st.file_uploader(
            "Upload a WSI (.svs, .tiff, .ndpi)",
            type=["svs", "tiff", "tif", "ndpi"],
        )
        local_path = st.text_input("…or local file path", value="")

        st.subheader("Defaults (TIA-aligned)")
        apply_stainnorm = st.toggle("Apply stain normalization", value=True)
        device = st.selectbox("Device", options=["cpu", "cuda"], index=0)

        with st.expander("Advanced (optional)", expanded=False):
            patch_size = st.number_input(
                "Patch size (px)",
                min_value=64,
                max_value=1024,
                value=DEFAULTS.patch_size,
                step=32,
            )
            stride = st.number_input(
                "Stride (px)",
                min_value=32,
                max_value=1024,
                value=DEFAULTS.stride,
                step=32,
            )
            min_mask_ratio = st.slider(
                "Min tissue ratio per patch",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS.min_mask_ratio),
                step=0.05,
            )
            overview_resolution = st.number_input(
                "Overview resolution (mpp)",
                min_value=1.0,
                max_value=20.0,
                value=float(DEFAULTS.overview_resolution),
                step=1.0,
            )

        run = st.button("Run pipeline", type="primary", use_container_width=True)

    if not run:
        st.info("Upload/select a WSI, then click **Run pipeline**.")
        return

    # Resolve input WSI path
    wsi_path: str | None = None
    tmp_dir: str | None = None
    if uploaded is not None:
        tmp_dir = tempfile.mkdtemp(prefix="wsi_dash_")
        wsi_path = str(Path(tmp_dir) / uploaded.name)
        with open(wsi_path, "wb") as f:
            f.write(uploaded.getbuffer())
    elif local_path.strip():
        wsi_path = local_path.strip().strip('"')

    if not wsi_path or not Path(wsi_path).exists():
        st.error("WSI path not found. Upload a file or provide a valid local path.")
        return

    try:
        tia_data, stainnorm, PatchPredictor, SlidingWindowPatchExtractor, WSIReader = (
            _safe_import_tiatoolbox()
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    # Open WSI
    with st.spinner("Opening WSI…"):
        wsi = WSIReader.open(wsi_path)
        wsi_info = wsi.info.as_dict() if hasattr(wsi.info, "as_dict") else dict(wsi.info)

    # Resolve units/resolutions robustly (fallback to baseline if missing mpp metadata)
    patch_units, patch_resolution = _resolve_units(
        wsi_info, DEFAULTS.patch_units, DEFAULTS.patch_resolution
    )
    overview_units, overview_resolution_resolved = _resolve_units(
        wsi_info, DEFAULTS.overview_units, float(overview_resolution)
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("WSI metadata")
        meta_show = {
            "vendor": wsi_info.get("vendor"),
            "objective_power": wsi_info.get("objective_power"),
            "mpp": wsi_info.get("mpp"),
            "level_count": wsi_info.get("level_count"),
            "slide_dimensions (baseline)": wsi_info.get("slide_dimensions"),
        }
        st.json(meta_show)

    with col2:
        st.subheader("Thumbnail")
        with st.spinner("Generating thumbnail…"):
            thumb = _read_thumbnail(wsi, resolution=overview_resolution_resolved, units=overview_units)
        st.image(thumb, caption=f"Thumbnail @ {overview_resolution_resolved} {overview_units}", use_container_width=True)

    st.subheader("Tissue segmentation")
    with st.spinner("Computing tissue mask…"):
        tissue_mask = _read_tissue_mask_thumbnail(
            wsi, resolution=overview_resolution_resolved, units=overview_units
        )
    mask_vis = (tissue_mask.astype(np.uint8) * 255)
    st.image(mask_vis, caption="Tissue mask (otsu)", use_container_width=True)

    st.subheader("Stain normalization preview")
    norm_preview = None
    try:
        stain_normalizer = _get_stain_normalizer(DEFAULTS.stain_method)
        if apply_stainnorm:
            norm_preview = stain_normalizer.transform(thumb.copy())
            norm_preview = _normalize_uint8_rgb(norm_preview)
    except Exception as e:
        stain_normalizer = None
        st.warning(f"Stain normalization unavailable: {e}")

    if apply_stainnorm and norm_preview is not None:
        st.image(norm_preview, caption=f"{DEFAULTS.stain_method} normalized preview", use_container_width=True)
    else:
        st.caption("Stain normalization is disabled (or unavailable).")

    st.subheader("Patch inference (tumor vs normal)")
    with st.spinner("Loading pretrained model…"):
        predictor = _get_predictor(DEFAULTS.model_name, DEFAULTS.batch_size)

    if stain_normalizer is None:
        apply_stainnorm = False

    with st.spinner("Extracting patches and running inference…"):
        patch_df = _extract_and_predict(
            wsi_path,
            patch_size=int(patch_size),
            stride=int(stride),
            patch_resolution=float(patch_resolution),
            patch_units=str(patch_units),
            min_mask_ratio=float(min_mask_ratio),
            predictor=predictor,
            stain_normalizer=stain_normalizer,
            device=device,
            apply_stainnorm=bool(apply_stainnorm),
        )

    st.write(f"Patches analysed: **{len(patch_df)}**")
    if len(patch_df) == 0:
        st.warning("No tissue patches were found with the current settings.")
        return

    burden, hotspot = _compute_scores(patch_df["tumor_prob"].to_numpy())
    c1, c2 = st.columns(2)
    c1.metric("Risk score: Tumor burden (mean prob)", f"{burden:.4f}")
    c2.metric("Risk score: Hotspot Top-10% (mean prob)", f"{hotspot:.4f}")

    st.subheader("Hotspot map")
    with st.spinner("Building hotspot overlay…"):
        overview, heat = _build_hotspot_heatmap(
            wsi,
            patch_df,
            patch_size=int(patch_size),
            patch_resolution=float(patch_resolution),
            patch_units=str(patch_units),
            overview_resolution=float(overview_resolution_resolved),
            overview_units=str(overview_units),
        )
        overlay = _make_overlay(overview, heat, alpha_max=0.70, cmap_name="inferno")

    oc1, oc2 = st.columns([1, 1])
    oc1.image(overview, caption="Overview", use_container_width=True)
    oc2.image(overlay, caption="Overview + hotspot overlay", use_container_width=True)

    st.subheader("Patch table")
    st.dataframe(
        patch_df.sort_values("tumor_prob", ascending=False).head(500),
        use_container_width=True,
        height=300,
    )

    csv_bytes = patch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download patch predictions (CSV)",
        data=csv_bytes,
        file_name=f"{Path(wsi_path).stem}_patch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Cleanup temp upload dir if used (best-effort; don't fail the app on Windows locks).
    if tmp_dir:
        try:
            for root, _, files in os.walk(tmp_dir, topdown=False):
                for name in files:
                    Path(root, name).unlink(missing_ok=True)
                Path(root).rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()

