#!/usr/bin/env python3
"""
VLASS full-map standalone stacker for local NRAO /stash products.

This script combines full VLASS maps from the local Lustre/STASH cache, outside
of the cutout server. It discovers matching tile/source-id products across VLASS
epochs and runs CASA-style regrid/smooth + RMS-weighted stacking.

Base paths
----------
STASH_QL_BASE = /stash/projects/vlass/cache/quicklook
STASH_SE_BASE = /stash/projects/vlass/cache/se_continuum_imaging

Default output products
-----------------------
By default, only the radio-beam/common-beam RMS-weighted mean map is produced:
  VLASS.ce.<tile>.<source>.v1.I.image.pbcor.fits
  VLASS.ce.<tile>.<source>.v1.I.alpha.fits

Product code convention
-----------------------
  ce    = common/radio-beam weighted mean
  cem   = common/radio-beam weighted median
  cel   = largest-beam weighted mean
  celm  = largest-beam weighted median
  ces   = survey-defined-beam weighted mean   [disabled by default]
  cesm  = survey-defined-beam weighted median [disabled by default]

Notes
-----
- Intensity input uses: *I.iter3.image.pbcor.tt0.subim.fits
- SE alpha input uses either combined tt1/tt0 by default, or *I.iter3.alpha.subim.fits when --spx-rms-weighted is set.
- RMS weights use:      *I.iter3.image.pbcor.tt0.rms.subim.fits unless --use-local-rms is set
- For QL, alpha is skipped because standard QL directories do not contain alpha maps.
- Source matching is by (tile_id, J-source-id without final .vN), e.g.
  T10t20 + J124159-003000.06.2048.
"""

from __future__ import annotations

import argparse
import glob
import html
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from scipy.ndimage import uniform_filter
from spectral_cube import SpectralCube
from radio_beam import Beam, Beams
import astropy.units as u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from casatasks import importfits, exportfits, imregrid, imsmooth, imhead, immath, imdev


# =============================================================================
# USER SETTINGS / OUTPUT SWITCHES
# =============================================================================
# Current requested default: ONLY radio-beam/common-beam weighted mean = True.
# Everything else is available, but off by default.

MAKE_RADIO_BEAM_MEAN_MAP = True
MAKE_RADIO_BEAM_MEDIAN_MAP = False

MAKE_LARGEST_BEAM_MEAN_MAP = False
MAKE_LARGEST_BEAM_MEDIAN_MAP = False

MAKE_SURVEY_DEFINED_BEAM_MEAN_MAP = False
MAKE_SURVEY_DEFINED_BEAM_MEDIAN_MAP = False

# Survey-defined beam option. This is only used if one of the survey-defined
# beam switches above is True.
SURVEY_DEFINED_BMAJ_ARCSEC = 2.5
SURVEY_DEFINED_BMIN_ARCSEC = 2.5
SURVEY_DEFINED_BPA_DEG = 0.0

# Delta images: combined map minus each matched input map. Default requested False.
MAKE_DELTA_IMAGES = True

# QA report image cutout size around brightest finite pixel.
QA_CUTOUT_SIZE_PIX = 50

# CASA imdev local RMS map for each generated combined map.
MAKE_IMDEV_RMS_MAPS = True

# Spectral-index output mode for SE alpha products.
# Default requested behavior: derive alpha from combined tt1 / combined tt0.
# If SPX_RMS_WEIGHTED=True, keep the older/direct alpha-map weighted-combination path.
SPX_TT1_TT0 = True
SPX_RMS_WEIGHTED = False
SPX_MASK_THRESHOLD = 3.0

# Final product storage. Default keeps the separate/source-directory style.
# If KEEP_IN_ONE_FOLDER=True, final FITS products are additionally copied to one flat folder.
KEEP_IN_ONE_FOLDER = False
KEEP_SEPARATE_AREA = True

IMDEV_GRID = [10, 10]
IMDEV_XLENGTH = "60arcsec"
IMDEV_YLENGTH = "60arcsec"
IMDEV_INTERP = "cubic"
IMDEV_STATTYPE = "xmadm"
IMDEV_STATALG = "chauvenet"
IMDEV_ZSCORE = -1
IMDEV_MAXITER = -1


# =============================================================================
# STASH paths
# =============================================================================
STASH_QL_BASE = Path("/stash/projects/vlass/cache/quicklook")
STASH_SE_BASE = Path("/stash/projects/vlass/cache/se_continuum_imaging")

FREQ_GHZ_DEFAULT = 3.0
J_SOURCE_RE = re.compile(r"(J\d{6}[+-]\d{6}\.\d{2}\.\d{4})(?:\.v\d+)?")
TIER_RE = re.compile(r"^(T\d{2})t\d{2}$")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Discover and combine full VLASS maps from local /stash cache."
    )
    ap.add_argument("--survey", choices=["SE", "QL", "both"], default="SE",
                    help="Which VLASS map family to combine. Default: SE")
    ap.add_argument("--product", choices=["intensity", "alpha", "both"], default="intensity",
                    help="Map product to combine. QL alpha is skipped. Default: intensity")
    ap.add_argument("--epochs", default="",
                    help="Comma/space list of epochs, e.g. 1.2,2.1,2.2,3.1,3.2 or VLASS2.2. Empty means all available.")
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="Specific tile IDs, e.g. --tiles T10t20 T08t13. If omitted, tile scan is controlled by --num-tiers.")
    ap.add_argument("--num-tiers", type=int, default=0,
                    help="Number of tier prefixes to process. 0 means all tiers. 1 means first tier prefix only, e.g. T01*. Ignored when --tiles is supplied.")
    ap.add_argument("--min-maps", type=int, default=2,
                    help="Minimum number of epoch maps needed before stacking. Default: 2")
    ap.add_argument("--max-fields", type=int, default=0,
                    help="Debug limit on number of tile/source groups to stack. 0 means no limit.")
    ap.add_argument("--source-id", default="",
                    help="Optional single source-id filter, e.g. J124159-003000.06.2048. Final .vN is ignored if supplied.")
    ap.add_argument("--input-dir", default="",
                    help="Optional test mode: discover one-folder local inputs instead of scanning /stash. The folder may contain tt0, tt1, alpha, and optional rms FITS files.")
    ap.add_argument("--make-radio-beam-median", action="store_true",
                    help="Also make the common/radio-beam weighted median product for this run.")
    ap.add_argument("--spx-tt1-tt0", dest="spx_tt1_tt0", action="store_true", default=SPX_TT1_TT0,
                    help="For SE alpha products, derive spectral-index maps as combined tt1 / combined tt0. Default: True")
    ap.add_argument("--no-spx-tt1-tt0", dest="spx_tt1_tt0", action="store_false",
                    help="Disable tt1/tt0 spectral-index derivation.")
    ap.add_argument("--spx-rms-weighted", action="store_true", default=SPX_RMS_WEIGHTED,
                    help="Keep/use the older direct RMS-weighted combination of existing alpha maps instead of tt1/tt0 derivation. Default: False")
    ap.add_argument("--keep-in-one-folder", action="store_true", default=KEEP_IN_ONE_FOLDER,
                    help="Also copy final stacked products into one flat folder under --outdir/all_stacked_products.")
    ap.add_argument("--keep-separate-area", action="store_true", default=KEEP_SEPARATE_AREA,
                    help="Also copy final products to a stash-like separate directory tree under --outdir/stash_like_outputs. Default: True")
    ap.add_argument("--no-keep-separate-area", dest="keep_separate_area", action="store_false",
                    help="Do not make the stash-like separate-directory copy.")
    ap.add_argument("--outdir", default="vlass_fullmap_stacks",
                    help="Output directory. Default: vlass_fullmap_stacks")
    ap.add_argument("--out-prefix", default="",
                    help="Optional prefix added to every output stack filename.")
    ap.add_argument("--template-hint", default="VLASS2",
                    help="Preferred template epoch string. Default: VLASS2; falls back to first input.")
    ap.add_argument("--use-local-rms", action="store_true",
                    help="Use running local RMS maps instead of archive tt0.rms FITS maps.")
    ap.add_argument("--rms-box", type=int, default=10,
                    help="Running RMS box size in pixels when --use-local-rms is set. Default: 10")
    ap.add_argument("--freq-ghz", type=float, default=FREQ_GHZ_DEFAULT,
                    help="Frequency used for Jy/beam <-> K conversion. Default: 3.0")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print discovered groups; do not run CASA stacking.")
    ap.add_argument("--keep-work", action="store_true",
                    help="Keep temporary CASA working directories for debugging.")
    return ap.parse_args()


def normalize_epochs(epoch_text: str, survey: str) -> Optional[List[str]]:
    if not epoch_text.strip():
        return None
    raw = [x.strip() for x in re.split(r"[,\s]+", epoch_text.strip()) if x.strip()]
    out = []
    for e in raw:
        if e.upper().startswith("VLASS"):
            out.append(e)
        else:
            out.append("VLASS" + e)
    return out


def available_epochs(base: Path, wanted: Optional[List[str]]) -> List[str]:
    eps = sorted([p.name for p in base.glob("VLASS*") if p.is_dir()])
    if wanted is None:
        return eps
    wanted_set = set(wanted)
    return [e for e in eps if e in wanted_set]


def selected_tiles_for_epochs(base: Path, epochs: Sequence[str], tiles: Optional[List[str]], num_tiers: int) -> List[str]:
    if tiles:
        return sorted(set(tiles))

    all_tiles = set()
    for ep in epochs:
        ep_dir = base / ep
        for p in ep_dir.glob("T??t??"):
            if p.is_dir():
                all_tiles.add(p.name)
    sorted_tiles = sorted(all_tiles)

    if num_tiers and num_tiers > 0:
        tiers = []
        for t in sorted_tiles:
            m = TIER_RE.match(t)
            if m and m.group(1) not in tiers:
                tiers.append(m.group(1))
        keep_tiers = set(tiers[:num_tiers])
        sorted_tiles = [t for t in sorted_tiles if TIER_RE.match(t) and TIER_RE.match(t).group(1) in keep_tiers]

    return sorted_tiles


# =============================================================================
# Discovery
# =============================================================================
def extract_source_id(path_or_name: str) -> Optional[str]:
    m = J_SOURCE_RE.search(str(path_or_name))
    if not m:
        return None
    return m.group(1)


def first_existing(patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0])
    return None


def normalize_source_id_filter(source_id: str) -> str:
    s = (source_id or "").strip()
    if not s:
        return ""
    found = extract_source_id(s)
    return found if found else s


def discover_one_survey_product(survey: str,
                                product: str,
                                epochs: Sequence[str],
                                tiles: Sequence[str]) -> Dict[Tuple[str, str], List[Dict[str, Path]]]:
    """Return groups keyed by (tile_id, source_id)."""
    groups: Dict[Tuple[str, str], List[Dict[str, Path]]] = {}

    if survey == "SE":
        base = STASH_SE_BASE
        for ep in epochs:
            for tile in tiles:
                tile_dir = base / ep / tile
                if not tile_dir.is_dir():
                    continue
                for src_dir in sorted(tile_dir.glob(f"{ep}.se.{tile}.J*.v*")):
                    if not src_dir.is_dir():
                        continue
                    source_id = extract_source_id(src_dir.name)
                    if source_id is None:
                        continue

                    if product == "intensity":
                        map_path = first_existing([
                            str(src_dir / f"{src_dir.name}.I.iter3.image.pbcor.tt0.subim.fits"),
                            str(src_dir / "*I.iter3.image.pbcor.tt0.subim.fits"),
                        ])
                    elif product == "alpha":
                        map_path = first_existing([
                            str(src_dir / f"{src_dir.name}.I.iter3.alpha.subim.fits"),
                            str(src_dir / "*I.iter3.alpha.subim.fits"),
                        ])
                    else:
                        continue

                    rms_path = first_existing([
                        str(src_dir / f"{src_dir.name}.I.iter3.image.pbcor.tt0.rms.subim.fits"),
                        str(src_dir / "*I.iter3.image.pbcor.tt0.rms.subim.fits"),
                    ])

                    if map_path is None:
                        continue
                    groups.setdefault((tile, source_id), []).append({
                        "survey": survey,
                        "epoch": ep,
                        "tile": tile,
                        "source_id": source_id,
                        "map": map_path,
                        "rms": rms_path,
                        "src_dir": src_dir,
                    })

    elif survey == "QL":
        if product == "alpha":
            return groups
        base = STASH_QL_BASE
        for ep in epochs:
            for tile in tiles:
                tile_dir = base / ep / tile
                qa_dir = base / ep / "QA_REJECTED"
                src_dirs = []
                if tile_dir.is_dir():
                    src_dirs.extend(sorted(tile_dir.glob(f"{ep}.ql.{tile}.J*.v*")))
                if qa_dir.is_dir():
                    src_dirs.extend(sorted(qa_dir.glob(f"{ep}.ql.{tile}.J*.v*")))

                for src_dir in src_dirs:
                    if not src_dir.is_dir():
                        continue
                    source_id = extract_source_id(src_dir.name)
                    if source_id is None:
                        continue
                    map_path = first_existing([
                        str(src_dir / f"{src_dir.name}.I.iter1.image.pbcor.tt0.subim.fits"),
                        str(src_dir / "*I.iter1.image.pbcor.tt0.subim.fits"),
                        str(src_dir / "*I.iter3.image.pbcor.tt0.subim.fits"),
                    ])
                    rms_path = first_existing([
                        str(src_dir / "*I.iter1.image.pbcor.tt0.rms.subim.fits"),
                        str(src_dir / "*I.iter3.image.pbcor.tt0.rms.subim.fits"),
                    ])
                    if map_path is None:
                        continue
                    groups.setdefault((tile, source_id), []).append({
                        "survey": survey,
                        "epoch": ep,
                        "tile": tile,
                        "source_id": source_id,
                        "map": map_path,
                        "rms": rms_path,
                        "src_dir": src_dir,
                    })
    return groups


def discover_spectral_index_tt1_tt0(epochs: Sequence[str],
                                    tiles: Sequence[str]) -> Dict[Tuple[str, str], List[Dict[str, Path]]]:
    """Return SE groups for alpha maps derived as combined tt1 / combined tt0."""
    groups: Dict[Tuple[str, str], List[Dict[str, Path]]] = {}
    base = STASH_SE_BASE
    for ep in epochs:
        for tile in tiles:
            tile_dir = base / ep / tile
            if not tile_dir.is_dir():
                continue
            for src_dir in sorted(tile_dir.glob(f"{ep}.se.{tile}.J*.v*")):
                if not src_dir.is_dir():
                    continue
                source_id = extract_source_id(src_dir.name)
                if source_id is None:
                    continue
                tt0_path = first_existing([
                    str(src_dir / f"{src_dir.name}.I.iter3.image.pbcor.tt0.subim.fits"),
                    str(src_dir / "*I.iter3.image.pbcor.tt0.subim.fits"),
                ])
                tt1_path = first_existing([
                    str(src_dir / f"{src_dir.name}.I.iter3.image.pbcor.tt1.subim.fits"),
                    str(src_dir / "*I.iter3.image.pbcor.tt1.subim.fits"),
                ])
                tt0_rms_path = first_existing([
                    str(src_dir / f"{src_dir.name}.I.iter3.image.pbcor.tt0.rms.subim.fits"),
                    str(src_dir / "*I.iter3.image.pbcor.tt0.rms.subim.fits"),
                ])
                tt1_rms_path = first_existing([
                    str(src_dir / f"{src_dir.name}.I.iter3.image.pbcor.tt1.rms.subim.fits"),
                    str(src_dir / "*I.iter3.image.pbcor.tt1.rms.subim.fits"),
                ])
                if tt0_path is None or tt1_path is None:
                    continue
                groups.setdefault((tile, source_id), []).append({
                    "survey": "SE",
                    "epoch": ep,
                    "tile": tile,
                    "source_id": source_id,
                    "map": tt0_path,
                    "tt1": tt1_path,
                    "rms": tt0_rms_path,
                    "tt1_rms": tt1_rms_path,
                    "src_dir": src_dir,
                    "spx_mode": "tt1_tt0",
                })
    return groups


def _epoch_from_name(path_or_name: str) -> Optional[str]:
    m = re.search(r"(VLASS\d+\.\d+)", str(path_or_name))
    return m.group(1) if m else None


def _tile_from_name(path_or_name: str) -> Optional[str]:
    m = re.search(r"(T\d{2}t\d{2})", str(path_or_name))
    return m.group(1) if m else None


def discover_one_folder_product(input_dir: Path,
                                survey: str,
                                product: str,
                                wanted_epochs: Optional[Sequence[str]] = None,
                                wanted_tiles: Optional[Sequence[str]] = None) -> Dict[Tuple[str, str], List[Dict[str, Path]]]:
    """Return groups keyed by (tile_id, source_id) from one local folder.

    This is test mode only. It does not scan /stash.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Path]]] = {}
    input_dir = Path(input_dir).expanduser().resolve()
    epoch_set = set(wanted_epochs or [])
    tile_set = set(wanted_tiles or [])

    if product == "intensity":
        pats = ["*I.iter3.image.pbcor.tt0.subim.fits", "*I.iter1.image.pbcor.tt0.subim.fits"]
    elif product == "alpha":
        pats = ["*I.iter3.alpha.subim.fits", "*I.iter1.alpha.subim.fits"]
    else:
        return groups

    candidates: List[Path] = []
    for pat in pats:
        candidates.extend(sorted(input_dir.glob(pat)))

    for map_path in sorted(set(candidates)):
        epoch = _epoch_from_name(map_path.name)
        tile = _tile_from_name(map_path.name)
        source_id = extract_source_id(map_path.name)
        if epoch is None or tile is None or source_id is None:
            continue
        if epoch_set and epoch not in epoch_set:
            continue
        if tile_set and tile not in tile_set:
            continue

        if product == "intensity":
            rms_path = Path(str(map_path).replace(".tt0.subim.fits", ".tt0.rms.subim.fits"))
        else:
            rms_path = Path(str(map_path).replace(".alpha.subim.fits", ".alpha.error.subim.fits"))
        if not rms_path.exists():
            rms_path = None

        groups.setdefault((tile, source_id), []).append({
            "survey": survey,
            "epoch": epoch,
            "tile": tile,
            "source_id": source_id,
            "map": map_path,
            "rms": rms_path,
            "src_dir": input_dir,
            "local_input_dir": input_dir,
        })
    return groups


def discover_one_folder_spectral_index_tt1_tt0(input_dir: Path,
                                                wanted_epochs: Optional[Sequence[str]] = None,
                                                wanted_tiles: Optional[Sequence[str]] = None) -> Dict[Tuple[str, str], List[Dict[str, Path]]]:
    """Return local one-folder SE groups for alpha derived as combined tt1 / combined tt0."""
    groups: Dict[Tuple[str, str], List[Dict[str, Path]]] = {}
    input_dir = Path(input_dir).expanduser().resolve()
    epoch_set = set(wanted_epochs or [])
    tile_set = set(wanted_tiles or [])

    candidates = sorted(input_dir.glob("*I.iter3.image.pbcor.tt0.subim.fits"))
    candidates += sorted(input_dir.glob("*I.iter1.image.pbcor.tt0.subim.fits"))

    for tt0_path in sorted(set(candidates)):
        epoch = _epoch_from_name(tt0_path.name)
        tile = _tile_from_name(tt0_path.name)
        source_id = extract_source_id(tt0_path.name)
        if epoch is None or tile is None or source_id is None:
            continue
        if epoch_set and epoch not in epoch_set:
            continue
        if tile_set and tile not in tile_set:
            continue

        tt1_path = Path(str(tt0_path).replace(".tt0.subim.fits", ".tt1.subim.fits"))
        tt0_rms_path = Path(str(tt0_path).replace(".tt0.subim.fits", ".tt0.rms.subim.fits"))
        tt1_rms_path = Path(str(tt0_path).replace(".tt0.subim.fits", ".tt1.rms.subim.fits"))
        if not tt1_path.exists():
            continue
        if not tt0_rms_path.exists():
            tt0_rms_path = None
        if not tt1_rms_path.exists():
            tt1_rms_path = None

        groups.setdefault((tile, source_id), []).append({
            "survey": "SE",
            "epoch": epoch,
            "tile": tile,
            "source_id": source_id,
            "map": tt0_path,
            "tt1": tt1_path,
            "rms": tt0_rms_path,
            "tt1_rms": tt1_rms_path,
            "src_dir": input_dir,
            "local_input_dir": input_dir,
            "spx_mode": "tt1_tt0",
        })
    return groups


# =============================================================================
# Small helpers
# =============================================================================
def rm_if_exists(path: str | Path) -> None:
    path = str(path)
    if os.path.exists(path):
        os.system("rm -rf " + path)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)


def force_fits_to_J2000_inplace(fits_path: str | Path) -> None:
    with fits.open(fits_path, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["RADESYS"] = "FK5"
        hdr["EQUINOX"] = 2000.0
        if "RADECSYS" in hdr:
            del hdr["RADECSYS"]
        hdul.flush()


def read_fits_2d(path: str | Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as h:
        data = np.squeeze(h[0].data).astype(np.float32)
        hdr = h[0].header.copy()
    return data, hdr


def robust_rms(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float64)
    good = np.isfinite(arr)
    if not np.any(good):
        return float("nan")
    clipped = sigma_clip(arr[good], sigma=3.0, maxiters=5)
    vals = np.asarray(clipped, dtype=np.float64)
    if hasattr(clipped, "mask"):
        vals = vals[~clipped.mask]
    if vals.size == 0:
        return float("nan")
    return float(np.std(vals))


def image_stats(path: str | Path) -> Dict[str, float | str | int]:
    data, hdr = read_fits_2d(path)
    finite = np.isfinite(data)
    vals = data[finite]
    stats: Dict[str, float | str | int] = {
        "file": str(path),
        "shape": f"{data.shape[0]} x {data.shape[1]}" if data.ndim == 2 else str(data.shape),
        "nfinite": int(vals.size),
        "finite_frac": float(vals.size / data.size) if data.size else float("nan"),
        "rms": robust_rms(data),
        "max": float(np.nanmax(data)) if vals.size else float("nan"),
        "mean": float(np.nanmean(data)) if vals.size else float("nan"),
        "median": float(np.nanmedian(data)) if vals.size else float("nan"),
        "bmaj_as": float(hdr.get("BMAJ", np.nan)) * 3600.0,
        "bmin_as": float(hdr.get("BMIN", np.nan)) * 3600.0,
        "bpa_deg": float(hdr.get("BPA", np.nan)),
        "bunit": str(hdr.get("BUNIT", "")),
    }
    try:
        w = WCS(hdr).celestial
        pix = np.sqrt(abs(w.proj_plane_pixel_area())) * 3600.0
        stats["pixscale_as"] = float(pix)
    except Exception:
        stats["pixscale_as"] = float("nan")
    return stats


def local_rms(img: np.ndarray, win: int) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    good = np.isfinite(img)
    if not np.any(good):
        return np.full_like(img, np.nan, dtype=np.float32)

    fill = np.where(good, img, 0.0).astype(np.float32, copy=False)
    w = good.astype(np.float32, copy=False)

    wsum = uniform_filter(w, win)
    mean = uniform_filter(fill, win)
    mean2 = uniform_filter(fill * fill, win)

    valid = wsum > 0
    out = np.full_like(fill, np.nan, dtype=np.float32)
    if np.any(valid):
        mean_loc = np.zeros_like(fill, dtype=np.float32)
        mean2_loc = np.zeros_like(fill, dtype=np.float32)
        mean_loc[valid] = mean[valid] / wsum[valid]
        mean2_loc[valid] = mean2[valid] / wsum[valid]
        var = mean2_loc - mean_loc**2
        var[var < 0] = 0
        out[valid] = np.sqrt(var[valid]).astype(np.float32, copy=False)
    return out


def get_beam_arcsec(imagename: str) -> Tuple[float, float, float]:
    bmaj = imhead(imagename=imagename, mode="get", hdkey="BMAJ")["value"]
    bmin = imhead(imagename=imagename, mode="get", hdkey="BMIN")["value"]
    bpa = imhead(imagename=imagename, mode="get", hdkey="BPA")["value"]
    return bmaj, bmin, bpa


def jybeam_to_K(in_image: str, out_image: str, freq_ghz: float) -> str:
    bmaj_as, bmin_as, _ = get_beam_arcsec(in_image)
    factor = 1.222e6 / (freq_ghz**2 * bmaj_as * bmin_as)
    rm_if_exists(out_image)
    immath(imagename=in_image, mode="evalexpr", expr=f"{factor}*IM0", outfile=out_image)
    imhead(imagename=out_image, mode="put", hdkey="bunit", hdvalue="K")
    return out_image


def K_to_jybeam(in_image: str, out_image: str, freq_ghz: float) -> str:
    bmaj_as, bmin_as, _ = get_beam_arcsec(in_image)
    factor = (freq_ghz**2 * bmaj_as * bmin_as) / 1.222e6
    rm_if_exists(out_image)
    immath(imagename=in_image, mode="evalexpr", expr=f"{factor}*IM0", outfile=out_image)
    imhead(imagename=out_image, mode="put", hdkey="bunit", hdvalue="Jy/beam")
    return out_image


def choose_template(input_fits: Sequence[Path], template_hint: str) -> Path:
    for f in input_fits:
        if template_hint and template_hint in str(f):
            return f
    return input_fits[0]


def get_output_products() -> List[Dict[str, object]]:
    products = []
    if MAKE_RADIO_BEAM_MEAN_MAP:
        products.append({"code": "ce", "label": "Radio beam weighted mean", "beam_mode": "common", "stat": "mean"})
    if MAKE_RADIO_BEAM_MEDIAN_MAP:
        products.append({"code": "cem", "label": "Radio beam weighted median", "beam_mode": "common", "stat": "median"})
    if MAKE_LARGEST_BEAM_MEAN_MAP:
        products.append({"code": "cel", "label": "Largest beam weighted mean", "beam_mode": "largest", "stat": "mean"})
    if MAKE_LARGEST_BEAM_MEDIAN_MAP:
        products.append({"code": "celm", "label": "Largest beam weighted median", "beam_mode": "largest", "stat": "median"})
    if MAKE_SURVEY_DEFINED_BEAM_MEAN_MAP:
        products.append({"code": "ces", "label": "Survey-defined beam weighted mean", "beam_mode": "survey", "stat": "mean"})
    if MAKE_SURVEY_DEFINED_BEAM_MEDIAN_MAP:
        products.append({"code": "cesm", "label": "Survey-defined beam weighted median", "beam_mode": "survey", "stat": "median"})
    return products


def combined_filename(tile: str, source_id: str, code: str, product: str) -> str:
    product_tail = "image.pbcor" if product == "intensity" else "alpha"
    return f"VLASS.{code}.{tile}.{source_id}.v1.I.{product_tail}.fits"


def beam_to_tuple(beam) -> Tuple[float, float, float]:
    return (
        float(beam.major.to_value("arcsec")),
        float(beam.minor.to_value("arcsec")),
        float(beam.pa.to_value("deg")),
    )


def update_combined_header(fits_path: Path,
                           records: Sequence[Dict[str, Path]],
                           input_beams: Sequence[Tuple[float, float, float]],
                           final_beam: Tuple[float, float, float],
                           product_code: str,
                           product_label: str,
                           product: str,
                           weight_mode: str) -> None:
    with fits.open(fits_path, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["RADESYS"] = "FK5"
        hdr["EQUINOX"] = 2000.0
        if "RADECSYS" in hdr:
            del hdr["RADECSYS"]

        hdr["BMAJ"] = final_beam[0] / 3600.0
        hdr["BMIN"] = final_beam[1] / 3600.0
        hdr["BPA"] = final_beam[2]
        hdr["NINPUT"] = len(records)
        hdr["COMBTYPE"] = product_code
        hdr["COMBLABL"] = product_label[:68]
        hdr["PRODTYPE"] = product
        hdr["WGT_MODE"] = weight_mode
        hdr["BMAJ_CE"] = final_beam[0]
        hdr["BMIN_CE"] = final_beam[1]
        hdr["BPA_CE"] = final_beam[2]
        hdr.add_history(f"Final radio/common beam: BMAJ={final_beam[0]:.6g} arcsec, BMIN={final_beam[1]:.6g} arcsec, BPA={final_beam[2]:.6g} deg")
        for i, (rec, b) in enumerate(zip(records, input_beams), start=1):
            hdr.add_history(f"Input {i} {rec.get('epoch','')}: BMAJ={b[0]:.6g} arcsec, BMIN={b[1]:.6g} arcsec, BPA={b[2]:.6g} deg")
            hdr[f"BMAJ{i}"] = b[0]
            hdr[f"BMIN{i}"] = b[1]
            hdr[f"BPA{i}"] = b[2]
            hdr[f"EPOCH{i}"] = str(rec.get("epoch", ""))[:68]
            hdr[f"INP{i}"] = Path(rec["map"]).name[:68]
        hdul.flush()


def stack_arrays(fits_list: Sequence[str | Path],
                 rms_fits_list: Optional[Sequence[str | Path]],
                 window_size: int,
                 do_mean: bool,
                 do_median: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], fits.Header]:
    imgs = []
    hdr = None
    for f in fits_list:
        d, h = read_fits_2d(f)
        if hdr is None:
            hdr = h
        imgs.append(d)
    imgs_arr = np.stack(imgs)

    if rms_fits_list:
        rms_maps = []
        for rf in rms_fits_list:
            rd, _ = read_fits_2d(rf)
            rms_maps.append(rd)
        rms_arr = np.stack(rms_maps)
    else:
        rms_arr = np.stack([local_rms(img, window_size) for img in imgs_arr])

    weights = 1.0 / (rms_arr**2)
    weights[~np.isfinite(weights)] = 0.0

    mean_stack = None
    median_stack = None

    if do_mean:
        num = np.nansum(weights * imgs_arr, axis=0)
        den = np.nansum(weights, axis=0)
        mean_stack = num / den
        mean_stack[den == 0] = np.nan
        mean_stack = mean_stack.astype(np.float32, copy=False)

    if do_median:
        # Same weighted median idea as the current stacker, but vectorized for full maps.
        vals = imgs_arr.copy()
        vals[~np.isfinite(vals)] = np.nan
        order = np.argsort(np.where(np.isfinite(vals), vals, np.inf), axis=0)
        vals_sorted = np.take_along_axis(vals, order, axis=0)
        weights_sorted = np.take_along_axis(weights, order, axis=0)
        good_sorted = np.isfinite(vals_sorted) & (weights_sorted > 0)
        weights_sorted = np.where(good_sorted, weights_sorted, 0.0)
        csum = np.cumsum(weights_sorted, axis=0)
        total = csum[-1]
        cutoff = 0.5 * total
        pick = np.argmax(csum >= cutoff[None, :, :], axis=0)
        median_stack = np.take_along_axis(vals_sorted, pick[None, :, :], axis=0)[0]
        median_stack[total <= 0] = np.nan
        median_stack = median_stack.astype(np.float32, copy=False)

    return mean_stack, median_stack, hdr


def stack_arrays_with_error(fits_list: Sequence[str | Path],
                            rms_fits_list: Optional[Sequence[str | Path]],
                            window_size: int,
                            do_mean: bool,
                            do_median: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], fits.Header]:
    """Same stacking as stack_arrays, but also returns mean/median error maps."""
    imgs = []
    hdr = None
    for f in fits_list:
        d, h = read_fits_2d(f)
        if hdr is None:
            hdr = h
        imgs.append(d)
    imgs_arr = np.stack(imgs)

    if rms_fits_list:
        rms_maps = []
        for rf in rms_fits_list:
            rd, _ = read_fits_2d(rf)
            rms_maps.append(rd)
        rms_arr = np.stack(rms_maps)
    else:
        rms_arr = np.stack([local_rms(img, window_size) for img in imgs_arr])

    weights = 1.0 / (rms_arr**2)
    weights[~np.isfinite(weights)] = 0.0

    mean_stack = None
    mean_err = None
    median_stack = None
    median_err = None

    if do_mean:
        num = np.nansum(weights * imgs_arr, axis=0)
        den = np.nansum(weights, axis=0)
        mean_stack = num / den
        mean_stack[den == 0] = np.nan
        mean_stack = mean_stack.astype(np.float32, copy=False)
        mean_err = np.sqrt(1.0 / den)
        mean_err[den == 0] = np.nan
        mean_err = mean_err.astype(np.float32, copy=False)

    if do_median:
        vals = imgs_arr.copy()
        vals[~np.isfinite(vals)] = np.nan
        order = np.argsort(np.where(np.isfinite(vals), vals, np.inf), axis=0)
        vals_sorted = np.take_along_axis(vals, order, axis=0)
        weights_sorted = np.take_along_axis(weights, order, axis=0)
        rms_sorted = np.take_along_axis(rms_arr, order, axis=0)
        good_sorted = np.isfinite(vals_sorted) & (weights_sorted > 0)
        weights_sorted = np.where(good_sorted, weights_sorted, 0.0)
        csum = np.cumsum(weights_sorted, axis=0)
        total = csum[-1]
        cutoff = 0.5 * total
        pick = np.argmax(csum >= cutoff[None, :, :], axis=0)
        median_stack = np.take_along_axis(vals_sorted, pick[None, :, :], axis=0)[0]
        median_err = np.take_along_axis(rms_sorted, pick[None, :, :], axis=0)[0]
        median_stack[total <= 0] = np.nan
        median_err[total <= 0] = np.nan
        median_stack = median_stack.astype(np.float32, copy=False)
        median_err = median_err.astype(np.float32, copy=False)

    return mean_stack, median_stack, mean_err, median_err, hdr


def robust_rms_mad(data: np.ndarray) -> float:
    good = np.isfinite(data)
    if not np.any(good):
        return float("nan")
    vals = np.asarray(data[good], dtype=np.float64)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return float(1.4826 * mad)


def make_alpha_array_and_error(tt1: np.ndarray,
                               tt0: np.ndarray,
                               sig1: Optional[np.ndarray],
                               sig0: Optional[np.ndarray],
                               threshold: float = SPX_MASK_THRESHOLD) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rms0_for_mask = robust_rms_mad(tt0)
    if not np.isfinite(rms0_for_mask) or rms0_for_mask == 0:
        finite_tt0 = tt0[np.isfinite(tt0)]
        rms0_for_mask = np.nanstd(finite_tt0) if finite_tt0.size else float("nan")
    cutoff = threshold * rms0_for_mask if np.isfinite(rms0_for_mask) else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = tt1 / tt0
    alpha[~np.isfinite(alpha)] = np.nan
    mask = np.isfinite(tt0) & (np.abs(tt0) >= cutoff)
    alpha[~mask] = np.nan

    alpha_err = None
    if sig1 is not None and sig0 is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_err = np.sqrt((sig1 / tt0)**2 + ((tt1 * sig0) / (tt0**2))**2)
        alpha_err[~np.isfinite(alpha_err)] = np.nan
        alpha_err[~mask] = np.nan
        alpha_err = alpha_err.astype(np.float32, copy=False)

    return alpha.astype(np.float32, copy=False), alpha_err


def write_error_fits(path: Path, data: np.ndarray, hdr: fits.Header, bunit: str = "") -> None:
    hdr2 = hdr.copy()
    if bunit:
        hdr2["BUNIT"] = bunit
    fits.writeto(path, data.astype(np.float32, copy=False), hdr2, overwrite=True)
    force_fits_to_J2000_inplace(path)


def copy_final_products(paths: Sequence[Path],
                        outdir: Path,
                        survey: str,
                        product: str,
                        tile: str,
                        source_id: str,
                        records: Sequence[Dict[str, Path]],
                        keep_in_one_folder: bool,
                        keep_separate_area: bool) -> None:
    existing = [Path(p) for p in paths if p is not None and Path(p).exists()]
    if keep_in_one_folder:
        flat = outdir / "all_stacked_products"
        flat.mkdir(parents=True, exist_ok=True)
        for p in existing:
            shutil.copy2(p, flat / p.name)
    if keep_separate_area:
        for rec in records:
            epoch = str(rec.get("epoch", "UNKNOWN"))
            src_name = Path(str(rec.get("src_dir", source_id))).name
            dest = outdir / "stash_like_outputs" / survey / epoch / tile / src_name
            dest.mkdir(parents=True, exist_ok=True)
            for p in existing:
                shutil.copy2(p, dest / p.name)


def write_array_fits(path: Path, data: np.ndarray, hdr: fits.Header) -> None:
    fits.writeto(path, data.astype(np.float32, copy=False), hdr, overwrite=True)
    force_fits_to_J2000_inplace(path)


def brightest_pixel(data: np.ndarray) -> Tuple[int, int]:
    arr = np.asarray(data, dtype=np.float64)
    if not np.isfinite(arr).any():
        return arr.shape[0] // 2, arr.shape[1] // 2
    idx = np.nanargmax(arr)
    y, x = np.unravel_index(idx, arr.shape)
    return int(y), int(x)


def make_zoom_png(fits_path: Path, png_path: Path, cutout_size: int, title: str, center_yx: Optional[Tuple[int, int]] = None, center_sky=None) -> None:
    data, hdr = read_fits_2d(fits_path)
    if center_sky is not None:
        try:
            w0 = WCS(hdr).celestial
            xw, yw = w0.world_to_pixel(center_sky)
            if np.isfinite(xw) and np.isfinite(yw):
                y0, x0 = int(round(float(yw))), int(round(float(xw)))
            elif center_yx is not None:
                y0, x0 = int(center_yx[0]), int(center_yx[1])
            else:
                y0, x0 = brightest_pixel(data)
        except Exception:
            if center_yx is not None:
                y0, x0 = int(center_yx[0]), int(center_yx[1])
            else:
                y0, x0 = brightest_pixel(data)
    elif center_yx is None:
        y0, x0 = brightest_pixel(data)
    else:
        y0, x0 = int(center_yx[0]), int(center_yx[1])
    y0 = max(0, min(y0, data.shape[0] - 1))
    x0 = max(0, min(x0, data.shape[1] - 1))
    half = max(1, int(cutout_size) // 2)
    y1 = max(0, y0 - half)
    y2 = min(data.shape[0], y0 + half)
    x1 = max(0, x0 - half)
    x2 = min(data.shape[1], x0 + half)
    cut = data[y1:y2, x1:x2]

    hdr2 = hdr.copy()
    try:
        hdr2["CRPIX1"] = float(hdr2.get("CRPIX1", 0.0)) - x1
        hdr2["CRPIX2"] = float(hdr2.get("CRPIX2", 0.0)) - y1
        w = WCS(hdr2).celestial
    except Exception:
        w = None

    finite = np.isfinite(cut)
    if finite.any():
        vmin, vmax = np.nanpercentile(cut[finite], [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(cut), np.nanmax(cut)
    else:
        vmin, vmax = -1, 1

    fig = plt.figure(figsize=(4.4, 4.0), dpi=130)
    ax = fig.add_subplot(111, projection=w) if w is not None else fig.add_subplot(111)
    im = ax.imshow(cut, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    if hasattr(ax, "coords"):
        try:
            ax.coords[0].set_axislabel("")
            ax.coords[1].set_axislabel("")
            ax.coords[0].set_ticks(number=3)
            ax.coords[1].set_ticks(number=3)
            ax.coords[0].set_ticklabel(size=7, exclude_overlapping=True)
            ax.coords[1].set_ticklabel(size=7, exclude_overlapping=True)
        except Exception:
            pass
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def make_delta_images(combined_path: Path,
                      input_paths: Sequence[str | Path],
                      outdir: Path,
                      code: str) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    combo, hdr = read_fits_2d(combined_path)
    outs = []
    for inp in input_paths:
        dat, _ = read_fits_2d(inp)
        delta = combo - dat
        out = outdir / f"{combined_path.stem}_minus_{Path(inp).stem}_{code}.fits"
        fits.writeto(out, delta.astype(np.float32), hdr, overwrite=True)
        force_fits_to_J2000_inplace(out)
        outs.append(out)
    return outs


def make_imdev_rms_map(combined_fits: Path, rms_outdir: Path) -> Optional[Path]:
    rms_outdir.mkdir(parents=True, exist_ok=True)
    work = rms_outdir / (combined_fits.stem + ".image")
    out_image = rms_outdir / (combined_fits.stem + ".rms")
    out_fits = rms_outdir / (combined_fits.stem + ".rms.fits")
    try:
        rm_if_exists(work)
        rm_if_exists(out_image)
        importfits(fitsimage=str(combined_fits), imagename=str(work), overwrite=True)
        imdev(
            imagename=str(work),
            outfile=str(out_image),
            overwrite=True,
            stretch=False,
            grid=IMDEV_GRID,
            anchor="ref",
            xlength=IMDEV_XLENGTH,
            ylength=IMDEV_YLENGTH,
            interp=IMDEV_INTERP,
            stattype=IMDEV_STATTYPE,
            statalg=IMDEV_STATALG,
            zscore=IMDEV_ZSCORE,
            maxiter=IMDEV_MAXITER,
        )
        exportfits(imagename=str(out_image), fitsimage=str(out_fits), overwrite=True)
        force_fits_to_J2000_inplace(out_fits)
        rm_if_exists(work)
        rm_if_exists(out_image)
        return out_fits
    except Exception as e:
        print(f"[WARN] imdev failed for {combined_fits}: {e}")
        try:
            rm_if_exists(work)
            rm_if_exists(out_image)
        except Exception:
            pass
        return None


def fmt_float(x, nd=5) -> str:
    try:
        x = float(x)
        if not np.isfinite(x):
            return "—"
        return f"{x:.{nd}g}"
    except Exception:
        return "—"


def relpath(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


def write_html_report(report_path: Path,
                      survey: str,
                      product: str,
                      tile: str,
                      source_id: str,
                      records: Sequence[Dict[str, Path]],
                      input_stats: Sequence[Dict[str, object]],
                      combined_infos: Sequence[Dict[str, object]],
                      qa_dir: Path,
                      delta_paths: Sequence[Path],
                      rms_paths: Sequence[Path]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def stats_table_rows(stats_list, base_path: Path):
        rows = []
        for st in stats_list:
            rows.append(
                "<tr>"
                f"<td class='mono'>{html.escape(Path(str(st['file'])).name)}</td>"
                f"<td>{html.escape(str(st.get('shape','')))}</td>"
                f"<td>{fmt_float(st.get('bmaj_as'), 4)} x {fmt_float(st.get('bmin_as'), 4)} @ {fmt_float(st.get('bpa_deg'), 4)}</td>"
                f"<td>{fmt_float(st.get('rms'), 6)}</td>"
                f"<td>{fmt_float(st.get('max'), 6)}</td>"
                f"<td>{fmt_float(st.get('mean'), 6)}</td>"
                f"<td>{fmt_float(st.get('median'), 6)}</td>"
                f"<td>{fmt_float(100.0 * float(st.get('finite_frac', np.nan)), 4)}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    input_rows = stats_table_rows(input_stats, report_path.parent)

    def common_zoom_sky_center():
        # Use one reference sky coordinate for every QA zoom card so input,
        # combined, RMS, and delta previews show the same area of sky.
        # Prefer the brightest finite pixel in the first input map.
        try:
            if input_stats:
                ref_path = Path(str(input_stats[0].get("file", "")))
                if ref_path.exists():
                    data0, hdr0 = read_fits_2d(ref_path)
                    y0, x0 = brightest_pixel(data0)
                    return WCS(hdr0).celestial.pixel_to_world(float(x0), float(y0))
        except Exception:
            pass
        try:
            if combined_infos:
                ref_path = Path(str(combined_infos[0].get("path", "")))
                if ref_path.exists():
                    data0, hdr0 = read_fits_2d(ref_path)
                    y0, x0 = brightest_pixel(data0)
                    return WCS(hdr0).celestial.pixel_to_world(float(x0), float(y0))
        except Exception:
            pass
        return None

    qa_center_sky = common_zoom_sky_center()

    def hide_noise_columns_for_combined(info: Dict[str, object]) -> bool:
        label = str(info.get("label", "")).lower()
        fname = Path(str(info.get("path", ""))).name.lower()
        return ("tt1" in label) or (".tt1." in fname) or ("alpha" in label) or (".alpha" in fname)

    combined_rows = []
    cards = []
    for info in combined_infos:
        st = info["stats"]
        png = Path(info["png"])
        try:
            make_zoom_png(Path(str(info["path"])), png, QA_CUTOUT_SIZE_PIX, str(info["label"]), center_sky=qa_center_sky)
        except Exception:
            pass
        cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(str(info['label']))}</h3>"
            f"<img src='{html.escape(relpath(png, report_path.parent))}' alt='{html.escape(str(info['label']))}'>"
            f"<p class='mono shortpath'>{html.escape(Path(str(info['path'])).name)}</p>"
            "</div>"
        )
        hide_noise = hide_noise_columns_for_combined(info)
        rms_cell = "—" if hide_noise else fmt_float(st.get('rms'), 6)
        peak_cell = "—" if hide_noise else fmt_float(st.get('max'), 6)
        combined_rows.append(
            "<tr>"
            f"<td>{html.escape(str(info['label']))}</td>"
            f"<td class='mono shortpath'>{html.escape(Path(str(info['path'])).name)}</td>"
            f"<td>{fmt_float(st.get('bmaj_as'), 4)} x {fmt_float(st.get('bmin_as'), 4)} @ {fmt_float(st.get('bpa_deg'), 4)}</td>"
            f"<td>{rms_cell}</td>"
            f"<td>{peak_cell}</td>"
            f"<td>{fmt_float(st.get('mean'), 6)}</td>"
            f"<td>{fmt_float(st.get('median'), 6)}</td>"
            f"<td>{fmt_float(100.0 * float(st.get('finite_frac', np.nan)), 4)}</td>"
            "</tr>"
        )

    input_cards = []
    for rec, st in zip(records, input_stats):
        png = qa_dir / (Path(str(st["file"])).stem + "_peakzoom.png")
        try:
            make_zoom_png(Path(str(st["file"])), png, QA_CUTOUT_SIZE_PIX, str(rec["epoch"]), center_sky=qa_center_sky)
        except Exception:
            pass
        input_cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(str(rec['epoch']))} input</h3>"
            f"<img src='{html.escape(relpath(png, report_path.parent))}' alt='{html.escape(str(rec['epoch']))} input'>"
            f"<p class='mono'>{html.escape(Path(str(st['file'])).name)}</p>"
            "</div>"
        )

    def short_preview_title(p0: Path, title_prefix: str, idx: int) -> str:
        name = p0.name
        rel = str(p0)
        if title_prefix != "Delta":
            if "alpha" in name.lower():
                return "Local RMS alpha"
            if ".tt1." in name.lower():
                return "Local RMS tt1"
            return f"{title_prefix} {idx}"

        if "/tt0/" in rel or "delta_images/tt0" in rel:
            kind = "tt0"
        elif "/tt1/" in rel or "delta_images/tt1" in rel:
            kind = "tt1"
        elif "/alpha/" in rel or "delta_images/alpha" in rel:
            kind = "alpha"
        else:
            kind = "map"

        lname = name.lower()
        if ".cem." in lname or "_cem" in lname or "median" in lname:
            stat = "median"
        else:
            stat = "mean"

        ep_match = re.search(r"(VLASS\d+(?:\.\d+)?)", name)
        epoch = ep_match.group(1) if ep_match else ""
        if not epoch:
            idx_match = re.search(r"_(\d{2})_", name)
            if idx_match:
                rec_idx = int(idx_match.group(1))
                if 0 <= rec_idx < len(records):
                    epoch = str(records[rec_idx].get("epoch", ""))
        return f"{kind} {stat}" + (f" − {epoch}" if epoch else f" {idx}")

    def delta_preview_center(p0: Path) -> Optional[Tuple[int, int]]:
        # For delta maps, show the same sky area as the matching combined map.
        # The delta FITS itself is unchanged; this only controls the QA PNG zoom center.
        stem = Path(p0).name.split("_minus_", 1)[0]
        for info0 in combined_infos:
            try:
                cpath = Path(str(info0.get("path", "")))
                if cpath.stem == stem and cpath.exists():
                    cdat, _ = read_fits_2d(cpath)
                    return brightest_pixel(cdat)
            except Exception:
                continue
        return None

    def file_preview_cards(paths: Sequence[Path], subdir_name: str, title_prefix: str) -> str:
        preview_cards = []
        preview_dir = qa_dir / subdir_name
        for idx, p0 in enumerate(paths, start=1):
            p0 = Path(p0)
            if not p0.exists():
                continue
            title = short_preview_title(p0, title_prefix, idx)
            shown_name = p0.name if title_prefix == "Delta" else relpath(p0, report_path.parent)
            png = preview_dir / (p0.stem + "_peakzoom.png")
            try:
                make_zoom_png(p0, png, QA_CUTOUT_SIZE_PIX, title, center_sky=qa_center_sky)
                preview_cards.append(
                    "<div class='card'>"
                    f"<h3>{html.escape(title)}</h3>"
                    f"<img src='{html.escape(relpath(png, report_path.parent))}' alt='{html.escape(title)}'>"
                    f"<p class='mono shortpath' title='{html.escape(relpath(p0, report_path.parent))}'>{html.escape(shown_name)}</p>"
                    "</div>"
                )
            except Exception:
                preview_cards.append(
                    "<div class='card'>"
                    f"<h3>{html.escape(title)}</h3>"
                    f"<p class='mono shortpath' title='{html.escape(relpath(p0, report_path.parent))}'>{html.escape(shown_name)}</p>"
                    "<p>Preview PNG could not be generated.</p>"
                    "</div>"
                )
        return "\n".join(preview_cards)

    rms_cards = file_preview_cards(rms_paths, "rms_peakzooms", "Local RMS")
    if not rms_cards:
        rms_cards = "<p>Not generated.</p>"

    delta_cards = file_preview_cards(delta_paths, "delta_peakzooms", "Delta")
    if not delta_cards:
        delta_cards = "<p>Disabled / not generated.</p>"

    input_names = "".join(
        f"<li><b>{html.escape(str(r['epoch']))}</b>: <span class='mono'>{html.escape(str(r['map']))}</span></li>" for r in records
    )

    content = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>VLASS full-map combination QA: {html.escape(tile)} {html.escape(source_id)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 22px; background: #f6f8fb; color: #172033; }}
h1, h2 {{ color: #0c3366; }}
.section {{ background: white; border: 1px solid #d5deea; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #d7e0ea; border-radius: 10px; padding: 10px; background: #ffffff; }}
.card img {{ width: 100%; border: 1px solid #e5eaf0; border-radius: 8px; }}
.card h3 {{ margin: 0 0 8px 0; font-size: 1.0em; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
th, td {{ border: 1px solid #d7e0ea; padding: 6px 7px; vertical-align: top; }}
th {{ background: #edf3fa; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.88em; }}
.shortpath {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
.badge {{ display: inline-block; background: #e7f0fb; color: #0c3366; padding: 3px 8px; border-radius: 999px; margin-right: 6px; }}
</style>
</head>
<body>
<h1>VLASS full-map combination QA</h1>
<div class="section">
  <span class="badge">Survey: {html.escape(survey)}</span>
  <span class="badge">Product: {html.escape(product)}</span>
  <span class="badge">Tile: {html.escape(tile)}</span>
  <span class="badge">Source: {html.escape(source_id)}</span>
  <p><b>Combination group:</b> Radio beam / RMS-weighted maps and any enabled optional products.</p>
</div>

<div class="section">
<h2>Input image names</h2>
<ul>{input_names}</ul>
</div>

<div class="section">
<h2>Input image QA table</h2>
<table>
<thead><tr><th>Input image</th><th>Shape</th><th>Beam BMAJ x BMIN @ BPA</th><th>RMS</th><th>Peak/max</th><th>Mean</th><th>Median</th><th>Finite %</th></tr></thead>
<tbody>{input_rows}</tbody>
</table>
</div>

<div class="section">
<h2>Combined image QA table</h2>
<table>
<thead><tr><th>Product</th><th>Combined image</th><th>Final beam BMAJ x BMIN @ BPA</th><th>RMS</th><th>Peak/max</th><th>Mean</th><th>Median</th><th>Finite %</th></tr></thead>
<tbody>{''.join(combined_rows)}</tbody>
</table>
<p class="small"><b>Finite %:</b> percentage of pixels with valid finite values, excluding NaN or blank pixels. A lower value can indicate blank edges, masked regions, or heavily clipped/masked output.</p>
</div>

<div class="section">
<h2>Peak zoom cutouts: input maps</h2>
<p>WCS zooms centered on the brightest pixel.</p>
<div class="grid">{''.join(input_cards)}</div>
</div>

<div class="section">
<h2>Peak zoom cutouts: combined maps</h2>
<p>WCS zooms centered on the brightest pixel.</p>
<div class="grid">{''.join(cards)}</div>
</div>

<div class="section">
<h2>Peak zoom cutouts: Local RMS maps</h2>
<div class="grid">{rms_cards}</div>
</div>

<div class="section">
<h2>Peak zoom cutouts: Delta images</h2>
<p>Combined map minus each matched input map. Delta previews use the same zoom center as the matching combined product.</p>
<div class="grid">{delta_cards}</div>
</div>

</body>
</html>
"""
    report_path.write_text(content)


# =============================================================================
# CASA stacking
# =============================================================================
def stack_one_group_spectral_index_tt1_tt0(records: Sequence[Dict[str, Path]],
                                           survey: str,
                                           product: str,
                                           tile: str,
                                           source_id: str,
                                           outdir: Path,
                                           out_prefix: str,
                                           template_hint: str,
                                           use_local_rms: bool,
                                           rms_box: int,
                                           freq_ghz: float,
                                           keep_work: bool,
                                           keep_in_one_folder: bool,
                                           keep_separate_area: bool) -> None:
    enabled_products = get_output_products()
    common_products = [p for p in enabled_products if str(p["beam_mode"]) == "common"]
    if not common_products:
        print("[WARN] No common/radio-beam output switches are True; no tt1/tt0 alpha product to generate.")
        return

    input_tt0_fits = [Path(r["map"]) for r in records]
    input_tt1_fits = [Path(r["tt1"]) for r in records if r.get("tt1") is not None]
    if len(input_tt1_fits) != len(input_tt0_fits):
        print("[WARN] Missing tt1 maps for one or more inputs; skipping tt1/tt0 alpha generation.")
        return

    tt0_rms_fits = [Path(r["rms"]) for r in records if r.get("rms") is not None]
    tt1_rms_fits = [Path(r["tt1_rms"]) for r in records if r.get("tt1_rms") is not None]
    use_external_tt0_rms = (not use_local_rms) and (len(tt0_rms_fits) == len(input_tt0_fits))
    use_external_tt1_rms = (not use_local_rms) and (len(tt1_rms_fits) == len(input_tt1_fits))
    weight_mode = "archive_tt0_tt1_rms" if (use_external_tt0_rms and use_external_tt1_rms) else f"local_running_rms_box_{rms_box}"

    template_fits = choose_template(input_tt0_fits, template_hint)
    product_outdir = outdir / survey / product / tile / source_id
    product_outdir.mkdir(parents=True, exist_ok=True)
    qa_dir = product_outdir / "qa_assets"
    delta_dir = product_outdir / "delta_images"
    rms_outdir = product_outdir / "stack_local_rms"
    workdir = Path(tempfile.mkdtemp(prefix="vlass_fullmap_stack_spx_", dir=str(product_outdir)))
    oldcwd = Path.cwd()
    casa_images: List[str] = []

    def _smooth_image_set(local_inputs: Sequence[Path], final_beam: Tuple[float, float, float], tag: str) -> List[str]:
        out = []
        for f in local_inputs:
            base = tag + "_" + f.stem
            casa_name = base + ".image"
            rm_if_exists(casa_name)
            importfits(fitsimage=str(f), imagename=casa_name, overwrite=True)
            casa_images.append(casa_name)
            regrid_name = base + "_regrid.image"
            rm_if_exists(regrid_name)
            imregrid(imagename=casa_name, template="template.image", output=regrid_name, overwrite=True)
            casa_images.append(regrid_name)
            regrid_K = base + "_regrid_K.image"
            jybeam_to_K(regrid_name, regrid_K, freq_ghz)
            casa_images.append(regrid_K)
            sm_name = base + "_common.image"
            rm_if_exists(sm_name)
            bmaj, bmin, bpa = final_beam
            imsmooth(imagename=regrid_K, outfile=sm_name, kernel="gauss", targetres=True,
                     major=str(float(bmaj) + 0.01) + "arcsec",
                     minor=str(float(bmin) + 0.01) + "arcsec",
                     pa=str(float(bpa)) + "deg", overwrite=True)
            casa_images.append(sm_name)
            jy_name = base + "_common_Jy.image"
            K_to_jybeam(sm_name, jy_name, freq_ghz)
            casa_images.append(jy_name)
            exportfits(jy_name, jy_name + ".fits", overwrite=True)
            out.append(jy_name + ".fits")
        return out

    def _smooth_rms_set(local_rms_inputs: Sequence[Path], final_beam: Tuple[float, float, float], tag: str) -> Optional[List[str]]:
        if not local_rms_inputs:
            return None
        out = []
        for rf in local_rms_inputs:
            base = tag + "_" + rf.stem
            rms_image = base + ".image"
            rm_if_exists(rms_image)
            importfits(fitsimage=str(rf), imagename=rms_image, overwrite=True)
            casa_images.append(rms_image)
            rms_regrid = base + "_regrid.image"
            rm_if_exists(rms_regrid)
            imregrid(imagename=rms_image, template="template.image", output=rms_regrid, overwrite=True)
            casa_images.append(rms_regrid)
            rms_sm = base + "_common.image"
            rm_if_exists(rms_sm)
            bmaj, bmin, bpa = final_beam
            imsmooth(imagename=rms_regrid, outfile=rms_sm, kernel="gauss", targetres=True,
                     major=str(float(bmaj) + 0.01) + "arcsec",
                     minor=str(float(bmin) + 0.01) + "arcsec",
                     pa=str(float(bpa)) + "deg", overwrite=True)
            casa_images.append(rms_sm)
            exportfits(rms_sm, rms_sm + ".fits", overwrite=True)
            out.append(rms_sm + ".fits")
        return out

    try:
        os.chdir(workdir)
        local_tt0 = []
        local_tt1 = []
        local_tt0_rms = []
        local_tt1_rms = []
        for i, f in enumerate(input_tt0_fits):
            lf = workdir / f"tt0_{i:02d}_{f.name}"
            os.symlink(f, lf)
            local_tt0.append(lf)
        for i, f in enumerate(input_tt1_fits):
            lf = workdir / f"tt1_{i:02d}_{f.name}"
            os.symlink(f, lf)
            local_tt1.append(lf)
        for i, rf in enumerate(tt0_rms_fits):
            lrf = workdir / f"tt0rms_{i:02d}_{rf.name}"
            os.symlink(rf, lrf)
            local_tt0_rms.append(lrf)
        for i, rf in enumerate(tt1_rms_fits):
            lrf = workdir / f"tt1rms_{i:02d}_{rf.name}"
            os.symlink(rf, lrf)
            local_tt1_rms.append(lrf)

        template_local = None
        for lf, original in zip(local_tt0, input_tt0_fits):
            if original == template_fits:
                template_local = lf
                break
        if template_local is None:
            template_local = local_tt0[0]

        images = [SpectralCube.read(str(f))[0] for f in local_tt0]
        beams = Beams(beams=[im.beam for im in images])
        common_beam = beams.common_beam()
        input_beams = [beam_to_tuple(im.beam) for im in images]
        final_beam = beam_to_tuple(common_beam)

        print("Common/radio beam:", common_beam)
        print("Weight mode:", weight_mode)
        print("Spectral-index mode: combined tt1 / combined tt0")

        rm_if_exists("template.image")
        importfits(fitsimage=str(template_local), imagename="template.image", overwrite=True)
        casa_images.append("template.image")

        sm_tt0 = _smooth_image_set(local_tt0, final_beam, "tt0")
        sm_tt1 = _smooth_image_set(local_tt1, final_beam, "tt1")
        sm_tt0_rms = _smooth_rms_set(local_tt0_rms, final_beam, "tt0rms") if use_external_tt0_rms else None
        sm_tt1_rms = _smooth_rms_set(local_tt1_rms, final_beam, "tt1rms") if use_external_tt1_rms else None

        do_mean = any(p["stat"] == "mean" for p in common_products)
        do_median = any(p["stat"] == "median" for p in common_products)
        tt0_mean, tt0_med, tt0_mean_err, tt0_med_err, out_hdr = stack_arrays_with_error(sm_tt0, sm_tt0_rms, max(1, int(rms_box)), do_mean, do_median)
        tt1_mean, tt1_med, tt1_mean_err, tt1_med_err, tt1_hdr = stack_arrays_with_error(sm_tt1, sm_tt1_rms, max(1, int(rms_box)), do_mean, do_median)

        combined_infos = []
        delta_paths: List[Path] = []
        imdev_rms_paths: List[Path] = []
        final_copy_paths: List[Path] = []

        def _make_smoothed_epoch_alpha_inputs(tag: str) -> List[Path]:
            alpha_inputs: List[Path] = []
            for j, (tt1_fp, tt0_fp) in enumerate(zip(sm_tt1, sm_tt0)):
                tt1_d, tt1_h = read_fits_2d(tt1_fp)
                tt0_d, _ = read_fits_2d(tt0_fp)
                alpha_d, _ = make_alpha_array_and_error(tt1_d, tt0_d, None, None, threshold=SPX_MASK_THRESHOLD)
                out_alpha = workdir / f"epoch_alpha_{tag}_{j:02d}.fits"
                fits.writeto(out_alpha, alpha_d.astype(np.float32, copy=False), tt1_h, overwrite=True)
                force_fits_to_J2000_inplace(out_alpha)
                alpha_inputs.append(out_alpha)
            return alpha_inputs

        epoch_alpha_inputs_cache: Dict[str, List[Path]] = {}

        for pdef in common_products:
            stat = str(pdef["stat"])
            code = str(pdef["code"])
            label = str(pdef["label"])
            tt0_arr = tt0_mean if stat == "mean" else tt0_med
            tt1_arr = tt1_mean if stat == "mean" else tt1_med
            tt0_err = tt0_mean_err if stat == "mean" else tt0_med_err
            tt1_err = tt1_mean_err if stat == "mean" else tt1_med_err
            if tt0_arr is None or tt1_arr is None:
                continue

            tt0_name = combined_filename(tile, source_id, code, "intensity")
            tt1_name = f"VLASS.{code}.{tile}.{source_id}.v1.I.tt1.fits"
            alpha_name = combined_filename(tile, source_id, code, "alpha")
            alpha_err_name = f"VLASS.{code}.{tile}.{source_id}.v1.I.alpha.error.fits"
            if out_prefix:
                pre = safe_name(out_prefix) + "_"
                tt0_name = pre + tt0_name
                tt1_name = pre + tt1_name
                alpha_name = pre + alpha_name
                alpha_err_name = pre + alpha_err_name

            tt0_path = product_outdir / tt0_name
            tt1_path = product_outdir / tt1_name
            alpha_path = product_outdir / alpha_name
            alpha_err_path = product_outdir / alpha_err_name

            write_array_fits(tt0_path, tt0_arr, out_hdr)
            update_combined_header(tt0_path, records, input_beams, final_beam, code, label + " tt0", "intensity", weight_mode)
            final_copy_paths.append(tt0_path)

            write_array_fits(tt1_path, tt1_arr, tt1_hdr)
            update_combined_header(tt1_path, records, input_beams, final_beam, code, label + " tt1", "tt1", weight_mode)
            final_copy_paths.append(tt1_path)

            alpha_arr, alpha_err_arr = make_alpha_array_and_error(tt1_arr, tt0_arr, tt1_err, tt0_err, threshold=SPX_MASK_THRESHOLD)
            write_array_fits(alpha_path, alpha_arr, out_hdr)
            update_combined_header(alpha_path, records, input_beams, final_beam, code, label + " alpha=tt1/tt0", "alpha", weight_mode)
            with fits.open(alpha_path, mode="update") as hdul:
                hdul[0].header.add_history("Spectral index generated as combined tt1 / combined tt0.")
                hdul.flush()
            final_copy_paths.append(alpha_path)

            if alpha_err_arr is not None:
                write_error_fits(alpha_err_path, alpha_err_arr, out_hdr, bunit="")
                update_combined_header(alpha_err_path, records, input_beams, final_beam, code, label + " alpha error", "alpha_error", weight_mode)
                with fits.open(alpha_err_path, mode="update") as hdul:
                    hdul[0].header.add_history("Alpha error propagated from combined tt1 and tt0 error maps.")
                    hdul.flush()
                final_copy_paths.append(alpha_err_path)

            st_tt0 = image_stats(tt0_path)
            png_tt0 = qa_dir / (tt0_path.stem + "_peakzoom.png")
            make_zoom_png(tt0_path, png_tt0, QA_CUTOUT_SIZE_PIX, label + " tt0")
            combined_infos.append({"label": label + " tt0", "path": tt0_path, "stats": st_tt0, "png": png_tt0})

            st_tt1 = image_stats(tt1_path)
            png_tt1 = qa_dir / (tt1_path.stem + "_peakzoom.png")
            make_zoom_png(tt1_path, png_tt1, QA_CUTOUT_SIZE_PIX, label + " tt1")
            combined_infos.append({"label": label + " tt1", "path": tt1_path, "stats": st_tt1, "png": png_tt1})

            st = image_stats(alpha_path)
            png_path = qa_dir / (alpha_path.stem + "_peakzoom.png")
            make_zoom_png(alpha_path, png_path, QA_CUTOUT_SIZE_PIX, label + " alpha=tt1/tt0")
            combined_infos.append({"label": label + " alpha=tt1/tt0", "path": alpha_path, "stats": st, "png": png_path})

            if MAKE_DELTA_IMAGES:
                delta_paths.extend(make_delta_images(tt0_path, sm_tt0, delta_dir / "tt0", code))
                delta_paths.extend(make_delta_images(tt1_path, sm_tt1, delta_dir / "tt1", code))
                if stat not in epoch_alpha_inputs_cache:
                    epoch_alpha_inputs_cache[stat] = _make_smoothed_epoch_alpha_inputs(stat)
                delta_paths.extend(make_delta_images(alpha_path, epoch_alpha_inputs_cache[stat], delta_dir / "alpha", code))

            if MAKE_IMDEV_RMS_MAPS and stat == "mean":
                rms_path = make_imdev_rms_map(tt0_path, rms_outdir)
                if rms_path is not None:
                    imdev_rms_paths.append(rms_path)
                    final_copy_paths.append(rms_path)

        input_stats = [image_stats(p) for p in input_tt0_fits]
        for rec, st in zip(records, input_stats):
            make_zoom_png(Path(str(st["file"])), qa_dir / (Path(str(st["file"])).stem + "_peakzoom.png"), QA_CUTOUT_SIZE_PIX, str(rec["epoch"]))

        report_path = product_outdir / f"VLASS_QA_{survey}_{product}_{tile}_{source_id}.html"
        write_html_report(report_path, survey, product, tile, source_id, records, input_stats, combined_infos, qa_dir, delta_paths, imdev_rms_paths)
        copy_final_products(final_copy_paths, outdir, survey, product, tile, source_id, records, keep_in_one_folder, keep_separate_area)
        print(f"QA HTML: {report_path}")

    finally:
        os.chdir(oldcwd)
        if not keep_work:
            for f in casa_images:
                try:
                    rm_if_exists(workdir / f)
                except Exception:
                    pass
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[DEBUG] kept workdir: {workdir}")


def stack_one_group(records: Sequence[Dict[str, Path]],
                    survey: str,
                    product: str,
                    tile: str,
                    source_id: str,
                    outdir: Path,
                    out_prefix: str,
                    template_hint: str,
                    use_local_rms: bool,
                    rms_box: int,
                    freq_ghz: float,
                    keep_work: bool,
                    spx_tt1_tt0: bool = SPX_TT1_TT0,
                    spx_rms_weighted: bool = SPX_RMS_WEIGHTED,
                    keep_in_one_folder: bool = KEEP_IN_ONE_FOLDER,
                    keep_separate_area: bool = KEEP_SEPARATE_AREA) -> None:
    enabled_products = get_output_products()
    if not enabled_products:
        print("[WARN] No output map switches are True; nothing to generate.")
        return

    if product == "alpha" and survey == "SE" and spx_tt1_tt0 and (not spx_rms_weighted):
        stack_one_group_spectral_index_tt1_tt0(
            records=records,
            survey=survey,
            product=product,
            tile=tile,
            source_id=source_id,
            outdir=outdir,
            out_prefix=out_prefix,
            template_hint=template_hint,
            use_local_rms=use_local_rms,
            rms_box=rms_box,
            freq_ghz=freq_ghz,
            keep_work=keep_work,
            keep_in_one_folder=keep_in_one_folder,
            keep_separate_area=keep_separate_area,
        )
        return

    input_fits = [Path(r["map"]) for r in records]
    rms_fits = [Path(r["rms"]) for r in records if r.get("rms") is not None]
    use_external_rms = (not use_local_rms) and (len(rms_fits) == len(input_fits))
    weight_mode = "archive_tt0_rms" if use_external_rms else f"local_running_rms_box_{rms_box}"

    template_fits = choose_template(input_fits, template_hint)
    product_outdir = outdir / survey / product / tile / source_id
    product_outdir.mkdir(parents=True, exist_ok=True)
    qa_dir = product_outdir / "qa_assets"
    delta_dir = product_outdir / "delta_images"
    rms_outdir = product_outdir / "stack_local_rms"

    workdir = Path(tempfile.mkdtemp(prefix="vlass_fullmap_stack_", dir=str(product_outdir)))
    oldcwd = Path.cwd()
    casa_images: List[str] = []

    try:
        os.chdir(workdir)
        local_inputs = []
        local_rms_files = []
        for i, f in enumerate(input_fits):
            lf = workdir / f"input_{i:02d}_{f.name}"
            os.symlink(f, lf)
            local_inputs.append(lf)
        for i, rf in enumerate(rms_fits):
            lrf = workdir / f"rms_{i:02d}_{rf.name}"
            os.symlink(rf, lrf)
            local_rms_files.append(lrf)

        template_local = None
        for lf, original in zip(local_inputs, input_fits):
            if original == template_fits:
                template_local = lf
                break
        if template_local is None:
            template_local = local_inputs[0]

        # Beam determination from original maps, same CASA/radio_beam style.
        images = [SpectralCube.read(str(f))[0] for f in local_inputs]
        beams = Beams(beams=[im.beam for im in images])
        common_beam = beams.common_beam()
        largest_index = int(np.argmax([im.beam.major.to_value("arcsec") for im in images]))
        largest_beam_input = images[largest_index].beam
        largest_bmaj = float(largest_beam_input.major.to_value("arcsec"))
        largest_bmin = float(largest_beam_input.minor.to_value("arcsec"))
        largest_bpa = 0.0
        survey_beam = Beam(
            major=SURVEY_DEFINED_BMAJ_ARCSEC * u.arcsec,
            minor=SURVEY_DEFINED_BMIN_ARCSEC * u.arcsec,
            pa=SURVEY_DEFINED_BPA_DEG * u.deg,
        )

        input_beams = [beam_to_tuple(im.beam) for im in images]
        final_beams = {
            "common": beam_to_tuple(common_beam),
            "largest": (largest_bmaj, largest_bmin, largest_bpa),
            "survey": beam_to_tuple(survey_beam),
        }

        print("Common/radio beam:", common_beam)
        print("Largest beam:", largest_bmaj, largest_bmin, "BPA=0")
        print("Survey-defined beam:", survey_beam)
        print("Weight mode:", weight_mode)

        rm_if_exists("template.image")
        importfits(fitsimage=str(template_local), imagename="template.image", overwrite=True)
        template_image_name = "template.image"
        casa_images.append(template_image_name)

        smoothed_fits: Dict[str, List[str]] = {"common": [], "largest": [], "survey": []}

        for f in local_inputs:
            base = f.stem
            casa_name = base + ".image"
            rm_if_exists(casa_name)
            importfits(fitsimage=str(f), imagename=casa_name, overwrite=True)
            casa_images.append(casa_name)

            regrid_name = base + "_regrid.image"
            rm_if_exists(regrid_name)
            imregrid(imagename=casa_name, template=template_image_name,
                     output=regrid_name, overwrite=True)
            casa_images.append(regrid_name)

            if product == "intensity":
                regrid_for_smooth = base + "_regrid_K.image"
                jybeam_to_K(regrid_name, regrid_for_smooth, freq_ghz)
                casa_images.append(regrid_for_smooth)
            else:
                regrid_for_smooth = regrid_name

            beam_specs = {
                "common": final_beams["common"],
                "largest": final_beams["largest"],
                "survey": final_beams["survey"],
            }
            for mode, (bmaj, bmin, bpa) in beam_specs.items():
                if not any(p["beam_mode"] == mode for p in enabled_products):
                    continue
                sm_name = base + f"_{mode}.image"
                rm_if_exists(sm_name)
                imsmooth(
                    imagename=regrid_for_smooth,
                    outfile=sm_name,
                    kernel="gauss",
                    targetres=True,
                    major=str(float(bmaj) + 0.01) + "arcsec",
                    minor=str(float(bmin) + 0.01) + "arcsec",
                    pa=str(float(bpa)) + "deg",
                    overwrite=True,
                )
                casa_images.append(sm_name)

                if product == "intensity":
                    jy_name = base + f"_{mode}_Jy.image"
                    K_to_jybeam(sm_name, jy_name, freq_ghz)
                    casa_images.append(jy_name)
                    exportfits(jy_name, jy_name + ".fits", overwrite=True)
                    smoothed_fits[mode].append(jy_name + ".fits")
                else:
                    exportfits(sm_name, sm_name + ".fits", overwrite=True)
                    smoothed_fits[mode].append(sm_name + ".fits")

        smoothed_rms_fits: Dict[str, Optional[List[str]]] = {"common": None, "largest": None, "survey": None}
        if use_external_rms:
            smoothed_rms_fits = {"common": [], "largest": [], "survey": []}
            for rf in local_rms_files:
                rbase = rf.stem
                rms_image = rbase + ".image"
                rm_if_exists(rms_image)
                importfits(fitsimage=str(rf), imagename=rms_image, overwrite=True)
                casa_images.append(rms_image)

                rms_regrid = rbase + "_regrid.image"
                rm_if_exists(rms_regrid)
                imregrid(imagename=rms_image, template=template_image_name,
                         output=rms_regrid, overwrite=True)
                casa_images.append(rms_regrid)

                for mode, (bmaj, bmin, bpa) in final_beams.items():
                    if not any(p["beam_mode"] == mode for p in enabled_products):
                        continue
                    rms_sm = rbase + f"_{mode}.image"
                    rm_if_exists(rms_sm)
                    imsmooth(
                        imagename=rms_regrid,
                        outfile=rms_sm,
                        kernel="gauss",
                        targetres=True,
                        major=str(float(bmaj) + 0.01) + "arcsec",
                        minor=str(float(bmin) + 0.01) + "arcsec",
                        pa=str(float(bpa)) + "deg",
                        overwrite=True,
                    )
                    casa_images.append(rms_sm)
                    exportfits(rms_sm, rms_sm + ".fits", overwrite=True)
                    assert smoothed_rms_fits[mode] is not None
                    smoothed_rms_fits[mode].append(rms_sm + ".fits")

        combined_infos = []
        delta_paths: List[Path] = []
        imdev_rms_paths: List[Path] = []

        for mode in sorted(set(str(p["beam_mode"]) for p in enabled_products)):
            mode_products = [p for p in enabled_products if p["beam_mode"] == mode]
            do_mean = any(p["stat"] == "mean" for p in mode_products)
            do_median = any(p["stat"] == "median" for p in mode_products)
            mean_arr, med_arr, out_hdr = stack_arrays(
                smoothed_fits[mode],
                smoothed_rms_fits[mode] if smoothed_rms_fits.get(mode) else None,
                max(1, int(rms_box)),
                do_mean=do_mean,
                do_median=do_median,
            )

            for pdef in mode_products:
                arr = mean_arr if pdef["stat"] == "mean" else med_arr
                if arr is None:
                    continue
                fname = combined_filename(tile, source_id, str(pdef["code"]), product)
                if out_prefix:
                    fname = safe_name(out_prefix) + "_" + fname
                out_path = product_outdir / fname
                write_array_fits(out_path, arr, out_hdr)
                update_combined_header(
                    out_path,
                    records,
                    input_beams,
                    final_beams[mode],
                    str(pdef["code"]),
                    str(pdef["label"]),
                    product,
                    weight_mode,
                )
                st = image_stats(out_path)
                png_path = qa_dir / (out_path.stem + "_peakzoom.png")
                make_zoom_png(out_path, png_path, QA_CUTOUT_SIZE_PIX, str(pdef["label"]))
                combined_infos.append({"label": pdef["label"], "path": out_path, "stats": st, "png": png_path})

                if MAKE_DELTA_IMAGES:
                    delta_paths.extend(make_delta_images(out_path, smoothed_fits[mode], delta_dir, str(pdef["code"])))

                if MAKE_IMDEV_RMS_MAPS:
                    rms_path = make_imdev_rms_map(out_path, rms_outdir)
                    if rms_path is not None:
                        imdev_rms_paths.append(rms_path)

        input_stats = [image_stats(p) for p in input_fits]
        for rec, st in zip(records, input_stats):
            make_zoom_png(Path(str(st["file"])), qa_dir / (Path(str(st["file"])).stem + "_peakzoom.png"), QA_CUTOUT_SIZE_PIX, str(rec["epoch"]))

        report_path = product_outdir / f"VLASS_QA_{survey}_{product}_{tile}_{source_id}.html"
        write_html_report(
            report_path,
            survey,
            product,
            tile,
            source_id,
            records,
            input_stats,
            combined_infos,
            qa_dir,
            delta_paths,
            imdev_rms_paths,
        )
        final_copy_paths = [Path(str(info["path"])) for info in combined_infos]
        final_copy_paths.extend(imdev_rms_paths)
        copy_final_products(final_copy_paths, outdir, survey, product, tile, source_id, records, keep_in_one_folder, keep_separate_area)
        print(f"QA HTML: {report_path}")

    finally:
        os.chdir(oldcwd)
        if not keep_work:
            for f in casa_images:
                try:
                    rm_if_exists(workdir / f)
                except Exception:
                    pass
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[DEBUG] kept workdir: {workdir}")


def print_group_summary(label: str, groups: Dict[Tuple[str, str], List[Dict[str, Path]]], min_maps: int) -> None:
    n_good = sum(1 for recs in groups.values() if len(recs) >= min_maps)
    print(f"[{label}] discovered groups: {len(groups)}; stackable groups with >= {min_maps} maps: {n_good}")


def main() -> int:
    global MAKE_RADIO_BEAM_MEDIAN_MAP
    args = parse_args()
    if bool(args.make_radio_beam_median):
        MAKE_RADIO_BEAM_MEDIAN_MAP = True
    if bool(args.keep_in_one_folder):
        args.keep_separate_area = False
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    surveys = ["SE", "QL"] if args.survey == "both" else [args.survey]
    products = ["intensity", "alpha"] if args.product == "both" else [args.product]

    jobs: List[Tuple[str, str, Tuple[str, str], List[Dict[str, Path]]]] = []
    source_id_filter = normalize_source_id_filter(getattr(args, "source_id", ""))
    if source_id_filter:
        print(f"[FILTER] source-id: {source_id_filter}")

    local_input_dir = Path(args.input_dir).expanduser().resolve() if str(args.input_dir).strip() else None
    if local_input_dir is not None:
        if not local_input_dir.is_dir():
            print(f"[ERROR] --input-dir is not a directory: {local_input_dir}")
            return 1
        print(f"[LOCAL TEST MODE] input-dir: {local_input_dir}")

    for survey in surveys:
        if local_input_dir is not None:
            wanted_epochs = normalize_epochs(args.epochs, survey)
            tiles = sorted(set(args.tiles or [])) if args.tiles else []
            epochs = wanted_epochs or []
            if survey != "SE":
                print("[WARN] --input-dir test mode currently supports SE-style filenames; skipping non-SE survey.")
                continue
            print(f"[{survey}] local input mode")
            if epochs:
                print(f"[{survey}] epoch filter: {', '.join(epochs)}")
            if tiles:
                print(f"[{survey}] tile filter: {', '.join(tiles)}")
        else:
            base = STASH_SE_BASE if survey == "SE" else STASH_QL_BASE
            wanted_epochs = normalize_epochs(args.epochs, survey)
            epochs = available_epochs(base, wanted_epochs)
            if not epochs:
                print(f"[WARN] No epochs found for {survey} under {base}")
                continue
            tiles = selected_tiles_for_epochs(base, epochs, args.tiles, args.num_tiers)
            if not tiles:
                print(f"[WARN] No tiles found for {survey} epochs {epochs}")
                continue

            print(f"[{survey}] epochs: {', '.join(epochs)}")
            print(f"[{survey}] tiles selected: {len(tiles)}")
            if len(tiles) <= 20:
                print(f"[{survey}] tile list: {', '.join(tiles)}")

        for product in products:
            if survey == "QL" and product == "alpha":
                print("[QL alpha] skipped: standard QL products do not include alpha maps.")
                continue
            if local_input_dir is not None and survey == "SE" and product == "alpha" and args.spx_tt1_tt0 and (not args.spx_rms_weighted):
                groups = discover_one_folder_spectral_index_tt1_tt0(local_input_dir, epochs, tiles)
                print_group_summary(f"{survey} {product} tt1/tt0 local-folder", groups, args.min_maps)
            elif local_input_dir is not None:
                groups = discover_one_folder_product(local_input_dir, survey, product, epochs, tiles)
                print_group_summary(f"{survey} {product} local-folder", groups, args.min_maps)
            elif survey == "SE" and product == "alpha" and args.spx_tt1_tt0 and (not args.spx_rms_weighted):
                groups = discover_spectral_index_tt1_tt0(epochs, tiles)
                print_group_summary(f"{survey} {product} tt1/tt0", groups, args.min_maps)
            else:
                groups = discover_one_survey_product(survey, product, epochs, tiles)
                print_group_summary(f"{survey} {product}", groups, args.min_maps)
            for key, recs in sorted(groups.items()):
                tile_id, source_id = key
                if source_id_filter and source_id != source_id_filter:
                    continue
                recs = sorted(recs, key=lambda r: str(r["epoch"]))
                if len(recs) >= args.min_maps:
                    jobs.append((survey, product, key, recs))
                    if args.max_fields and len(jobs) >= args.max_fields:
                        break
            if args.max_fields and len(jobs) >= args.max_fields:
                break
        if args.max_fields and len(jobs) >= args.max_fields:
            break

    if args.dry_run:
        for survey, product, (tile, source_id), recs in jobs[:50]:
            eps = ",".join(str(r["epoch"]) for r in recs)
            print(f"DRYRUN {survey} {product} {tile} {source_id}: {eps}")
        print(f"Dry run complete. Stackable jobs: {len(jobs)}")
        return 0

    print(f"Total stack jobs: {len(jobs)}")
    failures = 0
    for idx, (survey, product, (tile, source_id), recs) in enumerate(jobs, start=1):
        epochs_s = ",".join(str(r["epoch"]) for r in recs)
        print(f"\n[{idx}/{len(jobs)}] stacking {survey} {product} {tile} {source_id}")
        print(f"    epochs: {epochs_s}")
        for r in recs:
            print(f"    {r['epoch']}: {r['map']}")
        try:
            stack_one_group(
                records=recs,
                survey=survey,
                product=product,
                tile=tile,
                source_id=source_id,
                outdir=outdir,
                out_prefix=args.out_prefix,
                template_hint=args.template_hint,
                use_local_rms=args.use_local_rms,
                rms_box=max(1, int(args.rms_box)),
                freq_ghz=float(args.freq_ghz),
                keep_work=bool(args.keep_work),
                spx_tt1_tt0=bool(args.spx_tt1_tt0),
                spx_rms_weighted=bool(args.spx_rms_weighted),
                keep_in_one_folder=bool(args.keep_in_one_folder),
                keep_separate_area=bool(args.keep_separate_area),
            )
        except Exception as e:
            failures += 1
            print(f"[ERROR] Failed {survey} {product} {tile} {source_id}: {e}")

    print(f"\nDONE. jobs={len(jobs)} failures={failures} outdir={outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
