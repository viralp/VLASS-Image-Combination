# VLASS Full-Map Stack Standalone Script

This repository contains a standalone CASA/Python script for stacking local VLASS image products. It can run in two modes:

1. **/stash discovery mode**: automatically discovers VLASS SE or QL products from the NRAO `/stash` cache.
2. **local one-folder test mode**: combines a small set of FITS images placed in one directory, useful for testing one source before running the full `/stash` workflow.

Main script:

```bash
vlass_fullmap_stack_standalone_v5.py
```

The script performs CASA-style image preparation and stacking:

```text
input epoch maps
→ choose template image
→ regrid epoch maps to common pixel grid
→ compute radio-beam/common beam using radio_beam
→ smooth all epoch maps to the common beam
→ build RMS weights from archive RMS maps or local running RMS maps
→ create weighted mean and optional weighted median stacks
→ write FITS headers with input and final beam information
→ generate QA/weblog HTML with image previews
```

---

## Requirements

Run the script under CASA because it uses CASA tasks such as `importfits`, `exportfits`, `imregrid`, `imsmooth`, `immath`, `imhead`, and `imdev`.

Python packages required inside the CASA Python environment:

```text
numpy
astropy
scipy
spectral_cube
radio_beam
matplotlib
```

Example CASA command:

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py [options]
```

On NRAO Lustre, use the full CASA path if needed:

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py [options]
```

---

## Important default behavior

By default, the script makes only the **radio/common-beam RMS-weighted mean** product:

```python
MAKE_RADIO_BEAM_MEAN_MAP = True
MAKE_RADIO_BEAM_MEDIAN_MAP = False
MAKE_LARGEST_BEAM_MEAN_MAP = False
MAKE_LARGEST_BEAM_MEDIAN_MAP = False
MAKE_SURVEY_DEFINED_BEAM_MEAN_MAP = False
MAKE_SURVEY_DEFINED_BEAM_MEDIAN_MAP = False
```

Delta images and imdev RMS maps are enabled in this version:

```python
MAKE_DELTA_IMAGES = True
MAKE_IMDEV_RMS_MAPS = True
```

For SE spectral-index products, the default method is:

```text
combined alpha = combined tt1 / combined tt0
```

This is controlled by:

```python
SPX_TT1_TT0 = True
SPX_RMS_WEIGHTED = False
```

Use `--spx-rms-weighted` only if you want the older/direct method that combines existing alpha maps directly.

---

## Product naming convention

| Code | Meaning |
|---|---|
| `ce` | common/radio-beam weighted mean |
| `cem` | common/radio-beam weighted median |
| `cel` | largest-beam weighted mean |
| `celm` | largest-beam weighted median |
| `ces` | survey-defined-beam weighted mean |
| `cesm` | survey-defined-beam weighted median |

Example output names:

```text
VLASS.ce.T08t02.J004201-093000.06.2048.v1.I.image.pbcor.fits
VLASS.ce.T08t02.J004201-093000.06.2048.v1.I.tt1.fits
VLASS.ce.T08t02.J004201-093000.06.2048.v1.I.alpha.fits
VLASS.ce.T08t02.J004201-093000.06.2048.v1.I.alpha.error.fits
```

If the weighted median is enabled:

```text
VLASS.cem.T08t02.J004201-093000.06.2048.v1.I.image.pbcor.fits
VLASS.cem.T08t02.J004201-093000.06.2048.v1.I.tt1.fits
VLASS.cem.T08t02.J004201-093000.06.2048.v1.I.alpha.fits
VLASS.cem.T08t02.J004201-093000.06.2048.v1.I.alpha.error.fits
```

---

## FITS header information

The final stacked FITS headers include the final common/radio beam:

```text
BMAJ
BMIN
BPA
BMAJ_CE
BMIN_CE
BPA_CE
```

They also include input epoch beam information:

```text
BMAJ1, BMIN1, BPA1
BMAJ2, BMIN2, BPA2
EPOCH1, EPOCH2, ...
INP1, INP2, ...
```

Readable beam information is also written to FITS `HISTORY`, for example:

```text
Final radio/common beam: BMAJ=... arcsec, BMIN=... arcsec, BPA=... deg
Input 1 VLASS2.1: BMAJ=... arcsec, BMIN=... arcsec, BPA=... deg
Input 2 VLASS3.1: BMAJ=... arcsec, BMIN=... arcsec, BPA=... deg
```

---

## QA / weblog output

Each run creates an HTML QA report under:

```text
<outdir>/<survey>/<product>/<tile>/<source_id>/VLASS_QA_<survey>_<product>_<tile>_<source_id>.html
```

The report includes:

```text
Input image names
Input image QA table
Combined image QA table
Peak zoom cutouts: input maps
Peak zoom cutouts: combined maps
Peak zoom cutouts: Local RMS maps
Peak zoom cutouts: Delta images
```

In the combined QA table, **RMS** and **Peak/max** are reported only for `tt0` / intensity products. These columns are blanked with `—` for `tt1` and `alpha` rows.

`Finite %` means the percentage of pixels with valid finite values, excluding NaN or blank pixels. A lower value can indicate blank edges, masked regions, or heavily clipped/masked output.

---

## RMS weighting modes

### Archive RMS mode

By default, if archive RMS maps are available, the script uses them for weighting:

```text
*I.iter3.image.pbcor.tt0.rms.subim.fits
```

For SE `tt1/tt0` alpha generation, it also looks for:

```text
*I.iter3.image.pbcor.tt1.rms.subim.fits
```

The RMS maps are regridded and smoothed to the same final beam before weights are calculated.

### Local RMS mode

If no RMS maps are available, or for local testing, use:

```bash
--use-local-rms
```

Then the script estimates local RMS maps from the regridded + common-beam-smoothed science images using a running window.

Control the running RMS box size with:

```bash
--rms-box 50
```

The weight at each pixel is:

```text
weight = 1 / rms^2
```

---

## Spectral-index modes

### Default SE spectral-index mode: combined tt1 / combined tt0

Default:

```bash
--product alpha
```

This creates spectral-index products as:

```text
VLASS2/VLASS3 tt0 maps → common-beam weighted combined tt0
VLASS2/VLASS3 tt1 maps → common-beam weighted combined tt1
combined alpha = combined tt1 / combined tt0
```

Alpha error is propagated from the combined tt1 and tt0 error maps:

```text
sigma_alpha = sqrt[ (sigma_tt1 / tt0)^2 + (tt1 * sigma_tt0 / tt0^2)^2 ]
```

### Direct alpha-map weighted combination

Use this only if you want to combine existing epoch alpha maps directly:

```bash
--product alpha --spx-rms-weighted
```

This uses:

```text
*I.iter3.alpha.subim.fits
```

instead of deriving alpha from `tt1/tt0`.

---

## Delta images

If `MAKE_DELTA_IMAGES = True`, the script makes difference images:

```text
delta = combined map - matched smoothed epoch map
```

For SE `tt1/tt0` alpha mode, it generates:

```text
delta_images/tt0/    combined tt0 - each smoothed epoch tt0
delta_images/tt1/    combined tt1 - each smoothed epoch tt1
delta_images/alpha/  combined alpha - each epoch alpha made from smoothed epoch tt1/tt0
```

The QA report displays these delta images as peak-zoom image cards with short readable labels.

---

## Output storage options

### Default separate/source-directory style

By default:

```python
KEEP_IN_ONE_FOLDER = False
KEEP_SEPARATE_AREA = True
```

Main products and QA are written under:

```text
<outdir>/<survey>/<product>/<tile>/<source_id>/
```

Final products are also copied into a stash-like tree:

```text
<outdir>/stash_like_outputs/<survey>/<epoch>/<tile>/<source_directory>/
```

### Copy final FITS products to one folder

Use:

```bash
--keep-in-one-folder
```

This copies final FITS products into:

```text
<outdir>/all_stacked_products/
```

The normal QA/output tree is still created because the weblog and preview PNGs need an organized location.

---

# Command examples

## 1. Local one-folder test: SE alpha from tt1/tt0, mean only

Directory contains:

```text
VLASS2.1.se.T08t02.J004201-093000.06.2048.v1.I.iter3.image.pbcor.tt0.subim.fits
VLASS2.1.se.T08t02.J004201-093000.06.2048.v1.I.iter3.image.pbcor.tt1.subim.fits
VLASS3.1.se.T08t02.J004201-093000.06.2048.v1.I.iter3.image.pbcor.tt0.subim.fits
VLASS3.1.se.T08t02.J004201-093000.06.2048.v1.I.iter3.image.pbcor.tt1.subim.fits
```

Run:

```bash
cd ~/VLASS_script/non_standard/image_combined/test

casa --nologger -c /path/to/vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --keep-in-one-folder \
  --outdir ./stack_test
```

Main output:

```text
./stack_test/SE/alpha/T08t02/J004201-093000.06.2048/
```

Flat final FITS copy:

```text
./stack_test/all_stacked_products/
```

QA report:

```text
./stack_test/SE/alpha/T08t02/J004201-093000.06.2048/VLASS_QA_SE_alpha_T08t02_J004201-093000.06.2048.html
```

---

## 2. Local one-folder test: SE alpha from tt1/tt0, mean + median

Use `--make-radio-beam-median` to also produce the common/radio-beam weighted median product:

```bash
cd ~/VLASS_script/non_standard/image_combined/test

casa --nologger -c /path/to/vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --keep-in-one-folder \
  --outdir ./stack_test
```

This produces `ce` and `cem` products:

```text
ce   = common/radio-beam weighted mean
cem  = common/radio-beam weighted median
```

---

## 3. Local one-folder test: intensity/tt0 stacking only

Use this if you only want to combine `tt0` intensity maps:

```bash
cd ~/VLASS_script/non_standard/image_combined/test

casa --nologger -c /path/to/vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product intensity \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --keep-in-one-folder \
  --outdir ./stack_intensity_test
```

---

## 4. Local one-folder test: direct alpha-map combination

Use this only if you want to combine existing alpha maps directly:

```bash
cd ~/VLASS_script/non_standard/image_combined/test

casa --nologger -c /path/to/vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --spx-rms-weighted \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --keep-in-one-folder \
  --outdir ./stack_alpha_direct_test
```

This uses files like:

```text
*I.iter3.alpha.subim.fits
```

instead of generating alpha from combined `tt1/tt0`.

---

## 5. /stash SE intensity stacking for one source

This scans the NRAO `/stash` SE directory structure:

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product intensity \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --outdir ./stack_stash_intensity
```

By default, this uses archive RMS maps if available.

---

## 6. /stash SE intensity stacking using local RMS instead of archive RMS

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product intensity \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --outdir ./stack_stash_intensity_localrms
```

---

## 7. /stash SE alpha from combined tt1/tt0

This is the default alpha method:

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --outdir ./stack_stash_alpha
```

This discovers:

```text
*I.iter3.image.pbcor.tt0.subim.fits
*I.iter3.image.pbcor.tt1.subim.fits
```

Then produces:

```text
combined tt0
combined tt1
alpha = combined tt1 / combined tt0
alpha error
imdev RMS map from combined mean tt0
QA HTML report
Delta images
```

---

## 8. /stash SE alpha from combined tt1/tt0 with local RMS

Use this when archive RMS maps are missing or when you want RMS estimated from the smoothed maps:

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --outdir ./stack_stash_alpha_localrms
```

---

## 9. /stash SE direct alpha-map weighted combination

Use this if you want to keep the older method and combine existing alpha maps directly:

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product alpha \
  --spx-rms-weighted \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --outdir ./stack_stash_alpha_direct
```

---

## 10. Process both intensity and alpha products

```bash
/lustre/aoc/projects/vlass/vparekh/CASA/casa-6.7.1-12-py3.10.el8/bin/casa --nologger \
  -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product both \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --outdir ./stack_both_test
```

---

## 11. Dry run before processing

Use `--dry-run` to check what will be stacked without running CASA image operations:

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --dry-run
```

---

## 12. Limit number of fields for testing

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product intensity \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --max-fields 1 \
  --outdir ./stack_one_field_test
```

---

## 13. Scan only first tier of /stash tiles

If `--tiles` is not supplied, `--num-tiers` controls how many tier prefixes to scan:

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --survey SE \
  --product intensity \
  --epochs 2.1,3.1 \
  --num-tiers 1 \
  --max-fields 5 \
  --outdir ./stack_first_tier_test
```

---

## 14. Keep temporary CASA working directories

Useful for debugging intermediate CASA images:

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --keep-work \
  --outdir ./stack_debug
```

---

## 15. Add a custom output filename prefix

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --out-prefix TEST01 \
  --outdir ./stack_with_prefix
```

---

## 16. QL intensity stacking from /stash

QL supports intensity stacking only. Alpha is skipped for QL because standard QL directories do not contain alpha maps.

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --survey QL \
  --product intensity \
  --epochs 1.2,2.2,3.2 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --outdir ./stack_ql_intensity
```

---

## 17. QL intensity with local RMS

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --survey QL \
  --product intensity \
  --epochs 1.2,2.2,3.2 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --outdir ./stack_ql_intensity_localrms
```

---

# Command-line option summary

| Option | Purpose |
|---|---|
| `--survey SE` | Use SE products |
| `--survey QL` | Use QL products |
| `--survey both` | Process both SE and QL |
| `--product intensity` | Stack tt0 intensity maps |
| `--product alpha` | Make alpha products |
| `--product both` | Process intensity and alpha |
| `--epochs 2.1,3.1` | Restrict epochs |
| `--tiles T08t02` | Restrict tile(s) |
| `--source-id J...` | Restrict to one source ID |
| `--input-dir .` | Local one-folder test mode |
| `--use-local-rms` | Generate local running RMS maps instead of using archive RMS maps |
| `--rms-box 50` | Running RMS box size in pixels |
| `--make-radio-beam-median` | Also make common/radio-beam weighted median product |
| `--spx-rms-weighted` | Directly combine existing alpha maps instead of `tt1/tt0` |
| `--keep-in-one-folder` | Copy final FITS products into `<outdir>/all_stacked_products/` |
| `--keep-separate-area` | Copy final products into stash-like separate folders |
| `--no-keep-separate-area` | Disable stash-like copy |
| `--outdir` | Output directory |
| `--out-prefix` | Prefix added to output filenames |
| `--template-hint VLASS2` | Preferred template epoch string |
| `--freq-ghz 3.0` | Frequency for Jy/beam to K conversion |
| `--dry-run` | Print discovered jobs only |
| `--keep-work` | Keep temporary CASA images for debugging |
| `--max-fields 1` | Limit number of stack jobs |
| `--num-tiers 1` | Scan limited tier prefixes when `--tiles` is not supplied |

---

# Recommended quick tests

## Check local discovery only

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --dry-run
```

Expected output should show one stackable job:

```text
[LOCAL TEST MODE] input-dir: ...
[SE alpha tt1/tt0 local-folder] discovered groups: 1; stackable groups with >= 2 maps: 1
DRYRUN SE alpha T08t02 J004201-093000.06.2048: VLASS2.1,VLASS3.1
```

## Run the local alpha test

```bash
casa --nologger -c vlass_fullmap_stack_standalone_v5.py \
  --input-dir . \
  --survey SE \
  --product alpha \
  --epochs 2.1,3.1 \
  --tiles T08t02 \
  --source-id J004201-093000.06.2048 \
  --use-local-rms \
  --rms-box 50 \
  --make-radio-beam-median \
  --keep-in-one-folder \
  --outdir ./stack_test
```

Open the QA report:

```bash
firefox ./stack_test/SE/alpha/T08t02/J004201-093000.06.2048/VLASS_QA_SE_alpha_T08t02_J004201-093000.06.2048.html
```

Check final FITS products:

```bash
ls ./stack_test/all_stacked_products/
```

---

# Notes and cautions

- The local one-folder test mode currently supports SE-style filenames.
- QL alpha is skipped because standard QL products do not contain alpha maps.
- Use `--use-local-rms` when no archive RMS maps are present.
- The weighted mean gives the strongest sensitivity improvement.
- The weighted median is more robust to outliers and single-epoch artifacts, but it usually gives less RMS improvement than the weighted mean.
- Delta images are diagnostic products. They show differences between the final combined map and each matched smoothed epoch map.
- The imdev RMS map is generated from the final combined mean tt0 map in the SE `tt1/tt0` alpha workflow.
- `--keep-in-one-folder` copies final FITS products into one flat folder, but the normal QA/output tree is still created because the weblog needs a structured location.
