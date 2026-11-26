# Largeimage_StitchingandAnalysis

Pipeline for stitching ND2 points into a large image, correcting the illumination based on the tile overlaps, and analysing intensity after segmentation.

Turns Nikon ND2 large images (tiled multipoint **or** already stitched by NIS) into:
- the max projection of each channel (data TIFF and display PNG)
- a merged colour image
- intensity-vs-X profiles averaged only over segmented cells
- a QC folder that follows every processing step

All images in one experiment config share the same settings, display ranges and thresholds.

## Install

Python 3.12. For GPU segmentation (strongly recommended), first install the CUDA build of PyTorch for your GPU from https://pytorch.org. Then:

```
pip install -r requirements.txt
```

## Run

```
python run_pipeline.py experiment_test.yaml --channels            # only show channels/labels/colours of the files
python run_pipeline.py experiment_test.yaml                       # first run: everything
python run_pipeline.py experiment_test.yaml                       # re-run: reuses mosaics + Cellpose masks
python run_pipeline.py experiment_test.yaml --force-segmentation  # redo Cellpose only
python run_pipeline.py experiment_test.yaml --force               # redo everything
```

For a new experiment, copy `experiment_test.yaml` and edit it. It lists every option with a comment, and any line you delete uses the default. The main options are:
- **`images`**: list the ND2 files.
- **`merge_channels`**: `ask` prompts once at the start (Enter = all channels, `none` = no merge). You can also write a fixed list such as `[DAPI, RFP]`. The merged profile plot uses the same channels.
- **`flatfield.reference`**: the flat-field reference ND2 for that session (see below).
- **`channels`** (optional): the labels and colours are taken from the ND2 channel names automatically. A section written for one microscope is ignored for names that do not occur in the files.
- **`segmentation.channels`**: `[DAPI]` for nuclei. For cytoplasmic signal use `[Phalloidin, DAPI]`. If a listed channel is not in the data, it is skipped with a warning. If none are found, the nuclear channel is used: its name contains DAPI, Hoechst or 405, or else it is the channel with the shortest emission wavelength.
- All images in one config must have the same channels. The run stops at the start if they don't.
- **`display: scale_bar`** (optional): a scale bar is drawn on the coloured PNGs and JPG previews, with the same length for every image in the experiment. `length_um: auto` picks a round length of about 12% of the image width. You can instead give a number, set `position: bottom-left`, `color: black`, or `enabled: false`. The data TIFFs have no bar but carry the µm calibration, so in Fiji use Analyze → Tools → Scale Bar.
- **`reuse_settings`**: the path to an earlier run's `experiment/experiment_settings.json`. A batch processed later then uses exactly the same display ranges and segmentation normalisation.

Runtime for the test data is about 8 min for both images, on the GPU.

## What happens

| Step | Tiled ND2 | Pre-stitched ND2 |
|---|---|---|
| Max projection over Z | ✓ (QC 01: which Z plane gave the max) | ✓ |
| Camera offset (ORCA-Fusion = 100) | ✓ auto | ✓ auto (zero padding ignored) |
| Per-tile background, subtracted before the flat-field | ✓ | – |
| Flat-field (laser profile) | ✓ reference images, or fitted from overlaps (QC 03) | ✗ no tile layout in the file |
| Stitching | stage positions → phase correlation + NCC check → global fit → feather blend (QC 02, 04) | – |
| Cellpose-SAM segmentation, in blocks | ✓ (QC 05, 06) | ✓ |
| Small/dim objects removed (`min_area_um2`, `min_norm_intensity`) | ✓ | ✓ |
| Local background per X-bin from cell-free pixels | ✓ (QC 07, 08) | ✓ |
| Per-cell table, X profiles, images, plots | ✓ | ✓ |

## Outputs (`output/<experiment>/`)

```
<image>/channels/   <image>_<ch>.tif        corrected float32 data (µm calibrated, for ImageJ/Fiji)
                    <image>_<ch>.png        coloured, experiment-wide display range
                    <image>_<ch>_gray8.png, _preview.jpg
<image>/merged/     <image>_merge_<chs>.png          full resolution, with scale bar (for viewing)
                    <image>_merge_<chs>.tif          full resolution, no bar, µm calibrated (for Fiji / figures)
                    <image>_merge_<chs>_preview.jpg  4000 px wide, with scale bar (quick look, slides)
<image>/profiles/   one plot per channel (raw intensity)
                    overlay of the merge channels (normalised, see below)
                    <image>_merge_with_profile_<chs>.png  merged image with the overlay plot below, same X axis
<image>/tables/     <image>_cells.csv       one row per cell (position, area, raw/bkg/net intensity, outlier flag)
                    <image>_profile_x.csv   per X-bin stats (mean, median, SEM, n cells, pixel mean, coverage)
<image>/QC/         numbered figures in processing order
<image>/data/       corrected mosaic, labels, flat-field, valid mask (cache)
experiment/         compare_images_<ch>.png, all_profiles_x.csv, experiment_settings.json, summary.txt, log
```

## Profile definition

The data are binned along X (`profile.bin_um`, default 50 µm). In each bin:
- **Per-cell mean (solid line ± SEM):** each cell counts once, using its mean intensity minus the local background.
- **Pixel mean (dashed line):** the mean over all cell pixels in the bin.
- **Bins with fewer than `min_cells_per_bin` cells** are left as gaps and shaded grey.
- **Cell count:** shown in the lower panel.
- **Outlier cells** (debris, more than 20 robust SDs above the median) are excluded from both averages. They stay in the cell table with a flag.
- **`profile.statistic: median`** plots the median instead of the mean.
- **Colour-blind safe plots:** each channel differs by colour, line style and marker shape: circle, square, triangle, star, diamond, and so on, in the channel order of the ND2. A channel keeps its shape in every plot. In the comparison plots across images, each image has its own shape.
- **Normalisation in overlay plots:** in plots with several channels, each channel is divided by its average cell across the whole experiment. On the y-axis, 1 means the average cell of that channel, so a weak channel is no longer flattened by DAPI. The same factor is used for every image, so the images can still be compared with each other. Set `profile.overlay_normalize` to `image_mean` (each image scaled to its own average) or `none` (raw values). The factors are saved in `experiment_settings.json`, and the profile CSV has extra `*_norm` columns. The single-channel plots stay raw.

## Flat-field: important for acquisition

- **Fitting from overlaps works here, but it has limits.** Measured on the test data, the illumination drops to about 0.72 (DAPI), 0.77 (GFP) and 0.74 (RFP) at the field edges. The overlap fit corrects the X direction well: the overlap ratio goes from 0.85–1.33 to 1.00 ± 0.01. However:
  - In a single-row scan the **Y profile is not constrained**, so it is assumed flat.
  - **Photobleaching of the second exposure** (~4–5% for DAPI/GFP) cannot be separated from an off-centre beam.
- **Recommended:** record a flat-field reference every imaging session. Image a fluorescent slide or a dye layer with each laser line and the same objective, at about 10–20 positions, and set it as `flatfield.reference`. With 10–20% overlap this is enough; 30% also works.
- **Prefer tiled (multipoint) acquisitions.** NIS-stitched images cannot be illumination-corrected here.
- **Weak channels borrow the session's correction.** Sometimes a channel has too little signal in its own overlaps to fit the illumination, for example a marker that is off in part of the sample. That channel then gets the fit pooled from the other tiled images of the same session, but only when both of these hold:
  1. **Same session:** both images were acquired on the same day, with the same objective, camera, binning, pixel size, tile size and channels. This is read from the ND2 files.
  2. **The image's other channels match the session:** the image's well-measured channels must agree with the rest of the session within 3% along X and 5% over the field. Same-day images differ by ≤ 1.6% and ≤ 3.1%. This confirms the optics really were the same for this image.

  Otherwise the channel is left uncorrected, with a warning. `QC/03_flatfield_session_check.png` shows the comparison and the decision. To switch this off, set `flatfield.session_pooling: false`. The limits are `session_check_x_rms` and `session_check_field_rms`.
