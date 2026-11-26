"""Run the ND2 large-image pipeline on an experiment config.

    python run_pipeline.py experiment.yaml
    python run_pipeline.py experiment.yaml --channels             # only show what is in the files
    python run_pipeline.py experiment.yaml --force                # redo everything
    python run_pipeline.py experiment.yaml --force-segmentation   # redo Cellpose only
"""
import argparse

from nd2pipe.pipeline import describe_channels, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--channels", action="store_true",
                    help="print channels / labels / colours / segmentation channel of each image, then stop")
    ap.add_argument("--force", action="store_true", help="ignore cached mosaics and segmentations")
    ap.add_argument("--force-segmentation", action="store_true", help="re-run Cellpose only")
    a = ap.parse_args()
    if a.channels:
        describe_channels(a.config)
    else:
        run(a.config, force=a.force, force_segmentation=a.force_segmentation)
