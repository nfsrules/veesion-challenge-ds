# main.py
import argparse
import logging
import os
from optimizer.io_utils import load_dataframe
from optimizer import CameraModel, MultiCameraOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def main(args):
    """
    Main entry point for running the multi-camera optimization.

    Steps:
        1. Load data from --source.
        2. Filter for a single camera if --store and --camera_id are provided.
        3. Run optimization with the requested --target_fp_reduction.
        4. Save results if --save_path is specified.
    """

    df = load_dataframe(args.source)

    if args.camera_id:
        if not args.store:
            logging.error("Must provide --store when using --camera_id")
            return
        df = df[(df['store'] == args.store) & (df['camera_id'] == args.camera_id)]
        if df.empty:
            logging.error(f"No data found for store '{args.store}' and camera_id '{args.camera_id}'")
            return
        logging.info(f"Running optimization for store '{args.store}', camera_id '{args.camera_id}'")
    else:
        logging.info("Running optimization for all cameras")

    optimizer = MultiCameraOptimizer(df, CameraModel, verbose=not args.quiet)
    optimizer.run(target_fp_reduction=args.target_fp_reduction)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        optimizer.save(args.save_path)
        logging.info(f"Saved optimizer state to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Camera Optimizer")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Data source (e.g., CSV path)"
    )
    parser.add_argument(
        "--target_fp_reduction", type=int, default=100,
        help="Desired number of false positives to reduce globally (default: 100)"
    )
    parser.add_argument(
        "--store", type=str,
        help="Store name (required if --camera_id is provided)"
    )
    parser.add_argument(
        "--camera_id", type=str,
        help="Camera ID (e.g., '1') to run for a single camera; requires --store"
    )
    parser.add_argument(
        "--save_path", type=str, default="",
        help="Path to save the optimizer's results (JSON). If empty, results won't be saved."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed logs"
    )

    args = parser.parse_args()
    main(args)