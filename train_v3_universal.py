"""Legacy entry point. Use the explicit, split-aware trainer; no automatic calibration."""
if __name__ == "__main__":
    import runpy
    print("Legacy training replaced. See python train.py --help and docs/TRAINING.md.")
    runpy.run_module("train", run_name="__main__")
