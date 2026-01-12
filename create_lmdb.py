"""
NOTE: Remember to set split_percentage=1.0 in dataset_base.BaseDataset when running this script,
      in order to make sure that the whole dataset is covered
"""

from tqdm import tqdm
import lmdb
import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import hydra.utils

# Import config module to register structured configs with ConfigStore
import simlingo_training.config  # noqa: F401

# Patch get_original_cwd since it doesn't work with compose API
hydra.utils.get_original_cwd = lambda: os.path.abspath(".")

from simlingo_training.dataloader.dataset_base import BaseDataset


if __name__ == "__main__":

    GlobalHydra.instance().clear()
    config_dir = os.path.abspath("simlingo_training/config")
    initialize_config_dir(config_dir=config_dir, version_base="1.1")
    cfg = compose(config_name="config", overrides=["experiment=simlingo_seed1"])

    # Initialize the BaseDataset, which will take care of loading all the indexes (file paths)
    # The patched get_original_cwd() will resolve relative paths in BaseDataset.__init__
    data_module_cfg = OmegaConf.to_container(cfg.data_module, resolve=True)
    base_dataset_cfg = OmegaConf.to_container(cfg.data_module.base_dataset, resolve=True)

    # Disable LMDB loading - we're creating it, not reading from it
    base_dataset_cfg["use_lmdb"] = False

    print("Initializing driving dataset...")
    dataset = BaseDataset(
        split="train",
        bucket_name="all",
        dreamer=False,
        **data_module_cfg,
        **base_dataset_cfg,
    )

    # Second instance with dreamer=True to get alternative_trajectories paths
    # (dreamer data reuses same images/boxes/measurements, just adds trajectory files)
    print("Initializing dreamer dataset (for trajectory paths only)...")
    dataset_dreamer = BaseDataset(
        split="train",
        bucket_name="all",
        dreamer=True,
        **data_module_cfg,
        **base_dataset_cfg,
    )

    # Convert to LMDB with periodic commits for resumability
    lmdb_path = "/cluster/projects/vc/data/ad/open/write-folder/simlingo/lmdb_dataset"
    COMMIT_EVERY = 250000  # Commit every N files

    env = lmdb.open(lmdb_path, map_size=1600*1024**3)
    # Use mutable container to allow modification in nested function
    state = {"txn": env.begin(write=True), "write_count": 0}

    def safe_put(key, filepath):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return
        try:
            state["txn"].put(key, data, overwrite=False)
            state["write_count"] += 1
        except lmdb.KeyExistError:
            return

        if state["write_count"] % COMMIT_EVERY == 0:
            state["txn"].commit()
            state["txn"] = env.begin(write=True)
            print(f"  Committed {state['write_count']} files...")

    # Collect unique route directories from both datasets
    print("Collecting unique route directories...")
    route_dirs = set()
    for idx in range(len(dataset)):
        meas_dir = str(dataset.measurements[idx, 0], encoding='utf-8')
        # Get route dir (parent of measurements/)
        route_dir = os.path.dirname(meas_dir)
        route_dirs.add(route_dir)
    for idx in range(len(dataset_dreamer)):
        meas_dir = str(dataset_dreamer.measurements[idx, 0], encoding='utf-8')
        route_dir = os.path.dirname(meas_dir)
        route_dirs.add(route_dir)
    print(f"Found {len(route_dirs)} unique routes")

    # For each route, store all files in rgb/, rgb_augmented/, boxes/, measurements/
    # and derived commentary/QA paths
    for route_dir in tqdm(route_dirs, desc="Routes"):
        # 1. Images: rgb and rgb_augmented
        rgb_dir = os.path.join(route_dir, "rgb")
        rgb_aug_dir = os.path.join(route_dir, "rgb_augmented")
        if os.path.isdir(rgb_dir):
            for filename in os.listdir(rgb_dir):
                rgb_path = os.path.join(rgb_dir, filename)
                safe_put(rgb_path.encode(), rgb_path)
                # Augmented version
                aug_path = os.path.join(rgb_aug_dir, filename)
                safe_put(aug_path.encode(), aug_path)

        # 2. Boxes
        boxes_dir = os.path.join(route_dir, "boxes")
        if os.path.isdir(boxes_dir):
            for filename in os.listdir(boxes_dir):
                box_path = os.path.join(boxes_dir, filename)
                safe_put(box_path.encode(), box_path)

        # 3. Measurements + 4. Commentary + 5. QA/VQA
        meas_dir = os.path.join(route_dir, "measurements")
        if os.path.isdir(meas_dir):
            for filename in os.listdir(meas_dir): #NOTE: This code assumes that measurements is the superset (none of the other directories have MORE filenames in them)
                if not filename.endswith('.json.gz'):
                    continue
                meas_path = os.path.join(meas_dir, filename) 
                safe_put(meas_path.encode(), meas_path)

                # Commentary (derived path)
                commentary_path = meas_path.replace('measurements', 'commentary').replace('data/', 'commentary/')
                safe_put(commentary_path.encode(), commentary_path)

                # QA/VQA (derived path)
                qa_path = meas_path.replace('measurements', 'vqa').replace('data/', 'drivelm/')
                safe_put(qa_path.encode(), qa_path)

        # 6. Dreamer trajectories 
        dreamer_dir = route_dir.replace('data/', 'dreamer/') + '/dreamer'
        if os.path.isdir(dreamer_dir):
            for filename in os.listdir(dreamer_dir):
                if filename.endswith('.json.gz'):
                    dreamer_path = os.path.join(dreamer_dir, filename)
                    safe_put(dreamer_path.encode(), dreamer_path)

    # Final commit
    state["txn"].commit()
    env.close()
    print(f"Creating LMDB dataset finished. Total files: {state['write_count']}")

