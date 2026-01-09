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

    env = lmdb.open(lmdb_path, map_size=500*1024**3)
    # Use mutable container to allow modification in nested function
    state = {"txn": env.begin(write=True), "write_count": 0}

    def safe_put(key, filepath):
        """Store file in LMDB if it exists and not already stored."""
        if state["txn"].get(key) is not None:
            return  # Already stored
        if not os.path.exists(filepath):
            return  # File doesn't exist
        state["txn"].put(key, open(filepath, 'rb').read())
        state["write_count"] += 1
        if state["write_count"] % COMMIT_EVERY == 0:
            state["txn"].commit()
            state["txn"] = env.begin(write=True)
            print(f"  Committed {state['write_count']} files...")

    # 1. Images: rgb and rgb_augmented
    print("Storing images...")
    for sample_images in tqdm(dataset.images, desc="Images"):
        for path_bytes in sample_images:
            path = str(path_bytes, encoding="utf-8")
            safe_put(path.encode(), path)

            aug_path = path.replace("rgb", "rgb_augmented")
            safe_put(aug_path.encode(), aug_path)

    # 2. Boxes
    print("Storing boxes...")
    for sample_boxes in tqdm(dataset.boxes, desc="Boxes"):
        for path_bytes in sample_boxes:
            path = str(path_bytes, encoding='utf-8')
            safe_put(path.encode(), path)

    # 3. Measurements + 4. Commentary + 5. QA/VQA
    print("Storing measurements, commentary, and QA...")
    for idx in tqdm(range(len(dataset)), desc="Measurements/Commentary/QA"):
        meas_dir = str(dataset.measurements[idx, 0], encoding='utf-8')
        sample_start = dataset.sample_start[idx]

        for i in range(dataset.hist_len + dataset.pred_len):
            # Measurement file
            meas_path = f"{meas_dir}/{(sample_start + i):04}.json.gz"
            safe_put(meas_path.encode(), meas_path)

            # Commentary file (derived from measurement path)
            commentary_path = meas_path.replace('measurements', 'commentary').replace('data/', 'commentary/')
            safe_put(commentary_path.encode(), commentary_path)

            # QA/VQA file (derived from measurement path)
            qa_path = meas_path.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            safe_put(qa_path.encode(), qa_path)

    # 6. Dreamer alternative trajectories
    print("Storing dreamer trajectories...")
    for path_bytes in tqdm(dataset_dreamer.alternative_trajectories, desc="Dreamer"):
        path = str(path_bytes, encoding='utf-8')
        safe_put(path.encode(), path)

    # 7. Dreamer dataset has different samples - store its images/boxes/measurements too
    print("Storing dreamer dataset images...")
    for sample_images in tqdm(dataset_dreamer.images, desc="Dreamer Images"):
        for path_bytes in sample_images:
            path = str(path_bytes, encoding="utf-8")
            safe_put(path.encode(), path)
            aug_path = path.replace("rgb", "rgb_augmented")
            safe_put(aug_path.encode(), aug_path)

    print("Storing dreamer dataset boxes...")
    for sample_boxes in tqdm(dataset_dreamer.boxes, desc="Dreamer Boxes"):
        for path_bytes in sample_boxes:
            path = str(path_bytes, encoding='utf-8')
            safe_put(path.encode(), path)

    print("Storing dreamer dataset measurements...")
    for idx in tqdm(range(len(dataset_dreamer)), desc="Dreamer Measurements"):
        meas_dir = str(dataset_dreamer.measurements[idx, 0], encoding='utf-8')
        sample_start = dataset_dreamer.sample_start[idx]
        for i in range(dataset_dreamer.hist_len + dataset_dreamer.pred_len):
            meas_path = f"{meas_dir}/{(sample_start + i):04}.json.gz"
            safe_put(meas_path.encode(), meas_path)

    # Final commit
    state["txn"].commit()
    env.close()
    print(f"Creating LMDB dataset finished. Total files: {state['write_count']}")

