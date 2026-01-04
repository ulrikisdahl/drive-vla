import os
import shutil
import hydra

from omegaconf import OmegaConf
import torch
import wandb

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback
from hydra.utils import get_original_cwd
from tqdm import tqdm


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # turn off wandb uploading when in debug mode
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
    
    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"


    resume_path = cfg.resume_path
    valid_path_test = False
    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        valid_path_test = True
    else:
        valid_path_test = False
    with open("valid_path_test.txt", "w") as f:
        f.write(str(valid_path_test))

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None #f"pretrained/{(model_type_name)}"

    if cfg.data_module.base_dataset.use_data_prestage:
        repo_root = get_original_cwd()
        staged_root = os.path.join("/tmp", cfg.data_module.base_dataset.dataset_prestage_name)

        def _resolve_src_path(path_value):
            if os.path.isabs(path_value):
                return path_value
            return os.path.join(repo_root, path_value)

        def _iter_files_with_sizes(root_dir):
            for root, _, files in os.walk(root_dir):
                for name in files:
                    src_path = os.path.join(root, name)
                    try:
                        size = os.path.getsize(src_path)
                    except OSError:
                        size = 0
                    yield src_path, size

        def _copytree_with_progress(src, dst):
            os.makedirs(dst, exist_ok=True)
            files = list(_iter_files_with_sizes(src))
            total_bytes = sum(size for _, size in files)
            with tqdm(total=total_bytes, unit="B", unit_scale=True, ascii=True) as pbar:
                for src_path, size in files:
                    rel_path = os.path.relpath(src_path, src)
                    dst_path = os.path.join(dst, rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    pbar.update(size)

        def _stage_path(path_value):
            src = _resolve_src_path(path_value)
            dst = os.path.join(staged_root, path_value.lstrip(os.sep))
            if os.path.isdir(dst) and not cfg.data_module.base_dataset.dataset_prestage_overwrite:
                print(f"Prestage: using existing {dst}")
                return dst
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            print(f"Prestage: copying {src} -> {dst}")
            _copytree_with_progress(src, dst)
            return dst

        staged_data_path = _stage_path(cfg.data_module.base_dataset.data_path)
        staged_bucket_path = _stage_path(cfg.data_module.base_dataset.bucket_path)
        cfg.data_module.base_dataset.data_path = staged_data_path
        cfg.data_module.base_dataset.bucket_path = staged_bucket_path
        print(f"Prestage enabled: True ({staged_root})")
    else:
        print("Prestage enabled: False")
    if cfg.data_module.base_dataset.use_data_prestage and cfg.data_module.base_dataset.use_disk_cache:
        cfg.data_module.base_dataset.use_disk_cache = False
        print("Disk cache disabled because pre-staging is enabled.")

    data_cache = None
    data_cache_dir = None
    data_cache_size_bytes = None
    if cfg.data_module.base_dataset.use_disk_cache:
        from diskcache import Cache
        data_cache_dir = os.path.join("/tmp", cfg.data_module.base_dataset.dataset_cache_name)
        data_cache_size_bytes = int(cfg.data_module.base_dataset.dataset_cache_size_gb * (1024 ** 3))
        data_cache = Cache(data_cache_dir, size_limit=data_cache_size_bytes)
        print(f"Disk cache enabled: True ({data_cache_dir}, size_limit={data_cache_size_bytes} bytes)")
    else:
        print("Disk cache enabled: False")

    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        data_cache=data_cache,
        data_cache_dir=data_cache_dir,
        data_cache_size_bytes=data_cache_size_bytes,
        _recursive_=False
    )

    if cfg.data_module.base_dataset.use_disk_cache and cfg.data_module.base_dataset.cache_warmup:
        if getattr(data_module, "train_dataset", None) is None:
            data_module.setup("fit")
        train_dataset = data_module.train_dataset
        if train_dataset is not None:
            warmup_total = len(train_dataset)
            warmup_limit = cfg.data_module.base_dataset.cache_warmup_num_samples
            if warmup_limit is not None:
                warmup_total = min(warmup_total, warmup_limit)
            print(f"Cache warmup: start (samples={warmup_total})")
            for idx in tqdm(range(warmup_total), unit="sample", ascii=True):
                _ = train_dataset[idx]
            print("Cache warmup: done")

    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True
    
    valid_path_test = False
    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
        valid_path_test = True
    else:
        valid_path_test = False
        resume_path = None

    with open("valid_path_test.txt", "w") as f:
        f.write(str(valid_path_test))

    # setup lightning logger
    loggers = []
    # csvlogger = CSVLogger("log/", "CSVLogger")
    # loggers.append(csvlogger)
    # csvlogger = None

    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    wandblogger.watch(model)
    loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath="./checkpoints",
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=1,  #cfg.val_every_n_epochs,
        # every_n_train_steps=cfg.val_check_interval,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        checkpoint_callback, 
        model_summary, 
        # ThroughputMonitor(batch_size_fn=lambda batch: batch.driving_input.camera_images.size(0)), 
        VisualiseCallback(interval=1000, val_interval=1000)
    ]
    if not cfg.debug: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            # enable_checkpointing=False,
            gradient_clip_val=0.3,
            # gradient_clip_algorithm="value",
            # log_every_n_steps=10,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
            # val_check_interval=cfg.val_check_interval,
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)
    wandb.finish()

if __name__ == "__main__":
    main()
