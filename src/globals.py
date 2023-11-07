default_config = {
    "accumulate_grad_batches": 1,
    "backbone_kwargs": {
        "architecture": "convlstm", #
        #"layer_widths": "[4096, 1024]",
        #"nonlinearity": "gelu",
    },
    "batch_size_train": 32,
    "batch_size_val": 32,
    # "batch_size_finetune": 1024,
    "check_val_every_n_epoch": 5,  # 50,
    "dataset_dir": "data",
    "gradient_clip_val": None,
    "num_workers": 8,
    "precision": 32,
    "prefetch_factor": 2,
    "dataset": "cifar10",
    "dataset_kwargs": {
         "n_bodies": 1,  
        # "n_dims": 10,
        # "data_spectrum": "power",  # "uniform" or "exponential" or "power"
        # "data_spectrum_parameter": 0.99,
        # "data_spectrum_magnitude": 10,
        # "transformation_spectrum": "power",  # "uniform" or "exponential" or "power"
        # "transformation_spectrum_parameter": 256,
        # "transformation_spectrum_magnitude": 1,
    },
    "seed": 0,
    "shuffle_train": True,
    "reservoir_kwargs": {
        "dim": 64,
        "sparsity": 0.95,
        "spectral_radius": 0.95,
        "leak_rate": 0.5,
        "bias": False,
        "random_projection": "structured",
        "sparse_reservoir": True,
        "device": "cuda",
    },
    }
}