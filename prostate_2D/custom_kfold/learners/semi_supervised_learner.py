import os
import json
import torch
from collections import defaultdict

from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.apis.dxo import DXO, DataKind, from_shareable

from custom.utils.training_utils import SemanticSeg  # <- Note kfold!

class SemiSupervisedLearner(Learner, ModelLearner):
    def __init__(self, **config):
        Learner.__init__(self)
        ModelLearner.__init__(self)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epoch = config.get("n_epoch", 50)
        self.batch_size = config.get("batch_size", 4)
        self.lr = config.get("lr", 1e-4)
        self.channels = config.get("n_channels", 3)
        self.num_classes = config.get("n_classes", 2)
        self.input_shape = config.get("image_size", 384)
        self.split_file = config.get("split_file")
        self.data_root = config.get("data_root")
        self.initialized = False
        self.model = None  # Will be set after training

    def initialize(self, components, fl_ctx):
        site_id = fl_ctx.get_identity_name()
        if not os.path.isdir(self.data_root):
            raise ValueError(f"Invalid or missing data_root: {self.data_root}")

        print(f"[{site_id}] Initializing data from {self.data_root}...")

        with open(self.split_file) as f:
            splits = json.load(f)
        self.folds = splits[site_id]  # {"fold_0": {"train": [...], "val": [...]}, ...}

        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())  # Get run-specific workspace

        self.log_dir = os.path.join(app_dir, "logs", site_id)
        self.output_dir = os.path.join(app_dir, "checkpoints", site_id)
        self.initialized = True

    def average_model_weights(self, models):
        """Helper to average model weights (PyTorch state_dicts)"""
        avg_state_dict = defaultdict(float)
        n = len(models)
        # Assumption: all models have identical keys!
        state_dicts = [m.state_dict() for m in models]
        for key in state_dicts[0]:
            avg_tensor = sum([sd[key].float() for sd in state_dicts]) / n
            avg_state_dict[key] = avg_tensor
        # Create new model and load averaged weights
        new_model = type(models[0])(*models[0].args, **models[0].kwargs) if hasattr(models[0], 'args') else models[0].__class__()  # fallback
        new_model.load_state_dict(avg_state_dict)
        return new_model

    def train(self, shareable, fl_ctx, abort_signal):
        if not self.initialized:
            self.initialize(None, fl_ctx)
        site_id = fl_ctx.get_identity_name()

        # Prepare to accumulate metrics and models
        all_fold_metrics = []
        all_fold_models = []

        # Loop over folds
        for fold_id, sets in self.folds.items():
            # Compose full paths for each file
            train_set = [os.path.join(self.data_root, fn) for fn in sets["train"]]
            val_set = [os.path.join(self.data_root, fn) for fn in sets["val"]]
            print(f"[{site_id}] Training fold {fold_id} ({len(train_set)} train, {len(val_set)} val)")

            seg_model = SemanticSeg(
                lr=self.lr,
                n_epoch=self.n_epoch,
                channels=self.channels,
                num_classes=self.num_classes,
                input_shape=(self.input_shape, self.input_shape),
                batch_size=self.batch_size,
                num_workers=2,
                device="0",
                pre_trained=False,
                ckpt_point=False,
                use_fp16=False,
                transformer_depth=18,
                use_transfer_learning=True,
            )
            # Train on this fold
            seg_model.trainer(
                train_path=train_set,
                val_path=val_set,
                val_ap=None,
                cur_fold=int(fold_id.split("_")[-1]),
                output_dir=self.output_dir,
                log_dir=self.log_dir,
                phase="seg",
            )
            # From repo: metrics_threshold stores per-key metrics (e.g. val_dice)
            mt = seg_model.metrics_threshold
            if isinstance(mt, dict):
                metric = mt.get("val_dice", None)
            else:
                metric = mt
            if metric is not None:
                all_fold_metrics.append(metric)
            else:
                print(f"[{site_id}] Warning: val_dice not found in metrics_threshold for fold {fold_id}")
            all_fold_models.append(seg_model.net.cpu())

        # Average metrics and model weights
        avg_metric = sum(all_fold_metrics) / len(all_fold_metrics) if all_fold_metrics else None

        # Weight averaging (PyTorch): average state dicts for all models
        # We'll use one model as the "template"
        avg_state_dict = None
        if all_fold_models:
            state_dicts = [m.state_dict() for m in all_fold_models]
            avg_state_dict = {}
            for key in state_dicts[0].keys():
                avg_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
            # Load the averaged weights into a copy of the last model
            all_fold_models[0].load_state_dict(avg_state_dict)
            self.model = all_fold_models[0]
        else:
            print(f"[{site_id}] No trained models collected for averaging.")
            self.model = None

        print(f"[{site_id}] K-fold val_dice scores: {all_fold_metrics}, average: {avg_metric}")

        # NVFlare model update (weight diff as in original)
        # (copied from earlier code)
        global_weights = from_shareable(shareable).data
        local_weights = self.model.state_dict() if self.model is not None else {}

        model_diff = {}
        for k in local_weights:
            if k in global_weights:
                model_diff[k] = local_weights[k].cpu().numpy() - global_weights[k]
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        return dxo.to_shareable()

    def get_model(self):
        return self.model

    def set_model(self, model):
        if self.model is None:
            print("Model hasn't been trained yet; creating placeholder model.")
            self.model = model
        else:
            self.model.load_state_dict(model.state_dict())

    def finalize(self, fl_ctx):
        print("[FL Learner] Finalizing resources.")
