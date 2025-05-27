from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.apis.dxo import DXO, DataKind, from_shareable

from custom.utils.training_utils import SemanticSeg

import os
import json
import torch
import numpy as np

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
        self.model = None

    def initialize(self, components, fl_ctx):
        site_id = fl_ctx.get_identity_name()

        if not os.path.isdir(self.data_root):
            raise ValueError(f"Invalid or missing data_root: {self.data_root}")

        print(f"[{site_id}] Initializing data from {self.data_root}...")

        with open(self.split_file) as f:
            splits = json.load(f)
        if site_id not in splits:
            raise ValueError(f"Site '{site_id}' not found in split file: {self.split_file}")

        split = splits[site_id]  # {"trainval": [...], "test": [...]}
        self.trainval_paths = [os.path.join(self.data_root, cid) for cid in split["trainval"]]
        self.test_paths = [os.path.join(self.data_root, cid) for cid in split["test"]]

        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())  # Get run-specific workspace

        self.log_dir = os.path.join(app_dir, "logs", site_id)
        self.output_dir = os.path.join(app_dir, "checkpoints", site_id)
        self.initialized = True

    def train(self, shareable, fl_ctx, abort_signal):
        if not self.initialized:
            self.initialize(None, fl_ctx)

        site_id = fl_ctx.get_identity_name()
        train_set = self.trainval_paths
        val_set = self.test_paths

        print(f"[{site_id}] Starting training with SemanticSeg on {len(train_set)} trainval and {len(val_set)} holdout samples...")

        seg_model = SemanticSeg(
            lr=self.lr,
            n_epoch=self.n_epoch,
            channels=self.channels,
            num_classes=self.num_classes,
            input_shape=(self.input_shape, self.input_shape),
            batch_size=self.batch_size,
            num_workers=2,
            device="0",  # Assuming GPU 0
            pre_trained=False,
            ckpt_point=False,
            use_fp16=False,
            transformer_depth=18,
            use_transfer_learning=True
        )

        seg_model.trainer(
            train_path=train_set,
            val_path=val_set,
            val_ap=None,
            cur_fold=0,
            output_dir=self.output_dir,
            log_dir=self.log_dir,
            phase="seg"
        )

        self.model = seg_model.net

        # ---- Min-Max Fairness Metric Reporting section ----
        # min-max FL expects a SCALAR. Let's build a robust pipeline:
        # Here, metrics_threshold is usually a float, but COULD be a dict if you ever update SemanticSeg.
        mt = seg_model.metrics_threshold

        # Preferred: Maximize worst-case Dice (so min-max FL minimizes 1-Dice)
        if isinstance(mt, dict):
            # Your SemanticSeg could someday return a dict
            dice_score = mt.get("run_dice", None)
            if dice_score is not None:
                fair_metric = 1.0 - float(dice_score)
                print(f"[{site_id}] (dict) Using min-max FL fair_metric = 1-dice = {fair_metric}")
            else:
                fair_metric = 1.0
                print(f"[{site_id}] Dice not found in metrics_threshold dict, setting fair_metric = 1.0")
        elif isinstance(mt, (float, int, np.floating)):
            fair_metric = 1.0 - float(mt)   # mt is the best Dice if unchanged model
            print(f"[{site_id}] (float) Using min-max FL fair_metric = 1-dice = {fair_metric}")
        else:
            fair_metric = 1.0
            print(f"[{site_id}] ERROR: metrics_threshold is {type(mt)}, setting fair_metric = 1.0")

        # If you want to use val_loss instead, just replace the above block with:
        # if isinstance(mt, dict):
        #     val_loss = mt.get("val_loss", None)
        #     fair_metric = float(val_loss) if val_loss is not None else 1.0
        #     print(f"[{site_id}] Using min-max FL fair_metric = val_loss = {fair_metric}")
        # elif isinstance(mt, (float, int, np.floating)):
        #     fair_metric = float(mt)
        # elif:
        #     fair_metric = 1.0

        # ---- Prepare the model diff for NVFlare, add fair_metric in meta ----
        global_weights = from_shareable(shareable).data
        local_weights = self.model.state_dict()
        model_diff = {}
        for k in local_weights:
            if k in global_weights:
                model_diff[k] = local_weights[k].cpu().numpy() - global_weights[k]
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop("fair_metric", fair_metric)   # min-max aggregator uses this!
        return dxo.to_shareable()

    def get_model(self):
        return self.model

    def set_model(self, model):
        if self.model is None:
            print("Model hasn't been trained yet; creating placeholder model.")
            self.model = model  # Or reconstruct same architecture here
        else:
            self.model.load_state_dict(model.state_dict())

    def finalize(self, fl_ctx):
        print("[FL Learner] Finalizing resources.")
