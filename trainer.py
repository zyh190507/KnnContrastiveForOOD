# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import fitlog
import warnings
import random
import copy
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    hp_params,
    is_azureml_available,
    is_comet_available,
    is_fairscale_available,
    is_mlflow_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)

from utils import *

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    HPSearchBackend,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
)

from training_args import TrainingArguments
from transformers.utils import logging

# Evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
else:
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_mlflow_available():
    from transformers.integrations import MLflowCallback

    DEFAULT_CALLBACKS.append(MLflowCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

if is_azureml_available():
    from transformers.integrations import AzureMLCallback

    DEFAULT_CALLBACKS.append(AzureMLCallback)

if is_fairscale_available():
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

logger = logging.get_logger(__name__)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


class MyEvalPrediction(NamedTuple):

    prediction_by_knn: Union[np.ndarray, Tuple[np.ndarray]]
    prediction_by_cls: Union[np.ndarray, Tuple[np.ndarray]]
    prediction_combine: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class SimpleTrainer:
    """
    This Train is simple version for Origin Trainer.
    This is to exce the origin contrastive learning.
    """
    def __init__(self,
                 model: Union[PreTrainedModel, torch.nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 tokenizer: Optional["PreTrainedTokenizerBase"] = None,
                 number_labels: Optional[int] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[MyEvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if args is None:
            logger.info("No 'TrainingArgumenets' passed, using the current path as 'output_dir'" )
            args = TrainingArguments("tmp_trainer")
        self.args = args
        set_seed(self.args.seed)

        self.number_labels = number_labels
        self.model = model
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        filepath = os.path.join(args.output_dir, 'model_best.pkl')
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.lr_scheduler = optimizers
        #self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.args.clip = args.clip
        self.number_labels = number_labels
        self.sharded_dpp = False

        self.negative_data = self.create_negative_dataset()
        self.negative_keys = list(self.negative_data.keys())

    def create_negative_dataset(self):
        negative_dataset = {}
        data = self.train_dataset

        for line in data:
            label = int(line["label"])
            inputs = line
            inputs.pop("original_text")
            inputs.pop("sent_id")
            inputs.pop("label")
            inputs.pop("text")
            if label not in negative_dataset.keys():
                negative_dataset[label] = [inputs]
            else:
                negative_dataset[label].append(inputs)

        return negative_dataset


    def generate_positive_sample(self, label: torch.Tensor):
        positive_num = self.args.positive_num # 3
        # positive_num = 16
        positive_sample = []
        for index in range(label.shape[0]):
            input_label = int(label[index])
            positive_sample.extend(random.sample(self.negative_data[input_label], positive_num))

        return self.list_item_to_tensor(positive_sample)

    def list_item_to_tensor(self, inputs_list: List[Dict]):
        batch_list = {}
        for key, value in inputs_list[0].items():
            batch_list[key] = []
        for inputs in inputs_list:
            for key, value in inputs.items():
                batch_list[key].append(value)

        batch_tensor = {}
        for key, value in batch_list.items():
            batch_tensor[key] = torch.tensor(value)
        return batch_tensor

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=AdamW,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            else:
                #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset dese not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def valution_cal(self, model, val_dataloader) -> float:
        #model.eval()
        target =[]
        predict = []
        for step, inputs in enumerate(val_dataloader):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            output = model(inputs, mode='validation')
            predict += output[0]
            target += output[1]

        f1 = metrics.f1_score(target, predict, average='macro')
        return f1

    def train_mocoknn(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        model = self.model
        model.to(self.args.device)

        ################################################################################################
        train_dataloader = self.get_train_dataloader()
        valid_loader = self.get_eval_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        max_steps = math.ceil(self.args.supcont_pre_epoches * num_update_steps_per_epoch)
        self.args.warmup_steps = max_steps * 0.1
        sup_con_num_train_epochs = self.args.supcont_pre_epoches

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        #tr_loss = torch.tensor(0.0).to(self.args.device)
        tr_loss = torch.tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0

        best_f1 = 0
        global_step = 0
        model_best_param_dict = None
        for epoch in range(sup_con_num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, initial=global_step, desc="Iter (sup_loss)")
            model.train()
            for step, inputs in enumerate(epoch_iterator):
                positive_sample = None
                positive_sample = self.generate_positive_sample(inputs["labels"])
                positive_sample = self._prepare_inputs(positive_sample)

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                outputs = model(inputs, mode='train', positive_sample=positive_sample)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                epoch_iterator.set_description('Iter (sup_cont_loss=%5.3f)' % loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                self.optimizer.step()
                self.lr_scheduler.step()
                #model.zero_grad()

                global_step += 1
                tr_loss += loss.item()
            print('Epoch: [{0}]: Loss {loss:.4f}'.format(epoch, loss=tr_loss / global_step))
            cl_loss_name = 'sup_Clearning_loss'
            fitlog.add_loss(tr_loss / global_step, name=cl_loss_name, step=epoch)
            ### Get model performing in valid IND #####
            f1 = self.valution_cal(model, valid_loader)
            if f1 > best_f1:
                #torch.save(model, model_file_path)
                model_best_param_dict = copy.deepcopy(model.state_dict())
                best_f1 = f1
        model.load_state_dict(model_best_param_dict)

        ###################################################################################################
        # ind_pre_epochs
        # train_dataloader = self.get_train_dataloader()
        # valid_loader = self.get_eval_dataloader()
        # best_f1 = 0

        # num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        # max_steps = math.ceil(self.args.ind_pre_epoches * num_update_steps_per_epoch)
        # ind_con_num_train_epochs = self.args.ind_pre_epoches
        #
        # ## param need to be init again??
        # self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        #
        # tr_loss = torch.tensor(0.0).to(self.args.device)
        # self._total_loss_scalar = 0.0
        # self._globalstep_last_logged = 0
        #
        # #model.zero_grad()
        # #for step, inputs in enumerate(train_dataloader):
        # #    inputs.pop("sent_id")
        # #    inputs.pop("original_text")
        #
        # global_step = 0
        # model_best_param_dict = None
        # for epoch in range(ind_con_num_train_epochs):
        #     epoch_iterator = tqdm(train_dataloader, initial=global_step, desc="Iter (in_pre_loss)")
        #     model.train()
        #     for step, inputs in enumerate(epoch_iterator):
        #         for k, v in inputs.items():
        #             if isinstance(v, torch.Tensor):
        #                 inputs[k] = v.to(self.args.device)
        #
        #         loss = model(inputs, stage='in_pre', mode='finetune')
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        #
        #         self.optimizer.step()
        #         #self.lr_scheduler.step()
        #         #model.zero_grad()
        #         global_step += 1
        #         tr_loss += loss.item()
        #
        #     print('Epoch: [{0}]: Loss {loss:.4f}'.format(epoch, loss=tr_loss / global_step))
        #     ind_pre_loss_name = 'ind_pre_loss'
        #     fitlog.add_loss((tr_loss / global_step), name=ind_pre_loss_name, step=epoch)
        #
        #     ### Get model performing best #####
        #     f1 = self.valution_cal(model, valid_loader)
        #     if f1 > best_f1:
        #         #torch.save(model, model_file_path)
        #         model_best_param_dict = copy.deepcopy(model.state_dict())
        #         best_f1 = f1
        # model.load_state_dict(model_best_param_dict)

