# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

#from transformers.integrations import is_deepspeed_zero3_enabled
# from transformers.trainer import Trainer
from transformers.trainer import *
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


logger = logging.get_logger(__name__)


class Seq2SeqTrainer(Trainer):
    def cos_similarity(self, x1, x2):
        # 计算余弦相似度
        return F.cosine_similarity(x1, x2, dim=-1)

    def triplet_loss(self, anchor, positive, negative, margin):
        # 计算三元组损失
        positive_distance = 1 - self.cos_similarity(anchor, positive)
        negative_distance = 1 - self.cos_similarity(anchor, negative)
        loss = torch.clamp(margin + positive_distance - negative_distance, min=0)
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs_initial = self._prepare_inputs(inputs)
        inputs = {key: inputs_initial[key] for key in ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']}
        inputs_pos = {'input_ids': inputs_initial['pos_input_ids'], 'attention_mask': inputs_initial['pos_attention_mask'],
                      'labels': inputs_initial['labels'], 'decoder_input_ids': inputs_initial['decoder_input_ids']}
        inputs_neg = {'input_ids': inputs_initial['neg_input_ids'], 'attention_mask': inputs_initial['neg_attention_mask'],
                      'labels': inputs_initial['neg_summary'], 'decoder_input_ids': inputs_initial['neg_decoder_input_ids']}
        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() 用于在多GPU并行训练中平均

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed 在 `backward` 中通过 gradient_accumulation_steps 处理损失缩放
            loss = loss / self.args.gradient_accumulation_steps

        # 对比学习损失计算
        outputs_anchor = model(**inputs)  # H
        outputs_positive = model(**inputs_pos)  # H+
        outputs_negative = model(**inputs_neg)  # H-

        triplet_losses = [self.triplet_loss(anchor, positive, negative, 0.2)#margin = 0.4
                          for anchor, positive, negative in
                          zip(outputs_anchor.logits, outputs_positive.logits, outputs_negative.logits)]
        total_triplet_loss = torch.mean(torch.stack(triplet_losses))

        # 结合原始损失和对比学习损失
        total_loss = loss + 0.7 * total_triplet_loss

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        elif self.use_apex:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # deepspeed 在 gradient_accumulation_steps 中进行损失缩放
            total_loss = self.deepspeed.backward(total_loss)
        else:
            total_loss.backward()

        return total_loss.detach()

    #源码
    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (:obj:`nn.Module`):
    #             The model to train.
    #         inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument :obj:`labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         :obj:`torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs) #不改变
    #
    #     if is_sagemaker_mp_enabled():
    #         scaler = self.scaler if self.use_amp else None
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     if self.use_amp:
    #         with autocast():
    #             loss = self.compute_loss(model, inputs)#进入这 6.3395
    #     else:
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #         # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
    #         loss = loss / self.args.gradient_accumulation_steps #0.7924
    #
    #     if self.use_amp:
    #         self.scaler.scale(loss).backward()#进入这
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         # loss gets scaled under gradient_accumulation_steps in deepspeed
    #         loss = self.deepspeed.backward(loss)
    #     else:
    #         loss.backward()
    #
    #     return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels") #进入这 （2，50）
        else:
            labels = None
        outputs = model(**inputs)#outpus= Seq2SeqLMOutput
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)#进入这
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "min_length": self.model.config.min_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            #"synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        adj_mats = inputs.get('adj_mats',None)
        gt_attention_mask = inputs.get('gt_attention_mask',None)
        gt_input_ids = inputs.get('gt_input_ids',None)
        num_utt_ls = inputs.get('num_utt_ls',None)
        if gt_attention_mask is not None:
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                adj_mats = adj_mats,
                gt_attention_mask = gt_attention_mask,
                gt_input_ids = gt_input_ids,
                num_utt_ls = num_utt_ls,
                # test2=null,
                **gen_kwargs,
                )
        else:
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # test1=null,
                **gen_kwargs,
                )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()#进入这
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:#FALSE
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
            elif isinstance(v,list) and isinstance(v[0],torch.Tensor):
                inputs[k] = [x.to(self.args.device) for x in v]
            elif isinstance(v,dict) and isinstance(list(v.values())[0],torch.Tensor):
                for _k,_v in v.items():
                    v[_k] = _v.to(self.args.device)


        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
