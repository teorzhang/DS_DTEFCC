import random
from re import X
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

# from transformers import DataCollatorForSeq2Seq
InputDataClass = NewType("InputDataClass", Any)


@dataclass
class MyDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[
        bool, str, PaddingStrategy] = True  # 可以为布尔类型、字符串类型或者一个PaddingStrategy对象。当值为布尔类型时，True表示填充至最大序列长度，False表示不填充。当为字符串类型时，"longest"表示填充值最大序列长度，"max_length"表示填充值参数max_length设置的长度，"do_not_pad"表示不填充。
    max_length: Optional[int] = None  # 表示填充序列的最大长度，当设置padding="max_length"时，该参数才会有用
    pad_to_multiple_of: Optional[int] = None  # 表示填充的序列的倍数
    label_pad_token_id: int = -100  # 表示填充标签时的值，默认为-100
    dis_offset: int = 463
    adj_mat_ls: list = None

    def __call__(self, features):

        # features List[dict]     labels == summary
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        neg_summary = [feature["neg_summary"] for feature in features] if "neg_summary" in features[0].keys() else None
        input_ids = [feature["input_ids"] for feature in features] if "input_ids" in features[0].keys() else None
        inputs_pos = [feature["pos_input_ids"] for feature in features] if "pos_input_ids" in features[0].keys() else None
        inputs_neg = [feature["neg_input_ids"] for feature in features] if "neg_input_ids" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                # 填充标签
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        # if neg_summary is not None:
        #     max_label_length = max(len(l) for l in labels)
        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         # 填充标签
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["neg_summary"]))
        #         feature["neg_summary"] = (
        #             feature["neg_summary"] + remainder if padding_side == "right" else remainder + feature["neg_summary"]
        #         )
        if neg_summary is not None:
            max_label_length = max(len(l) for l in labels)
            for feature in features:
                # 填充标签
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["neg_summary"]))
                truncated_neg_summary = (
                    feature["neg_summary"][:max_label_length] if len(feature["neg_summary"]) > max_label_length else
                    feature["neg_summary"]
                )
                feature["neg_summary"] = (
                    truncated_neg_summary + remainder if padding_side == "right" else remainder + truncated_neg_summary
                )

        # if self.max_utt_threshold > 0:
        #     for feature in feature:
        #         for n in ['gt_input_ids','gt_attention_mask','']

        if 'gt_input_ids' in features[0].keys():

            normal_features_ls = [
                {'attention_mask': x['attention_mask'], 'input_ids': x['input_ids'], 'labels': x['labels']} for x in features]
            normal_padded_featurs = self.tokenizer.pad(
                normal_features_ls,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            padded_features = {'gt_input_ids': [], "gt_attention_mask": [], "adj_mats": []}

            padded_features['num_utt_ls'], padded_features['gt_input_ids'], padded_features['gt_attention_mask'] = \
                pad_list_of_tensor(features, self.tokenizer)
            max_num_utt = max([len(feature['gt_input_ids']) for feature in features])
            # 设padded_features['adj_mats'][adj_type] = []
            adj_mats_ls = self.adj_mat_ls
            if any([m in features[0].keys() for m in adj_mats_ls]): padded_features['adj_mats'] = {}
            for adj_type in adj_mats_ls:
                if adj_type in features[0].keys():
                    padded_features['adj_mats'][adj_type] = []

            for feature in features:
                #  sudo_features = []
                #  for idx in range(len(feature['gt_attention_mask'])):
                #      sudo_dict = {'attention_mask':feature['gt_attention_mask'][idx],'input_ids':feature['gt_input_ids'][idx]}
                #      sudo_features.append(sudo_dict)

                #  padded_sudo_features = self.tokenizer.pad(
                #      sudo_features, #List[Dict[str,List[int]]]
                #      padding=self.padding,
                #      max_length=self.max_length,
                #      pad_to_multiple_of=self.pad_to_multiple_of,
                #      return_tensors="pt",
                #  )
                #  # dict[str:tensor(batch_first)]
                #  padded_features['gt_input_ids'].append(padded_sudo_features['input_ids'])
                #  padded_features['gt_attention_mask'].append(padded_sudo_features['attention_mask'])
                for adj_type in adj_mats_ls:
                    if adj_type in feature.keys():
                        mat = feature[adj_type]
                        ori_mat_size = len(mat)
                        mat = torch.tensor(mat)
                        if not ori_mat_size <= max_num_utt:
                            raise ValueError(f"{adj_type} has size {ori_mat_size}, expected at most {max_num_utt}")
                        padded_mat = torch.zeros((max_num_utt, max_num_utt), dtype=mat.dtype)
                        padded_mat[:ori_mat_size, :ori_mat_size] = mat
                        if adj_type == 'distance_adj':
                            padded_mat += self.dis_offset
                    padded_features['adj_mats'][adj_type].append(padded_mat)

            if 'adj_mats' in padded_features.keys():
                for k, v in padded_features['adj_mats'].items():
                    padded_features['adj_mats'][k] = torch.stack(v)
            # 3个返回
            padded_features['input_ids'] = normal_padded_featurs['input_ids']
            padded_features['attention_mask'] = normal_padded_featurs['attention_mask']
            padded_features['labels'] = normal_padded_featurs['labels']
        else:


            normal_features_ls = [
                {'attention_mask': x['attention_mask'], 'input_ids': x['input_ids'], 'labels': x['labels']} for x in features]
            normal_padded_featurs = self.tokenizer.pad(
                normal_features_ls,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            padded_features = {'pos_input_ids': [], "pos_attention_mask": [],'neg_input_ids': [], "neg_attention_mask": []}
            padded_features['num_utt_ls_pos'], padded_features['pos_input_ids'], padded_features['pos_attention_mask'] = \
                pad_list_of_tensor(features, self.tokenizer)
            padded_features['num_utt_ls_neg'], padded_features['neg_input_ids'], padded_features['neg_attention_mask'] = \
                pad_list_of_tensor2(features, self.tokenizer)

            padded_features['input_ids'] = normal_padded_featurs['input_ids']
            padded_features['attention_mask'] = normal_padded_featurs['attention_mask']
            padded_features['labels'] = normal_padded_featurs['labels']
            padded_features['neg_summary'] = torch.stack([torch.tensor(x['neg_summary']) for x in features])

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=padded_features["labels"])
            padded_features["decoder_input_ids"] = decoder_input_ids
            neg_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=padded_features["neg_summary"])
            padded_features["neg_decoder_input_ids"] = neg_decoder_input_ids

        return padded_features


def pad_list_of_tensor(features, tokenizer):
    """
    features: List[Dict[str,List[str]]
    """
    max_seq_len_in_a_batch = 0
    len_ls = []
    for sample in features:
        len_ls.append(len(sample['pos_input_ids']))
        # for utt in sample['pos_input_ids']:
        l = len(sample['pos_input_ids'])
        if l > max_seq_len_in_a_batch:
            max_seq_len_in_a_batch = l  # 找到最大长度

    for sample in features:  # 填充pad
        for k, v in sample.items():
            if k == 'pos_input_ids':
                # for utt in v:
                diff = max_seq_len_in_a_batch - len(v)
                v += [tokenizer.pad_token_id] * diff  # utt = utt + is wrong
            elif k == 'pos_attention_mask':
                # for mask in v:
                diff = max_seq_len_in_a_batch - len(v)
                v += [0] * diff
    return len_ls, torch.stack([torch.tensor(x['pos_input_ids']) for x in features]), torch.stack([torch.tensor(x['pos_attention_mask']) for x in features])

def pad_list_of_tensor2(features, tokenizer):
    """
    features: List[Dict[str,List[str]]
    """
    max_seq_len_in_a_batch = 0
    len_ls = []
    for sample in features:
        len_ls.append(len(sample['neg_input_ids']))
        # for utt in sample['pos_input_ids']:
        l = len(sample['neg_input_ids'])
        if l > max_seq_len_in_a_batch:
            max_seq_len_in_a_batch = l  # 找到最大长度

    for sample in features:  # 填充pad
        for k, v in sample.items():
            if k == 'neg_input_ids':
                # for utt in v:
                diff = max_seq_len_in_a_batch - len(v)
                v += [tokenizer.pad_token_id] * diff  # utt = utt + is wrong
            elif k == 'neg_attention_mask':
                # for mask in v:
                diff = max_seq_len_in_a_batch - len(v)
                v += [0] * diff
    return len_ls, torch.stack([torch.tensor(x['neg_input_ids']) for x in features]), torch.stack([torch.tensor(x['neg_attention_mask']) for x in features])
