from typing import Optional
import transformers
import torch
from collections import defaultdict
from models.llava import conversation as conversation_lib
from ..datasets.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, DEFAULT_PLACEHOLDER_TOKEN, IMAGE_TOKEN_INDEX
from models.llava.mm_utils import tokenizer_image_and_pattern_token

def collate_fn(
    batch, 
    tokenizer: Optional[transformers.PreTrainedTokenizer]=None, 
    conv_type="llava_v1", 
    use_mm_start_end=True, 
    generation_only=False,
    local_rank=-1
):
    encoded_pattern_list = []
    pattern_param_list = []
    pattern_endpoints_list = []
    sample_type_list = []
    pattern_transf_list = []
    question_pattern_list = []
    question_pattern_param_list = []
    question_pattern_endpoints_list = []
    question_pattern_transf_list = []
    image_path_list = []
    images_clip_list = []
    conversation_list = []
    gt_pattern_list = []
    question_conv_list = []
    questions_list = []
    offset_list = [0]
    cnt = 0
    for (
        pattern_dict,
        question_pattern_dict,
        image_path, 
        images_clip, 
        conversations, 
        question_only_convs, 
        questions, 
        gt_pattern, 
        sample_type
        ) in batch:
        sample_type_list.append(sample_type)
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            pattern_param_list.append([{}])
        pattern_endpoint = pattern_dict.get("endpoints", None)
        if pattern_endpoint is not None:
            pattern_endpoint = torch.cat(pattern_endpoint)
        else:
            pattern_endpoint = torch.zeros(0, 2)
        pattern_endpoints_list.append(pattern_endpoint)
        pattern_transf = pattern_dict.get("transformations", None)
        if pattern_transf is not None:
            pattern_transf = torch.cat(pattern_transf)
        else:
            pattern_transf = torch.zeros(0, 7)
        pattern_transf_list.append(pattern_transf)
        question_pattern_list.append(question_pattern_dict.get("description", []))
        
        question_pattern_param = question_pattern_dict.get("params", None)
        if question_pattern_param is not None:
            question_pattern_param_list.append(question_pattern_param)
        else:
            question_pattern_param_list.append([{}])
        question_pattern_endpoint = question_pattern_dict.get("endpoints", None)
        if question_pattern_endpoint is not None:
            question_pattern_endpoint = torch.cat(question_pattern_endpoint)
        else:
            question_pattern_endpoint = torch.zeros(0, 2)
        question_pattern_endpoints_list.append(question_pattern_endpoint)
        question_pattern_transf = question_pattern_dict.get("transformations", None)
        if question_pattern_transf is not None:
            question_pattern_transf = torch.cat(question_pattern_transf)
        else:
            question_pattern_transf = torch.zeros(0, 7)
        question_pattern_transf_list.append(question_pattern_transf)
        # conversation_list.extend(conversations)
        image_path_list.append(image_path)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        question_conv_list.extend(question_only_convs)
        questions_list.append(questions)
        gt_pattern_list.append(gt_pattern)
        cnt += len(conversations)
        offset_list.append(cnt)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        for i in range(len(question_conv_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            question_conv_list[i] = question_conv_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    question_pattern_ids = [tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids if len(pattern_tokens) > 0 else [] for pattern_tokens in question_pattern_list]
    questions_ids = [
        tokenizer_image_and_pattern_token(question, tokenizer, question_pattern_id, pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN, return_tensors="pt")
        for question, question_pattern_id in zip(question_conv_list, question_pattern_ids)
    ]
    questions_ids = torch.nn.utils.rnn.pad_sequence(
        questions_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    question_pattern_endpoint_max_len = max([len(endpoints) for endpoints in question_pattern_endpoints_list])
    question_pattern_endpoints = torch.zeros(len(question_pattern_endpoints_list), question_pattern_endpoint_max_len, 2, dtype=torch.float32)
    question_pattern_endpoint_masks = torch.zeros(len(question_pattern_endpoints_list), question_pattern_endpoint_max_len, dtype=torch.bool)
    for i, endpoints in enumerate(question_pattern_endpoints_list):
        if len(endpoints) == 0:
            continue
        question_pattern_endpoints[i, :len(endpoints)] = endpoints
        question_pattern_endpoint_masks[i, :len(endpoints)] = True
    question_pattern_transf_max_len = max([len(transf) for transf in question_pattern_transf_list])
    question_pattern_transfs = torch.zeros(len(question_pattern_transf_list), question_pattern_transf_max_len, 7, dtype=torch.float32)
    question_pattern_transf_masks = torch.zeros(len(question_pattern_transf_list), question_pattern_transf_max_len, dtype=torch.bool)
    for i, transf in enumerate(question_pattern_transf_list):
        if len(transf) == 0:
            continue
        question_pattern_transfs[i, :len(transf)] = transf
        question_pattern_transf_masks[i, :len(transf)] = True
    
    question_attention_masks = questions_ids.ne(tokenizer.pad_token_id)
    if not generation_only:
        pattern_ids = [tokenizer(pattern_tokens, is_split_into_words=True, add_special_tokens=False).input_ids if len(pattern_tokens) > 0 else [] for pattern_tokens in encoded_pattern_list]
        # from IPython.core.debugger import set_trace; set_trace()
        input_ids = [
            tokenizer_image_and_pattern_token(prompt, tokenizer, pattern_id, pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN, return_tensors="pt")
            for prompt, pattern_id in zip(conversation_list, pattern_ids)
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        endpoint_max_len = max([len(endpoints) for endpoints in pattern_endpoints_list])
        pattern_endpoints = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, 2, dtype=torch.float32)
        pattern_endpoint_masks = torch.zeros(len(pattern_endpoints_list), endpoint_max_len, dtype=torch.bool)
        for i, endpoints in enumerate(pattern_endpoints_list):
            if len(endpoints) == 0:
                continue
            pattern_endpoints[i, :len(endpoints)] = endpoints
            pattern_endpoint_masks[i, :len(endpoints)] = True
        pattern_transf_max_len = max([len(transf) for transf in pattern_transf_list])
        pattern_transfs = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, pattern_transf_list[0].shape[-1], dtype=torch.float32)
        pattern_transf_masks = torch.zeros(len(pattern_transf_list), pattern_transf_max_len, dtype=torch.bool)
        for i, transf in enumerate(pattern_transf_list):
            if len(transf) == 0:
                continue
            pattern_transfs[i, :len(transf)] = transf
            pattern_transf_masks[i, :len(transf)] = True
        attention_masks = input_ids.ne(tokenizer.pad_token_id)


    if not generation_only:
        conv = conversation_lib.default_conversation.copy()
        targets = input_ids.clone()
        param_targets = []
        if conv_type == "llava_v1":
            sep = conv.sep + conv.roles[1] + ": "
        else:
            sep = "[/INST] "
        for conversation, target, pattern_id, pattern_params, qusetion_pattern_params in zip(conversation_list, targets, pattern_ids, pattern_param_list, question_pattern_param_list):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            param_target = defaultdict(list)
            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            n_patterns = 0
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep
                _n_question_patterns = parts[0].count(DEFAULT_PLACEHOLDER_TOKEN)
                instruction_len = len(tokenizer_image_and_pattern_token(parts[0], tokenizer, pattern_id[n_patterns:n_patterns+_n_question_patterns], pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN)) - 2
                _n_patterns = rou.count(DEFAULT_PLACEHOLDER_TOKEN)
                round_len = len(tokenizer_image_and_pattern_token(rou, tokenizer, pattern_id[n_patterns:n_patterns+_n_patterns], pattern_place_holder_token=DEFAULT_PLACEHOLDER_TOKEN))
                for param_dict in pattern_params[n_patterns+_n_question_patterns:n_patterns+_n_patterns]:
                    for k, v in param_dict.items():
                        param_target[k].append(v)
                n_patterns += _n_patterns


                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX
            param_target = {k: torch.concatenate(v) for k, v in param_target.items()}
            param_targets.append(param_target)
            
        param_targets_keys = set([k for param_target in param_targets for k in param_target.keys()])
        new_param_targets = dict()
        param_target_masks = dict()
        for key in param_targets_keys:
            max_len = max([len(param_target[key]) for param_target in param_targets if key in param_target])
            shape = (len(param_targets), max_len, [param_target[key].shape[-1] for param_target in param_targets if key in param_target][0])
            new_param_targets[key] = torch.zeros(*shape, dtype=torch.float32)
            param_target_masks[key] = torch.zeros(len(param_targets), max_len, dtype=torch.bool)
            for i, param_target in enumerate(param_targets):
                if key in param_target:
                    param_target_masks[key][i, :len(param_target[key])] = True
                    new_param_targets[key][i, :len(param_target[key])] = param_target[key]
        
    question_params = []
    for qusetion_pattern_params in question_pattern_param_list:
        question_param_dict = defaultdict(list)
        for param_dict in qusetion_pattern_params:
            for k, v in param_dict.items():
                question_param_dict[k].append(v)
        question_param = {k: torch.cat(v) for k, v in question_param_dict.items()}
        question_params.append(question_param)
    question_param_targets_keys = set([k for question_param_target in question_params for k in question_param_target.keys()])
    new_question_param_dict = dict()
    question_param_masks = dict()
    for key in question_param_targets_keys:
        max_len = max([len(param_target[key]) for param_target in question_params if key in param_target])
        shape = (len(question_params), max_len, [param_target[key].shape[-1] for param_target in question_params if key in param_target][0])
        new_question_param_dict[key] = torch.zeros(*shape, dtype=torch.float32)
        question_param_masks[key] = torch.zeros(len(param_targets), max_len, dtype=torch.bool)
        for i, param_target in enumerate(question_params):
            if key in param_target:
                question_param_masks[key][i, :len(param_target[key])] = True
                new_question_param_dict[key][i, :len(param_target[key])] = param_target[key]


    input_len = input_ids.shape[1] if not generation_only else questions_ids.shape[1]

    out_dict = {
        "input_len": input_len,
        "image_paths": image_path_list,
        "sample_type": torch.LongTensor(sample_type_list),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "questions_pattern_endpoints": question_pattern_endpoints,
        "questions_pattern_endpoints_mask": question_pattern_endpoint_masks,
        "questions_pattern_transformations": question_pattern_transfs,
        "questions_pattern_transformations_mask": question_pattern_transf_masks,
        "question_param_dict": new_question_param_dict,
        "question_param_masks": question_param_masks,
        "question_ids": questions_ids,
        "question_attention_masks": question_attention_masks,
        "conversation_list": conversation_list,
    }
    if not generation_only:
        out_dict.update({
            "param_targets": new_param_targets,
            "param_target_endpoints": pattern_endpoints,
            "param_target_endpoints_mask": pattern_endpoint_masks,
            "param_target_transformations": pattern_transfs,
            "param_target_transformations_mask": pattern_transf_masks,
            "param_target_masks": param_target_masks,
            "gt_patterns": gt_pattern_list,
            "attention_masks": attention_masks,
            "input_ids": input_ids,
            "labels": targets,
        })
    return out_dict