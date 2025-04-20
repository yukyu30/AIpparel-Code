from enum import Enum

import numpy as np
import torch
import torch.distributed as dist


def discretize(params: np.ndarray, bin_size: int, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    params: np.ndarray = (params  - shift) / scale
    params = np.clip(params, 0, 1) * bin_size
    params = params.astype(int).clip(0, bin_size - 1)
    return params

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_PLACEHOLDER_TOKEN = "<placeholder>"


HMR_SHORT_QUESTION_LIST = [
    "I have a description of a person's pose, can you give the SMPL pose of this person?",
    "Give you a word descrption of a human, please output the SMPL pose.",
    "Describe what this perosn is doing using SMPL pose.",
    "What's the SMPL pose of this person?",
    "Use SMPL pose to describe this person's behavior."
]
 
DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST = [
    "I have a word description of a piece of garment, can you provide the sewing pattern? {sent}",
    "There is a garment like this: {sent} Please output this sewing pattern.",
    "{sent} Give the sewing pattern.",
    "What's the sewing pattern? {sent}",
    "Describe the garment as a sewing pattern. {sent}",
    "Sewing pattern is described as words: {sent} The sewing pattern is?",
    "I have a garment in this style: {sent} Can you provide the sewing pattern?",
    "Can you provide a sewing pattern for this garment? {sent}",
    "What is the sewing pattern of this garment? {sent}",
    "Sewing pattern can be described as words: {sent} And it can also be described as a sewing pattern as tokens. Can you output this?",
]

SPECULATIVE_TEXT_SHORT_QUESTION_LIST = [
    "Please output a garment that can be worn in this situation as a sewing pattern. {sent}",
    "Can you provide a garment that can be worn in this situation as a sewing pattern? {sent}",
    "What is an example of a piece of garment that can be worn in this situation? {sent} Output as a sewing pattern.",
    "I want to dress up for the following occasion. {sent} Can you provide a sewing pattern that can be worn?",
    "What would be a garment that can be worn in this situation? Output in the form of a sewing pattern. {sent}",
    "I want a piece of garment most suitable for this occasion. {sent} Can you provide a sewing pattern that can be worn?"
]

SHORT_QUESTION_WITH_TEXT_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "The person in the image is wearing a garment with details given in words: {sent} Can you provide the sewing pattern of the garment?",
    DEFAULT_IMAGE_TOKEN + "\n" + "The garment in the image is described as: {sent} Output the sewing pattern of the garment.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "The person is wearing in the this style: {sent}. Please respond with sewing pattern.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "The garment in the image is in this style: {sent}. Please output sewing pattern.",
    DEFAULT_IMAGE_TOKEN + "\n" + "There is a person in the middle of the image with details given in words: {sent}, Use a sewing pattern to describe the person's garment.",
]

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you predict the sewing pattern of the person in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "There is a person in the middle of the image, please output this person's sewing pattern.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the garment in this image? Please respond with sewing pattern.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the person wearing in this image? Please output sewing pattern.",
    DEFAULT_IMAGE_TOKEN + "\n" + "There is a person in the middle of the image, use sewing pattern to describe the person's garment.",
]

EDITING_QUESTION_LIST = [
    "I have a sewing pattern: {pattern}. Edit the sewing pattern according to the instruction. {sent}",
    "I want to edit the sewing pattern {pattern} based on the following instruction: {sent} Output in the form of a sewing pattern.",
    "Modify the sewing pattern {pattern} based on the following guidance: {sent} Respond with a sewing pattern.",
    "I have a sewing pattern: {pattern}. Please provide a new sewing pattern based on the instruction: {sent}",
    "How would the sewing pattern {pattern} look like if I edit it based on the following instruction? {sent} Respond with a new sewing pattern.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output sewing pattern and explain its style.",
    "Please output sewing pattern and explain when to wear it.",
    "Please output sewing pattern and give some explanation.",
]

ANSWER_LIST = [
    "It is {pattern}.",
    "It is a sewing pattern for {pattern}.",
    "Sure, {pattern}.",
    "Sure, it is {pattern}.",
    "Sure, the sewing pattern is {pattern}.",
    "{pattern}.",
    "The sewing pattern is {pattern}.",
    "The sewing pattern of the person is {pattern}.",
    "The sewing pattern of this person's garment is {pattern}.",
]
