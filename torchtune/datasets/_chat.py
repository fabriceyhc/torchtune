# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import ast
import json
import textwrap
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._utils import _get_chat_format
from torchtune.data import (
    ChatFormat,
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
    openai_to_llama2_messages,
    sharegpt_to_llama2_messages,
    validate_messages,
)
from torchtune.modules.tokenizers import Tokenizer


class ChatDataset(Dataset):
    """
    Class that supports any custom dataset with multiturn conversations.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> foreach turn{format into template -> tokenize}

    If the column/key names differ from the expected names in the ``ChatFormat``,
    then the ``column_map`` argument can be used to provide this mapping.

    Use ``convert_to_messages`` to prepare your dataset into the Llama2 chat format
    and roles::

        [
            Message(
                role=<system|user|assistant>,
                content=<message>,
            ),
            ...
        ]

    This class supports multi-turn conversations. If a tokenizer sample with multiple
    turns does not fit within ``max_seq_len`` then it is truncated.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an ``encode`` and ``decode`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        convert_to_messages (Callable[[Mapping[str, Any]], List[Message]]): function that keys into the desired field in the sample
            and converts to a list of :class:`~torchtune.data.Message` that follows the Llama format with the expected keys
        chat_format (Optional[ChatFormat]): template used to format the chat. This is used to add structured text around the actual
            messages, such as the [INST] tags in Llama2 and in Mistral. The extra text will still get tokenized as normal text, not
            as special tokens. In models like Llama3 where the tokenizer adds tags as special tokens, ``chat_format`` is not needed,
            unless you want to structure messages in a particular way for inference. If the placeholder variable names in the
            template do not match the column/key names in the dataset, use ``column_map`` to map them. For a list of all possible
            chat formats, check out :ref:`chat_formats`. Default: None.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        source: str,
        convert_to_messages: Callable[[Mapping[str, Any]], List[Message]],
        chat_format: Optional[ChatFormat] = None,
        max_seq_len: int,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._convert_to_messages = convert_to_messages
        self.chat_format = chat_format
        self.max_seq_len = max_seq_len
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        messages = self._convert_to_messages(sample)
        if self.chat_format is not None:
            messages = self.chat_format.format(messages)
        validate_messages(messages)
        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )
        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels


def chat_dataset(
    *,
    tokenizer: Tokenizer,
    source: str,
    conversation_style: str,
    chat_format: Optional[str] = None,
    max_seq_len: int,
    train_on_input: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> ChatDataset:
    """
    Build a configurable dataset with conversations. This method should be
    used to configure a custom chat dataset from the yaml config instead of
    using :class:`~torchtune.datasets.ChatDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an ``encode`` and ``decode`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the Llama style. Supported styles are: "sharegpt"
        chat_format (Optional[str]): name of ``ChatFormat`` class used to format the messages. See the description in
            :class:`~torchtune.datasets.ChatDataset` for more details. For a list of all possible chat formats,
            check out :ref:`chat_formats`. Default: None.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.

    Examples:
        >>> from torchtune.datasets import chat_dataset
        >>> dataset = chat_dataset(
        ...   tokenizer=tokenizer,
        ...   source="HuggingFaceH4/no_robots",
        ...   conversation_style="sharegpt",
        ...   chat_format=ChatMLFormat,
        ...   max_seq_len=2096,
        ...   train_on_input=True
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.chat_dataset
            source: HuggingFaceH4/no_robots
            conversation_style: sharegpt
            chat_format: ChatMLFormat
            max_seq_len: 2096
            train_on_input: True

    Returns:
        ChatDataset: the configured :class:`~torchtune.datasets.ChatDataset`

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        convert_to_messages = sharegpt_to_llama2_messages
    elif conversation_style == "openai":
        convert_to_messages = openai_to_llama2_messages
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=convert_to_messages,
        chat_format=_get_chat_format(chat_format) if chat_format is not None else None,
        max_seq_len=max_seq_len,
        train_on_input=train_on_input,
        **load_dataset_kwargs,
    )

TEMPLATE = textwrap.dedent("""
    ## Instructions 

    Review the medical note below. Explain whether or not the medical notes describe the use of a particular drug:
        - Heroin
        - Cocaine
        - Methamphetamine
        - Benzodiazepine
        - Prescription Opioids
        - Marijuana
        - Fentanyl
    
    Notes:
    - Any mention of drug use within the entire note, you will mark that sentence as drugUseAnswer = True.
    - If drugUseAnswer = False, then do not mark any of the other substances.
    - IDU, IVDA, IVDU are all indicative of injectionDrugUseAnswer = True.
    - If a patient denies a particular drug, do not mark that drug as being present in the note. 
        - Ex: 'patient denied using heroin' then drugUseAnswer = False and heroinUseAnswer = False. 
    - If the notes mentions amphetamines within the context of illicit use, that can be marked as methamphetamineUseAnswer = True. 
        - Ex: If these are amphetamine salts for treatment of ADHD then methamphetamineUseAnswer = False.
    - For oxycontin or other opiates, do not highlight any that are prescribed and taken as medications. Only annotate if it is illicit use.
        - Ex: If patient is illicitly using suboxone, then that would be marked as prescriptionOpioidsUseAnswer = True.

    Think step by step and provide explainations for your answers. Then summarize your analysis in the following JSON format: 
    
    ```json
    {
        "heroinUseAnswer": <boolean>,
        "cocaineUseAnswer": <boolean>,
        "methamphetamineUseAnswer": <boolean>,
        "benzodiazepineUseAnswer": <boolean>,
        "prescriptionOpioidsUseAnswer": <boolean>,
        "marijuanaUseAnswer": <boolean>,
        "fentanylUseAnswer": <boolean>,
        "injectionDrugUseAnswer": <boolean>,
        "drugUseAnswer": <boolean>
    }
    ```
    
    ## Medical Note
    
    {{medical_note}}

    ## Explanation and Summary
""")

def format_answer(label):
    answers = {
        "heroinUse": bool(label[0]),
        "cocaineUse": bool(label[1]),
        "methamphetamineUse": bool(label[2]),
        "benzodiazepineUse": bool(label[3]),
        "prescriptionOpioidsUse": bool(label[4]),
        "marijuanaUse": bool(label[5]),
        "fentanylUse": bool(label[6]),
        "injectionDrugUse": bool(label[7]),
        "drugUse": bool(label[8])
    }

    return json.dumps(answers, indent=4)

def message_converter(sample: Mapping[str, Any], train_on_input=None) -> List[Message]:
    input_msg = TEMPLATE.replace("{{medical_note}}", sample["text"])
    output_msg = format_answer(np.array(ast.literal_eval(sample["label"])))

    user_message = Message(
        role="user",
        content=input_msg,
        masked=False,  # True if not training on prompt
    )
    assistant_message = Message(
        role="assistant",
        content=output_msg,
        masked=False,
    )
    # A single turn conversation
    messages = [user_message, assistant_message]

    return messages

def custom_dataset(
    *,
    tokenizer: Tokenizer,
    max_seq_len: int = 2048,  # You can expose this if you want to experiment
) -> ChatDataset:

    return ChatDataset(
        tokenizer=tokenizer,
        # For local csv files, we specify "csv" as the source, just like in
        # load_dataset
        source="csv",
        convert_to_messages=message_converter,
        # Llama3 does not need a chat format
        chat_format=None,
        max_seq_len=max_seq_len,
        # To load a local file we specify it as data_files just like in
        # load_dataset
        data_files="/data2/fabricehc/overdose/data/processed/under_over_sample/train.csv",
        split="train"
    )