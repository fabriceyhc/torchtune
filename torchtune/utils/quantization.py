# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

import torch
from torchao.quantization.quant_api import (
    # apply_weight_only_int8_quant,
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
    Int8DynActInt4WeightQuantizer,
    Quantizer,
)

from torchtune.modules.low_precision._utils import (
    _get_torchao_version,
    _is_fbcode,
    _nightly_version_ge,
)

ao_version, is_nightly = _get_torchao_version()
if _is_fbcode() or (is_nightly and _nightly_version_ge(ao_version, "2024-07-03")):
    from torchao.quantization.quant_api import quantize_ as quantize
else:
    from torchao.quantization.quant_api import quantize

# importing TORCH_VERSION_AFTER_2_3 because `Int8DynActInt4WeightQuantizer`
# is only available after 2.3 so we have to guard the pytorch versions to decide
# the list of supported quantizers
from torchao.utils import TORCH_VERSION_AFTER_2_3, TORCH_VERSION_AFTER_2_4

__all__ = [
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyGPTQQuantizer",
    "Int8WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
    "get_quantizer_mode",
]


class Int8WeightOnlyQuantizer(Quantizer):
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        return quantize(model, "int8_weight_only")


_quantizer_to_mode = {
    Int4WeightOnlyQuantizer: "4w",
    Int8WeightOnlyQuantizer: "8w",
    Int4WeightOnlyGPTQQuantizer: "4w-gptq",
    Int8DynActInt4WeightQuantizer: "8da4w",
}


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization.

    For example, in the case of int4 weight only quantization, we'll return "4w".
    If the quantizer is not recognized as a known quantizer, we'll return None.

    Currently supported:

    - :class:`~torchao.quantization.quant_api.Int4WeightOnlyQuantizer`: "4w"
    - :class:`~torchao.quantization.quant_api.Int8WeightOnlyQuantizer`: "8w"
    - :class:`~torchao.quantization.quant_api.Int4WeightOnlyGPTQQuantizer`: "4w-gptq"
    - :class:`~torchao.quantization.quant_api.Int8DynActInt4WeightQuantizer`: "8da4w" (requires ``torch>=2.3.0``)
    - :class:`~torchao.quantization.prototype.qat.Int8DynActInt4WeightQATQuantizer`: "8da4w-qat" (requires ``torch>=2.4.0``)

    Args:
        quantizer (Optional[Callable]): A callable object that implements the `quantize` method.

    Returns:
        Optional[str]: The quantization mode.
    """
    return _quantizer_to_mode.get(type(quantizer), None)
