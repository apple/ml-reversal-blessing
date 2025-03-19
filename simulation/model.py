#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from torchtune.models.llama3._component_builders import llama3


def llama3_2b():
    """
    Builder for creating a Llama3 model initialized w/ 2b parameter values.
    Returns:
        TransformerDecoder: Instantiation of Llama3 2B model
    """
    return llama3(
        vocab_size=128_256,
        num_layers=24,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=7168,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )
