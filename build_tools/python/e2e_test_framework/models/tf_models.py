## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines Tensorflow models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions
import e2e_test_framework.models.utils as model_utils

TF_MODELS_MANUAL_ROOT_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual"

MINILM_L12_H384_UNCASED_INT32_SEQLEN128 = common_definitions.Model(
    id=unique_ids.MODEL_MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    name="MiniLML12H384Uncased",
    tags=["int32", "seqlen128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/commit/44acabbec0ef496f6dbc93adadea57f376b7c0ec
    source_url=
    f"{TF_MODELS_MANUAL_ROOT_DIR}/MiniLML12H384Uncased_2023-05-07.timestamp_1683504734.mlirbc",
    entry_function="predict",
    input_types=["1x128xi32", "1x128xi32", "1x128xi32"])

BERT_FOR_MASKED_LM_FP32_SEQLEN512 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_FOR_MASKED_LM_FP32_SEQLEN512_TF,
    name="BertForMaskedLMTF",
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Converted from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#tfbertformaskedlm
    source_url=
    f"{TF_MODELS_MANUAL_ROOT_DIR}/BertForMaskedLMTF_2023-05-07.timestamp_1683504734.mlirbc",
    entry_function="forward",
    input_types=["1x512xi32", "1x512xi32"])

EFFICIENTNET_V2_S_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_V2_S_FP32_TF,
    name="EfficientNetV2STF",
    tags=["fp32", "cnn", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Converted from https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/efficientnet_v2.py
    source_url=
    f"{TF_MODELS_MANUAL_ROOT_DIR}/EfficientNetV2STF_2023-05-07.timestamp_1683504734.mlirbc",
    entry_function="forward",
    input_types=["1x384x384x3xf32"])

# This is the model used in the MLPerf Inference Suite.
BERT_LARGE_TF_FP32_SEQLEN384 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_TF_FP32_SEQLEN384,
    name="BertLargeTF",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Derived from https://github.com/mlcommons/inference/tree/master/language/bert
    # Instructions on how to regenerate the model: https://gist.github.com/mariecwhite/e61ccebd979d98d097946ac7725bcc29
    source_url=
    f"{TF_MODELS_MANUAL_ROOT_DIR}/BertLargeTF_2023-05-07.timestamp_1683504734.mlirbc",
    entry_function="serving_default",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"])

TF_MODELS_ROOT_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1683544084"

ID_FORMAT = r"{}-batch-${{BATCH_SIZE}}"
NAME_FORMAT = r"{}Batch${{BATCH_SIZE}}"
SOURCE_URL_FORMAT = f"{TF_MODELS_ROOT_DIR}" r"/{}/batch_${{BATCH_SIZE}}/hlo.mlirbc"

# Derived from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
BERT_LARGE_384_FP32_TF_BATCHES = model_utils.generate_batch_models(
    id_template=ID_FORMAT.format(unique_ids.MODEL_BERT_LARGE_384_FP32_TF),
    name_template=NAME_FORMAT.format("BertLargeTF"),
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=SOURCE_URL_FORMAT.format("BERT_LARGE"),
    entry_function="forward",
    input_type_templates=[r"${BATCH_SIZE}x384xi32", r"${BATCH_SIZE}x384xi32"],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# Converted from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
RESNET50_3X224X224_FP32_TF_BATCHES = model_utils.generate_batch_models(
    id_template=ID_FORMAT.format(unique_ids.MODEL_RESNET50_3X224X224_FP32_TF),
    name_template=NAME_FORMAT.format("Resnet50TF"),
    tags=["fp32", "cnn"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=SOURCE_URL_FORMAT.format("RESNET50"),
    entry_function="forward",
    input_type_templates=[r"${BATCH_SIZE}x224x224x3xf32"],
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# Derived from https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#tft5model.
T5_LARGE_512_FP32_TF_BATCHES = model_utils.generate_batch_models(
    id_template=ID_FORMAT.format(unique_ids.MODEL_T5_LARGE_512_FP32_TF),
    name_template=NAME_FORMAT.format("T5LargeTF"),
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=SOURCE_URL_FORMAT.format("T5_LARGE"),
    entry_function="forward",
    input_type_templates=[r"${BATCH_SIZE}x512xi32", r"${BATCH_SIZE}x512xi32"],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
