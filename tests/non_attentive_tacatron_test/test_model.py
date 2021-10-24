from unittest.mock import patch

import torch

from src.models.feature_models.non_attentive_tacatron.config import (
    DecoderConfig, DurationConfig, EncoderConfig, GaussianUpsampleConfig,
    ModelConfig, PostNetConfig, RangeConfig,
)
from src.models.feature_models.non_attentive_tacatron.model import (
    Attention, Decoder, DurationPredictor, Encoder, NonAttentiveTacatron,
    Postnet, Prenet, RangePredictor,
)

DECODER_CONFIG = DecoderConfig()
DURATION_CONFIG = DurationConfig()
RANGE_CONFIG = RangeConfig()
ENCODER_CONFIG = EncoderConfig()
ATTENTION_CONFIG = GaussianUpsampleConfig(
    duration_config=DURATION_CONFIG, range_config=RANGE_CONFIG
)
POSTNET_CONFIG = PostNetConfig()
MODEL_CONFIG = ModelConfig(
    encoder_config=ENCODER_CONFIG,
    attention_config=ATTENTION_CONFIG,
    decoder_config=DECODER_CONFIG,
    postnet_config=POSTNET_CONFIG,
)
N_PHONEMES = 100
N_SPEAKER = 4
EMBEDDING_DIM = MODEL_CONFIG.phonem_embedding_dim + MODEL_CONFIG.speaker_embedding_dim
INPUT_PHONEMES = torch.randint(N_PHONEMES, size=(16, 50), dtype=torch.long)
INPUT_SPEAKERS = torch.randint(N_SPEAKER, size=(16,), dtype=torch.long)
PHONEM_EMB = torch.randn(16, 50, MODEL_CONFIG.phonem_embedding_dim, dtype=torch.float)
SPEAKER_EMB = torch.randn(16, MODEL_CONFIG.speaker_embedding_dim, dtype=torch.float)
EMBEDDING = torch.cat(
    [PHONEM_EMB, torch.repeat_interleave(SPEAKER_EMB.unsqueeze(1), 50, dim=1)], dim=-1
)
INPUT_LENGTH = torch.arange(35, 51, dtype=torch.long)
INPUT_DURATIONS = torch.randint(5, 10, size=(16, 50), dtype=torch.long)
for i, l in enumerate(INPUT_LENGTH):
    INPUT_DURATIONS[i, l:] = 0
DURATIONS_MAX = INPUT_DURATIONS.cumsum(dim=1).max(dim=1).values
INPUT_MELS = torch.randn(
    16, DURATIONS_MAX.max(), MODEL_CONFIG.n_mel_channels, dtype=torch.float
)
for i, l in enumerate(DURATIONS_MAX):
    INPUT_MELS[i, l:, :] = 0
ATTENTION_OUT_DIM = EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim
DECODER_RNN_OUT = torch.randn(16, 1, DECODER_CONFIG.decoder_rnn_dim)
ATTENTION_OUT = torch.randn((16, DURATIONS_MAX.max(), ATTENTION_OUT_DIM))
for i, l in enumerate(DURATIONS_MAX):
    ATTENTION_OUT[i, l:, :] = 0
MODEL_INPUT = (
    INPUT_PHONEMES,
    INPUT_LENGTH,
    INPUT_SPEAKERS,
    INPUT_DURATIONS,
    INPUT_MELS,
)
MODEL_INFERENCE_INPUT = (
    INPUT_PHONEMES,
    INPUT_SPEAKERS,
)


def test_encoder_layer_forward():
    expected_shape = (16, 50, MODEL_CONFIG.phonem_embedding_dim)
    layer = Encoder(ModelConfig.phonem_embedding_dim, config=ENCODER_CONFIG)
    out = layer(PHONEM_EMB.transpose(1, 2), INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (out[idx, length - 1] != 0).any(), f"Wrong zero vector for id = {idx}"


def test_encoder_layer_inference():
    expected_shape = (16, 50, MODEL_CONFIG.phonem_embedding_dim)
    layer = Encoder(ModelConfig.phonem_embedding_dim, config=ENCODER_CONFIG)
    out = layer.inference(PHONEM_EMB.transpose(1, 2))
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_duration_layer_forward():
    expected_shape = (16, 50, 1)
    layer = DurationPredictor(EMBEDDING_DIM, config=DURATION_CONFIG)
    zero_value = layer.projection.linear_layer.bias
    if zero_value is None:
        zero_value = 0
    out = layer(EMBEDDING, INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == zero_value
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            out[idx, length - 1] != zero_value
        ).any(), f"Wrong zero vector for id = {idx}"


def test_duration_layer_inference():
    expected_shape = (16, 50, 1)
    layer = DurationPredictor(EMBEDDING_DIM, config=DURATION_CONFIG)
    out = layer.inference(EMBEDDING)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_range_layer_forward():
    expected_shape = (16, 50, 1)
    layer = RangePredictor(EMBEDDING_DIM, config=RANGE_CONFIG)
    zero_value = layer.projection.linear_layer.bias
    if zero_value is None:
        zero_value = 0
    out = layer(EMBEDDING, INPUT_DURATIONS.unsqueeze(2), INPUT_LENGTH)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"
    for idx, length in enumerate(INPUT_LENGTH):
        assert (
            out[idx, length:] == zero_value
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            out[idx, length - 1] != zero_value
        ).any(), f"Wrong zero vector for id = {idx}"


def test_range_layer_forward_inference():
    expected_shape = (16, 50, 1)
    layer = RangePredictor(EMBEDDING_DIM, config=RANGE_CONFIG)
    out = layer.inference(EMBEDDING, INPUT_DURATIONS.unsqueeze(2))
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_attention_layer_forward():
    expected_shape_out = (
        16,
        DURATIONS_MAX.max().item(),
        EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim,
    )
    expected_shape_dur = (16, 50, 1)
    layer = Attention(
        EMBEDDING_DIM, config=ATTENTION_CONFIG, device=torch.device("cpu")
    )
    dur, out = layer(EMBEDDING, INPUT_LENGTH, INPUT_DURATIONS)
    assert (
        dur.shape == expected_shape_dur
    ), f"Wrong shape, expected {expected_shape_dur}, got: {dur.shape}"
    assert (
        out.shape == expected_shape_out
    ), f"Wrong shape, expected {expected_shape_out}, got: {out.shape}"


@patch("src.models.feature_models.non_attentive_tacatron.model.DurationPredictor.inference")
def test_attention_layer_inference(mock_duration):
    mock_duration.return_value = INPUT_DURATIONS.unsqueeze(2)
    expected_shape_out = (
        16,
        DURATIONS_MAX.max().item(),
        EMBEDDING_DIM + ATTENTION_CONFIG.positional_dim,
    )
    layer = Attention(
        EMBEDDING_DIM, config=ATTENTION_CONFIG, device=torch.device("cpu")
    )
    out = layer.inference(EMBEDDING)
    assert (
        out.shape == expected_shape_out
    ), f"Wrong shape, expected {expected_shape_out}, got: {out.shape}"


def test_prenet_layer():
    expected_shape = (16, 1, DECODER_CONFIG.prenet_layers[-1])

    layer = Prenet(
        MODEL_CONFIG.n_mel_channels,
        DECODER_CONFIG.prenet_layers,
        dropout=DECODER_CONFIG.prenet_dropout,
    )
    out = layer(INPUT_MELS[:, 0, :].unsqueeze(1))
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_decoder_layer_forward():
    expected_shape = (16, DURATIONS_MAX.max(), MODEL_CONFIG.n_mel_channels)

    layer = Decoder(
        MODEL_CONFIG.n_mel_channels,
        ATTENTION_OUT_DIM,
        config=DECODER_CONFIG,
    )
    out = layer(ATTENTION_OUT, INPUT_MELS)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_decoder_layer_inference():
    expected_shape = (16, DURATIONS_MAX.max(), MODEL_CONFIG.n_mel_channels)

    layer = Decoder(
        MODEL_CONFIG.n_mel_channels,
        ATTENTION_OUT_DIM,
        config=DECODER_CONFIG,
    )
    out = layer.inference(ATTENTION_OUT)
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_postnet_layer():
    expected_shape = INPUT_MELS.transpose(1, 2).shape

    layer = Postnet(
        MODEL_CONFIG.n_mel_channels,
        config=POSTNET_CONFIG,
    )
    out = layer(INPUT_MELS.transpose(1, 2))
    assert (
        out.shape == expected_shape
    ), f"Wrong shape, expected {expected_shape}, got: {out.shape}"


def test_model_forward():
    expected_mel_shape = INPUT_MELS.shape
    expected_duration_shape = (16, 50, 1)

    model = NonAttentiveTacatron(
        N_PHONEMES, N_SPEAKER, device=torch.device("cpu"), config=MODEL_CONFIG
    )
    durations, mel_fixed, mel_predicted = model(MODEL_INPUT)
    assert (
        durations.shape == expected_duration_shape
    ), f"Wrong shape, expected {expected_duration_shape}, got: {durations.shape}"
    assert (
        mel_predicted.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_predicted.shape}"
    assert (
        mel_fixed.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_fixed.shape}"
    for idx, length in enumerate(DURATIONS_MAX):
        assert (
            mel_fixed[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            mel_fixed[idx, length - 1] != 0
        ).any(), f"Wrong zero vector for id = {idx}"
        assert (
            mel_predicted[idx, length:] == 0
        ).all(), "All values of tensor higher sequence length must be zero"
        assert (
            mel_predicted[idx, length - 1] != 0
        ).any(), f"Wrong zero vector for id = {idx}"


@patch("src.models.feature_models.non_attentive_tacatron.model.DurationPredictor.inference")
def test_model_inference(mock_duration):
    mock_duration.return_value = INPUT_DURATIONS.unsqueeze(2)
    expected_mel_shape = INPUT_MELS.shape

    model = NonAttentiveTacatron(
        N_PHONEMES, N_SPEAKER, device=torch.device("cpu"), config=MODEL_CONFIG
    )
    mel_predicted = model.inference(MODEL_INFERENCE_INPUT)
    assert (
        mel_predicted.shape == expected_mel_shape
    ), f"Wrong shape, expected {expected_mel_shape}, got: {mel_predicted.shape}"
