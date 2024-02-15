import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import io

import commons
import utils
from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def save_audio_as_mp3(audio, sample_rate, file_name):
    # Convert the numpy array to bytes
    audio_bytes = io.BytesIO()
    audio_scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(audio_bytes, sample_rate, audio_scaled)

    # Convert the bytes to AudioSegment
    audio_bytes.seek(0)  # Go to the start of BytesIO object
    audio_segment = AudioSegment.from_wav(audio_bytes)

    # Export as MP3
    audio_segment.export(file_name, format="mp3")


# Single Speaker
hps = utils.get_hparams_from_file("../drive/MyDrive/vits-finetune/ljs_base/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(
    "../drive/MyDrive/vits-finetune/ljs_base/G_1000.pth", net_g, None
)

stn_tst = get_text(
    "My name is Barrack Obama and I am the president of the United State", hps
)

with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = (
        net_g.infer(
            x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
        )[0][0, 0]
        .data.cpu()
        .float()
        .numpy()
    )

# Save single speaker audio
# save_audio_as_mp3(
#     audio,
#     hps.data.sampling_rate,
#     "../drive/MyDrive/vits-finetune/single_speaker_synthesized_audio.mp3",
# )

# # Multiple Speakers
# hps = utils.get_hparams_from_file("../drive/MyDrive/vits-finetune/vltk_base/config.json")

# net_g = SynthesizerTrn(
#     len(symbols),
#     hps.data.filter_length // 2 + 1,
#     hps.train.segment_size // hps.data.hop_length,
#     n_speakers=hps.data.n_speakers,
#     **hps.model
# ).cuda()
# _ = net_g.eval()

# _ = utils.load_checkpoint(
#     "../drive/MyDrive/vits-finetune/checkpoints/G_epoch_20000.pth", net_g, None
# )

# stn_tst = get_text("We have the best economy and government in history", hps)
# with torch.no_grad():
#     x_tst = stn_tst.cuda().unsqueeze(0)
#     x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
#     sid = torch.LongTensor([10]).cuda()
#     audio = (
#         net_g.infer(
#             x_tst,
#             x_tst_lengths,
#             sid=sid,
#             noise_scale=0.667,
#             noise_scale_w=0.8,
#             length_scale=1,
#         )[0][0, 0]
#         .data.cpu()
#         .float()
#         .numpy()
#     )

# # Save multiple speakers audio
# save_audio_as_mp3(
#     audio,
#     hps.data.sampling_rate,
#     "../drive/MyDrive/vits-finetune/multiple_speakers_synthesized_audio.mp3",
# )
