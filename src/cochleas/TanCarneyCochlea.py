from os import makedirs

import numpy as np
from brian2 import Hz, SpikeMonitor, kHz, ms, plot, run, seed, show
from brian2hears import Sound, TanCarney, ZhangSynapse, dB, erbspace
from joblib import Memory
from scipy import signal
from sorcery import dict_of

from utils.path_utils import Paths
from utils.custom_sounds import Tone, ToneBurst
from utils.log_utils import logger, tqdm
from utils.manual_fixes_to_b2h.MiddleEar import MiddleEar

from utils.cochlea_utils import CFMAX, CFMIN, NUM_CF, AnfResponse
from utils.hrtf_utils import run_hrtf

COCHLEA_KEY = f"TanCarney"
CACHE_DIR = Paths.ANF_SPIKES_DIR + COCHLEA_KEY + "/"
makedirs(CACHE_DIR, exist_ok=True)

memory = Memory(location=CACHE_DIR, verbose=0)


def resample_sound(sound: Sound, original_fs, target_fs=50000):
    ratio = target_fs / original_fs
    sound_data = np.array(sound)
    new_length = int(len(sound_data) * ratio)
    # Resample using scipy's resample
    resampled_data = signal.resample(sound_data, new_length)
    return Sound(resampled_data, samplerate=target_fs * Hz)


def resample_binaural_sound(binaural_sound: Sound):
    original_fs = float(binaural_sound.samplerate / Hz)
    # Resample both channels
    left_resampled = resample_sound(binaural_sound.left, original_fs)
    right_resampled = resample_sound(binaural_sound.right, original_fs)
    return Sound((left_resampled, right_resampled), samplerate=50 * kHz)


@memory.cache
def sound_to_spikes(
    sound: Sound | Tone | ToneBurst, angle, params: dict, plot_spikes=False
) -> AnfResponse:
    hrtf_params = params["hrtf_params"]
    rng_seed = params["rng_seed"]
    noise_level = params["omni_noise_level"] * dB
    seed(rng_seed)
    coch_par = params.get("cochlea_params", None)
    logger.debug(
        f"genenerating spikes for {dict_of(sound,angle,plot_spikes,hrtf_params)}"
    )
    binaural = run_hrtf(sound, angle, hrtf_params)
    logger.debug(f"binaural sound post hrtf level={binaural.level}")
    noise = Sound.whitenoise(binaural.duration).atlevel(noise_level)
    logger.debug(f"binaural sound post noise level={binaural.level}")
    # TanCarney needs 50kHz
    binaural_sound = resample_binaural_sound(binaural + noise)
    cf = erbspace(CFMIN, CFMAX, NUM_CF)
    binaural_IHC_response = {}

    logger.info("generating simulated IHC response...")

    for sound, channel in zip([binaural_sound.left, binaural_sound.right], ["L", "R"]):
        logger.debug(f"working on ear {channel}...")
        ihc = TanCarney(MiddleEar(sound), cf, param=coch_par)
        G = ZhangSynapse(ihc, cf, n_per_channel=10)
        M = SpikeMonitor(G)
        for chunk in tqdm(range(10), desc="IHCsim"):
            run(sound.duration / 10)

        if plot_spikes:
            plot(M.t / ms, M.i, ".", markersize=1)
            show()

        binaural_IHC_response[channel] = M.spike_trains()
    logger.info("generation complete.")
    return AnfResponse(binaural_IHC_response, binaural_sound.left, binaural_sound.right)
