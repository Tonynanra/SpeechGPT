import numpy as np
import torch
from transformers.models.whisper import WhisperFeatureExtractor
from typing import List

class customWhisperFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mels = kwargs.pop("feature_size")

    def _torch_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio using the PyTorch STFT implementation.
        Args:
            waveform (:obj:`np.array`): 1D waveform of padded/truncated raw audio.
        Returns:
            log_spec(:obj:`np.ndarray`): Log-mel spectrogram of the audio with shape (n_mels, time).
        """

        ################ Reference implementation ################
        # waveform = torch.from_numpy(waveform).type(torch.float32)

        # window = torch.hann_window(self.n_fft)
        # stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        # magnitudes = stft[..., :-1].abs() ** 2

        # mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        # mel_spec = mel_filters.T @ magnitudes

        # log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        # log_spec = (log_spec + 4.0) / 4.0
        # return log_spec.numpy()
        pass
    

    
    @staticmethod
    # Copied from transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        Args:
            input_values (:obj:`List[np.ndarray]`):
                List of 1D padded/truncated raw waveforms to normalize.
            attention_mask (:obj:`List[np.ndarray]`):
                List of attention masks to apply the normalization.
            padding_value (:obj:`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
        Returns:
            List[np.ndarray]: List of normalized waveforms.
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values