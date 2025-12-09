import torch
import torch.utils.data
import librosa
from librosa.filters import mel as librosa_mel_fn
import numpy as np

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def _stft_compatible(y, n_fft, hop_length, win_length, window, center=False):
    """
    Versión compatible de torch.stft que maneja return_complex automáticamente
    """
    # Primero intentamos con return_complex=True (compatible con PyTorch >= 2.0)
    try:
        spec_complex = torch.stft(
            y,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # Convertimos a formato real para mantener compatibilidad
        return torch.stack([spec_complex.real, spec_complex.imag], dim=-1)
    except TypeError:
        # Fallback para versiones antiguas que requieren return_complex=False
        return torch.stft(
            y,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )


def _pad_center_manual(data, size):
    """
    Reemplazo manual de librosa.util.pad_center
    """
    n = data.shape[-1]
    lpad = int((size - n) // 2)
    rpad = int(size - n - lpad)
    
    if lpad < 0 or rpad < 0:
        raise ValueError(f"Target size ({size}) must be at least input size ({n})")
    
    return torch.nn.functional.pad(data, (lpad, rpad), mode='constant', value=0)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = _stft_compatible(
        y,
        n_fft,
        hop_size,
        win_size,
        hann_window[wnsize_dtype_device],
        center
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spectrogram_torch_conv(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    
    # ******************** ConvSTFT ************************#
    freq_cutoff = n_fft // 2 + 1
    fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(n_fft, device=y.device, dtype=y.dtype)))
    forward_basis = fourier_basis[:freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
    
    # Reemplazo de librosa.util.pad_center
    try:
        # Intentar con librosa si está disponible
        import librosa.util
        window_padded = torch.as_tensor(librosa.util.pad_center(
            torch.hann_window(win_size, device=y.device, dtype=y.dtype).cpu().numpy(), 
            size=n_fft
        ), device=y.device, dtype=y.dtype).float()
    except (AttributeError, ImportError):
        # Usar implementación manual
        window_padded = _pad_center_manual(torch.hann_window(win_size, device=y.device, dtype=y.dtype), n_fft).float()
    
    forward_basis = forward_basis * window_padded

    import torch.nn.functional as F

    assert center is False

    forward_transform_squared = F.conv1d(y, forward_basis.to(y.device), stride=hop_size)
    spec2 = torch.stack([forward_transform_squared[:, :freq_cutoff, :], forward_transform_squared[:, freq_cutoff:, :]], dim=-1)

    # ******************** Verification ************************#
    spec1 = _stft_compatible(
        y.squeeze(1),
        n_fft,
        hop_size,
        win_size,
        hann_window[wnsize_dtype_device],
        center
    )
    
    # Verificación con tolerancia
    if not torch.allclose(spec1, spec2, atol=1e-4):
        print("⚠️ Advertencia: spec1 y spec2 no son idénticos (diferencia > 1e-4)")
        print(f"  Diferencia máxima: {(spec1 - spec2).abs().max().item()}")

    spec = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = _stft_compatible(
        y,
        n_fft,
        hop_size,
        win_size,
        hann_window[wnsize_dtype_device],
        center
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
