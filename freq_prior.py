"""Radial power-spectrum prior: instead of matching a scalar ink-fraction
budget, match the *spatial-frequency content* of the real drawings --
how much energy lives at fine-line-scale frequencies vs. blob-scale
frequencies. A blob and a thin line can have the same total ink but very
different frequency signatures (a thin line has much more high-frequency
energy for the same area) -- this pushes the synthesized image toward
actually looking line-like in the same sense the real dataset does,
rather than just matching how much ink is used."""
import numpy as np
import torch

IMG_SIZE = 128
N_BINS = 32


def _radial_bins(size=IMG_SIZE, n_bins=N_BINS):
    freqs = np.fft.fftshift(np.fft.fftfreq(size))
    fy, fx = np.meshgrid(freqs, freqs, indexing="ij")
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = r.max()
    bin_idx = np.clip((r / r_max * n_bins).astype(int), 0, n_bins - 1)
    return torch.from_numpy(bin_idx)


_BIN_IDX = _radial_bins()


def radial_spectrum(ink):
    """ink: (H,W) or (1,1,H,W) torch tensor in [0,1] (1 = dark/ink).
    Returns an (N_BINS,) radially-averaged log-power spectrum, differentiable."""
    if ink.dim() == 4:
        ink = ink[0, 0]
    F = torch.fft.fft2(ink)
    F = torch.fft.fftshift(F)
    power = (F.real ** 2 + F.imag ** 2)
    log_power = torch.log1p(power)

    bins = _BIN_IDX
    out = torch.zeros(N_BINS, dtype=log_power.dtype)
    counts = torch.zeros(N_BINS, dtype=log_power.dtype)
    out = out.scatter_add(0, bins.flatten(), log_power.flatten())
    counts = counts.scatter_add(0, bins.flatten(), torch.ones_like(log_power.flatten()))
    return out / counts.clamp(min=1)


@torch.no_grad()
def reference_spectrum(images):
    """images: list of (H,W) numpy/torch arrays in [0,1] (grayscale, white
    background). Returns the average radial log-power spectrum of their
    ink (1-img) maps, as a plain torch tensor (no grad needed -- this is
    a fixed target)."""
    specs = []
    for im in images:
        if not torch.is_tensor(im):
            im = torch.from_numpy(im)
        ink = 1 - im.float()
        specs.append(radial_spectrum(ink))
    return torch.stack(specs).mean(dim=0)
