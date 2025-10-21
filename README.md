# Generalized-Modulation

# Filters, Delays, Modulations and Demodulations

Perfect — here’s a **conceptual and mathematical overview** of the four pillars covered in *DUT52.PS.pdf* (DAFx’98 tutorial by Julius O. Smith):
“**Filters, Delays, Modulations and Demodulations**.”
This distills the main **ideas, equations, and design insights** while staying mathematically accurate to the original paper.

---

# 🎛 1. Filters — Concept and Mathematics

## Concept

Filters shape or extract specific frequency components of signals.
They are the **fundamental tone-shaping** and **analysis tools** in DSP.

Two primary categories:

* **IIR filters (recursive)** — infinite impulse response, efficient but can be unstable.
* **FIR filters (non-recursive)** — finite impulse response, always stable, often used in convolution.

### Types (and purposes)

| Type                        | Function                                        |
| --------------------------- | ----------------------------------------------- |
| **Lowpass (LP)**            | Keeps slow variations, removes fast ones.       |
| **Highpass (HP)**           | Removes DC or rumble.                           |
| **Bandpass (BP)**           | Extracts a frequency band (formant, resonance). |
| **Bandreject / Notch (BR)** | Suppresses narrow bands (hum, whistle).         |
| **Resonator**               | Emphasizes a narrow band (adds tone color).     |

---

## Mathematical Form

### Analog prototype

A general **2nd-order filter** in Laplace domain:
[
H(s) = \frac{\omega_0^2}{s^2 + 2\zeta\omega_0 s + \omega_0^2}
]
where

* ( \omega_0 = 2\pi f_c ): natural frequency
* ( \zeta ): damping ratio (related to resonance or ( Q = 1/(2\zeta) ))

---

### Digital implementation

#### Canonical Biquad (IIR form)

[
y[n] = b_0x[n] + b_1x[n-1] + b_2x[n-2] - a_1y[n-1] - a_2y[n-2]
]
Coefficients derived from bilinear transform of analog prototype.

---

#### State Variable Filter (SVF)

Highly flexible and stable for real-time control:
[
\begin{aligned}
v_1[n] &= v_1[n-1] + F \cdot (x[n] - v_2[n-1] - 2R \cdot v_1[n-1]) \
v_2[n] &= v_2[n-1] + F \cdot v_1[n]
\end{aligned}
]
with

* ( F = 2\sin(\pi f_c / f_s) ) (normalized frequency)
* ( R = 1/(2Q) )

Outputs:
[
\begin{cases}
LP = v_2[n] \
BP = v_1[n] \
HP = x[n] - 2R v_1[n] - v_2[n]
\end{cases}
]

**Advantages:**
Smooth tuning, multiple outputs, stable for small ( F ), good for sweeps (filters in synthesizers).

---

### Filter Normalization (Level Compensation)

Filters change gain when ( f_c ) or ( Q ) vary → unwanted loudness shifts.

#### Norm criteria

| Type   | Description                                       |
| ------ | ------------------------------------------------- |
| **L₁** | Guarantees no clipping (safe).                    |
| **L₂** | Perceptually even loudness for broadband signals. |
| **L∞** | Equal amplitude for sinusoidal inputs.            |

Example normalization (for SVF):
[
G_{norm} = \sqrt{\zeta} \cdot \frac{1}{\sqrt{1 + (f_c / f_0)^2}}
]
This prevents resonance “blow-up” when ( Q ) → high.

---

# ⏱ 2. Delays — Concept and Mathematics

## Concept

Delay elements store signal samples to create **echo, comb filtering, flanging, chorus, or reverb**.

* **FIR delay:** feedforward delay (coloration, combs).
* **IIR delay:** feedback delay (resonance, echo, infinite tail).

---

## Mathematical Forms

### FIR Comb

[
y[n] = x[n] + g , x[n - M]
]
Frequency response:
[
H(e^{j\omega}) = 1 + g e^{-j\omega M}
]
Periodic peaks every ( 2\pi/M ).
Use for static coloration (string resonances, phasing).

---

### IIR Comb

[
y[n] = c,x[n] + g,y[n - M]
]
Gain response has resonant peaks at:
[
\omega_k = \frac{2\pi k}{M}
]
Bandwidth (approx):
[
\Delta f = \frac{(1 - |g|) f_s}{\pi M}
]
Stable if ( |g| < 1 ).

---

### “Natural” Comb (with LP in feedback)

Adds realism by damping high frequencies:
[
y[n] = c,x[n] + g \cdot H_{LP}(y[n - M])
]
where ( H_{LP} ) is a 1st-order lowpass:
[
H_{LP}(z) = \frac{1 - a}{1 - a z^{-1}}
]
→ reduces metallic sound, simulates acoustic resonators.

---

### Normalization

Avoid overload and loudness jumps.

* ( L_\infty ): ( c = 1 / (1 - |g|) )
* ( L_2 ): ensures broadband power balance.

---

# 🌊 3. Modulations — Concept and Mathematics

## Concept

Modulation means **multiplying** two signals → shifts or spreads spectra.

### Ring Modulation (RM)

[
y(t) = m(t) \cdot c(t)
]
with carrier ( c(t) = \sin(2\pi f_c t) )

Resulting spectrum:
[
f_{sum} = f_m + f_c, \quad f_{diff} = |f_m - f_c|
]
Carrier frequency cancels — produces metallic/in-harmonic tones.

---

### Amplitude Modulation (AM)

[
y(t) = [1 + \delta m(t)] \cdot c(t)
]
Spectrum: carrier ± sidebands.

* For low ( f_m ) (<20 Hz): **tremolo**
* For audio-rate ( f_m ): **sideband creation**

---

### Single Sideband Modulation (SSB)

Uses **Hilbert transform** to cancel one sideband.

Hilbert pair:
[
m_H(t) = \mathcal{H}{m(t)}, \quad c_H(t) = \mathcal{H}{c(t)}
]
Upper sideband (USB):
[
y_{USB}(t) = \text{Re}{ [m(t) + j m_H(t)] [c(t) + j c_H(t)] }
]
Lower sideband (LSB):
[
y_{LSB}(t) = \text{Re}{ [m(t) + j m_H(t)] [c(t) - j c_H(t)] }
]
Used for pitch shifting and spectral translation.

---

# 📡 4. Demodulations — Concept and Mathematics

## Concept

Demodulation extracts **amplitude envelopes or control signals** from modulated inputs — e.g., in AM receivers, compressors, or envelope followers.

---

## Mathematical Models

### Envelope Detection

* **Rectify:** half-wave, full-wave, or squared signal
* **Lowpass filter:** smooths the amplitude envelope

Half-wave:
[
y[n] = \max(x[n], 0)
]
Full-wave:
[
y[n] = |x[n]|
]
Square-law (RMS-like):
[
y[n] = x[n]^2
]

---

### Averaging (Envelope smoothing)

1st-order lowpass smoother:
[
y[n] = (1 - a) , x[n] + a,y[n-1]
]
with ( a = e^{-1/(\tau f_s)} ) where τ = time constant.

---

### Attack–Release Averager

Different constants for attack/release:
[
a = e^{-1/(\tau_a f_s)}, \quad r = e^{-1/(\tau_r f_s)}
]
[
y[n] =
\begin{cases}
(1 - a)|x[n]| + a,y[n-1], & \text{if } |x[n]| > y[n-1] \
(1 - r)|x[n]| + r,y[n-1], & \text{otherwise}
\end{cases}
]
Used in compressors, envelope followers, and AM detectors.

---

# 🧩 Summary Table

| Domain                 | Key Equation                                          | Main Use                 | Normalization |                        |                          |
| ---------------------- | ----------------------------------------------------- | ------------------------ | ------------- | ---------------------- | ------------------------ |
| **Filter (SVF)**       | 2 × 1st-order integrators, ( F = 2\sin(\pi f_c/f_s) ) | LP/BP/HP control         | √ζ gain comp  |                        |                          |
| **Comb (FIR/IIR)**     | ( y = x + g,x[n-M] ) / ( y = c x + g y[n-M] )         | Echo, coloration, reverb | ( c=1/(1-     | g                      | ) )                      |
| **Modulation (RM/AM)** | ( y = m·c ) / ( y=(1+δm)c )                           | Timbre change, tremolo   | none          |                        |                          |
| **Demodulation**       | Envelope via                                          | x                        | + LP          | AM detection, envelope | attack/release smoothing |

---

✅ **In essence:**

* Filters and delays are **spectral shapers**.
* Modulations and demodulations are **spectral movers**.
* Normalization ensures **perceptual consistency** when parameters vary.

---

Would you like me to follow this up with a **diagrammatic summary** (flow/block diagrams for each process: filter, comb, modulator, demodulator)? It’d visually connect the math and signal paths exactly as presented in the DAFx paper.

---


absolutely. below is a compact, production-friendly toolkit in pure Python + NumPy + Numba (jit), using simple “procedural classes” with vectorized/JIT’d inner loops.

it covers:

* state-variable filter (TPT/SVF) with LP/BP/HP outputs (+ optional loudness normalization)
* FIR & IIR combs (with optional fractional delay; IIR can include a 1-pole LP inside the feedback loop)
* modulated delay (vibrato / flanger / chorus)
* ring & amplitude modulation; SSB (single-sideband) using an FFT Hilbert
* envelope follower (attack/release), plus simple demod helpers

> everything processes 1-D mono blocks. you can call per-sample parameter arrays for sweeps (e.g., `fc` as an array) or scalars for static settings.

```python
# dsp_toolkit.py
from __future__ import annotations
import numpy as np
from numba import njit

# -----------------------------
# Utilities
# -----------------------------

def _as_array(param, N):
    """Broadcast scalar or 1-D array to length N (no copy for arrays)."""
    if np.isscalar(param):
        return np.full(N, float(param), dtype=np.float64)
    arr = np.asarray(param, dtype=np.float64)
    if arr.shape[0] != N:
        raise ValueError(f"param length {arr.shape[0]} != block length {N}")
    return arr

@njit(cache=True, fastmath=True)
def _one_pole_lp_block(x, a, z0):
    """
    First-order lowpass: y[n] = (1-a)*x[n] + a*y[n-1]
    a in [0,1). Returns (y, zf)
    """
    N = x.shape[0]
    y = np.empty_like(x)
    y_prev = z0
    one_minus_a = 1.0 - a
    for n in range(N):
        y_cur = one_minus_a * x[n] + a * y_prev
        y[n] = y_cur
        y_prev = y_cur
    return y, y_prev

def _hilbert_analytic(x: np.ndarray) -> np.ndarray:
    """
    Analytic signal via FFT method (no SciPy). Complex ndarray.
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    X = np.fft.rfft(x)
    H = np.zeros_like(X, dtype=np.complex128)

    if N % 2 == 0:
        # even: k=0 and k=N/2 untouched; 1..N/2-1 doubled
        H[0] = 1
        H[-1] = 1
        if X.size > 2:
            H[1:-1] = 2
    else:
        # odd: k=0 untouched; 1..(N-1)/2 doubled
        H[0] = 1
        if X.size > 1:
            H[1:] = 2

    x_analytic = np.fft.irfft(X * H, n=N)
    return x + 1j * x_analytic

# -----------------------------
# TPT State-Variable Filter (SVF)
# Based on topology-preserving transform; stable, sweepable.
# -----------------------------

@njit(cache=True, fastmath=True)
def _svf_tpt_block(x, F, Q, z1_in, z2_in, norm_gain):
    """
    TPT-SVF per Vadim Zavalishin style.
    Inputs:
      x: signal
      F: 0..1 (~ 2*sin(pi*fc/fs))
      Q: >= 0  (Q = 1/(2*ζ))
      z1_in, z2_in: state
      norm_gain: pre/post gain factor (for loudness normalization)
    Returns: (lp, bp, hp, z1_out, z2_out)
    """
    N = x.shape[0]
    lp = np.empty_like(x)
    bp = np.empty_like(x)
    hp = np.empty_like(x)

    z1 = z1_in
    z2 = z2_in

    for n in range(N):
        f = F[n]
        q = Q[n] if Q.shape[0] == N else Q[0]
        g = f  # in this parameterization
        R = 1.0 / (2.0 * q + 1e-12)  # avoid div0; q=0 => max damping

        # optional input gain to help keep level steady
        xn = x[n] * norm_gain[n] if norm_gain.shape[0] == N else x[n] * norm_gain[0]

        # core TPT-SVF
        hp_n = (xn - (R + 1.0) * z1 - z2) / (1.0 + R * g + g * g)
        bp_n = g * hp_n + z1
        lp_n = g * bp_n + z2

        # update states
        z1 = g * hp_n + bp_n
        z2 = g * bp_n + lp_n

        hp[n] = hp_n
        bp[n] = bp_n
        lp[n] = lp_n

    return lp, bp, hp, z1, z2

class SVF:
    """
    State-Variable Filter producing LP/BP/HP simultaneously.

    Parameters
    ----------
    fs : float
        Sample rate.
    mode : str
        'lp', 'bp', 'hp' (what .process returns by default).
    normalize : bool
        If True, applies light L2-ish gain comp vs fc & Q.
    """
    def __init__(self, fs: float, mode: str = "lp", normalize: bool = True):
        self.fs = float(fs)
        self.mode = mode
        self.normalize = normalize
        self.z1 = 0.0
        self.z2 = 0.0

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0

    def _map_fc(self, fc):
        # F = 2*sin(pi*fc/fs); clamp for stability
        F = 2.0 * np.sin(np.pi * _as_array(fc, self._N) / self.fs)
        return np.clip(F, 0.0, 1.9)  # headroom; TPT is robust

    def _norm_gain(self, fc, Q):
        if not self.normalize:
            return np.array([1.0])
        # simple psycho-ish comp: √ζ with a mild HF roll-off
        Qarr = _as_array(Q, self._N) if np.isscalar(Q) else np.asarray(Q, dtype=np.float64)
        zeta = 1.0 / (2.0 * Qarr + 1e-12)
        g_q = np.sqrt(np.clip(zeta, 1e-4, 10.0))
        # mild 1st-order LP vs frequency to keep broadband loudness steady
        fc_arr = _as_array(fc, self._N)
        # heuristic: higher fc -> slightly lower gain
        g_fc = 1.0 / np.sqrt(1.0 + (fc_arr / (0.15 * self.fs))**2)
        return g_q * g_fc

    def process(self, x: np.ndarray, fc, Q=0.707, return_all=False):
        """
        Process a block.

        fc: Hz (scalar or array len N)
        Q : quality factor (scalar or array len N), Q=1/(2ζ)
        return_all: if True, returns (lp, bp, hp). Else returns mode-selected output.
        """
        x = np.asarray(x, dtype=np.float64)
        self._N = x.shape[0]
        F = self._map_fc(fc)
        Qarr = _as_array(Q, self._N) if np.isscalar(Q) else np.asarray(Q, dtype=np.float64)
        ng = self._norm_gain(fc, Qarr)

        lp, bp, hp, self.z1, self.z2 = _svf_tpt_block(x, F, Qarr, self.z1, self.z2, ng)

        if return_all:
            return lp, bp, hp
        return {"lp": lp, "bp": bp, "hp": hp}[self.mode]


# -----------------------------
# Delay line with fractional indexing (linear interp)
# -----------------------------

@njit(cache=True, fastmath=True)
def _frac_sample(buf, wptr, delay_samp):
    """
    Read past sample at fractional delay from circular buffer.
    delay_samp >= 0. Linear interpolation.
    """
    N = buf.shape[0]
    read_pos = wptr - delay_samp
    while read_pos < 0.0:
        read_pos += N
    i0 = int(read_pos) % N
    frac = read_pos - int(read_pos)
    i1 = (i0 + 1) % N
    return (1.0 - frac) * buf[i0] + frac * buf[i1]

@njit(cache=True, fastmath=True)
def _write_advance(buf, wptr, val):
    N = buf.shape[0]
    buf[wptr] = val
    wptr += 1
    if wptr >= N:
        wptr = 0
    return wptr

# -----------------------------
# FIR & IIR Comb (with optional feedback LP in IIR)
# -----------------------------

@njit(cache=True, fastmath=True)
def _comb_fir_block(x, delay_samps, g, buf, wptr, mix):
    """
    y = dry + g * x[n - M]
    delay_samps: scalar or array
    """
    N = x.shape[0]
    y = np.empty_like(x)
    g0 = g
    for n in range(N):
        M = delay_samps[n] if delay_samps.shape[0] == N else delay_samps[0]
        delayed = _frac_sample(buf, wptr, M)
        out = (1.0 - mix) * x[n] + mix * (x[n] + g0 * delayed)
        y[n] = out
        # write current input to buffer
        wptr = _write_advance(buf, wptr, x[n])
    return y, wptr

@njit(cache=True, fastmath=True)
def _comb_iir_block(x, delay_samps, g, c, buf, wptr, lp_a, lp_state, mix):
    """
    Feedback comb: y = c*x + g*y[n-M]; optional 1-pole LP in feedback path.
    lp_a in [0,1): feedback lowpass coefficient (higher -> darker).
    """
    N = x.shape[0]
    y = np.empty_like(x)
    y_prev = 0.0
    lp_s = lp_state
    for n in range(N):
        M = delay_samps[n] if delay_samps.shape[0] == N else delay_samps[0]
        # read past output (stored in buffer)
        y_delayed = _frac_sample(buf, wptr, M)
        # apply LP in feedback path: fb = LP(y_delayed)
        fb = (1.0 - lp_a) * y_delayed + lp_a * lp_s
        lp_s = fb
        # comb recursion
        y_n = c * x[n] + g * fb
        # mix
        out = (1.0 - mix) * x[n] + mix * y_n
        y[n] = out
        # write current output to buffer (because feedback uses y)
        wptr = _write_advance(buf, wptr, y_n)
        y_prev = y_n
    return y, wptr, lp_s

class CombFIR:
    """
    FIR comb: y = x + g * x[n-M].
    """
    def __init__(self, fs, max_delay_s=2.0, mix=1.0, g=0.5):
        self.fs = float(fs)
        self.mix = float(mix)
        self.g = float(g)
        max_len = int(np.ceil(max_delay_s * self.fs)) + 2
        self.buf = np.zeros(max_len, dtype=np.float64)
        self.wptr = 0

    def reset(self):
        self.buf[:] = 0.0
        self.wptr = 0

    def process(self, x, delay_s):
        x = np.asarray(x, dtype=np.float64)
        N = x.shape[0]
        delay_samps = _as_array(delay_s * self.fs, N)
        y, self.wptr = _comb_fir_block(x, delay_samps, self.g, self.buf, self.wptr, self.mix)
        return y

class CombIIR:
    """
    IIR comb: y = c*x + g*y[n-M], with optional LP in feedback (lp_a).
    Keep |g| <= 1 for stability. For “natural” resonators, set lp_a ~ 0.6..0.98.
    """
    def __init__(self, fs, max_delay_s=2.0, mix=1.0, g=0.7, c=1.0, lp_a=0.0):
        self.fs = float(fs)
        self.mix = float(mix)
        self.g = float(g)
        self.c = float(c)
        self.lp_a = float(lp_a)  # 0=no LP; 0.9 is fairly dark
        max_len = int(np.ceil(max_delay_s * self.fs)) + 2
        self.buf = np.zeros(max_len, dtype=np.float64)
        self.wptr = 0
        self.lp_state = 0.0

    def reset(self):
        self.buf[:] = 0.0
        self.wptr = 0
        self.lp_state = 0.0

    def process(self, x, delay_s):
        x = np.asarray(x, dtype=np.float64)
        N = x.shape[0]
        delay_samps = _as_array(delay_s * self.fs, N)
        y, self.wptr, self.lp_state = _comb_iir_block(
            x, delay_samps, self.g, self.c, self.buf, self.wptr, self.lp_a, self.lp_state, self.mix
        )
        return y

# -----------------------------
# Modulated Delay (vibrato/flanger/chorus building block)
# -----------------------------

@njit(cache=True, fastmath=True)
def _mod_delay_block(x, base_M, depth, lfo, buf, wptr, mix, feedback):
    """
    base_M: base delay in samples
    depth : modulation depth in samples
    lfo   : array in [-1,1]
    feedback: portion of delayed signal fed back into write
    """
    N = x.shape[0]
    y = np.empty_like(x)
    fb_state = 0.0
    for n in range(N):
        M = base_M + depth * (0.5 * (lfo[n] + 1.0))  # map [-1,1] -> [0,1]
        delayed = _frac_sample(buf, wptr, M)
        # output
        out = (1.0 - mix) * x[n] + mix * delayed
        y[n] = out
        # write with feedback
        w = x[n] + feedback * delayed
        wptr = _write_advance(buf, wptr, w)
    return y, wptr

class ModulatedDelay:
    """
    Generic modulated delay line. Use:
      - Vibrato: mix=1.0, small base (5-10ms), depth few ms, no feedback.
      - Flanger: mix ~0.5, base < 15ms, depth ~ few ms, feedback +-0.3.
      - Chorus: base 10-25ms, depth ~ a few ms, possibly multiple voices.
    """
    def __init__(self, fs, max_delay_s=0.05, mix=0.5, feedback=0.0):
        self.fs = float(fs)
        self.mix = float(mix)
        self.feedback = float(feedback)
        max_len = int(np.ceil(max_delay_s * self.fs)) + 2
        self.buf = np.zeros(max_len, dtype=np.float64)
        self.wptr = 0

    def reset(self):
        self.buf[:] = 0.0
        self.wptr = 0

    def process(self, x, base_delay_s, depth_s, lfo):
        x = np.asarray(x, dtype=np.float64)
        N = x.shape[0]
        lfo = _as_array(lfo, N)
        base_M = base_delay_s * self.fs
        depth = depth_s * self.fs
        y, self.wptr = _mod_delay_block(
            x, base_M, depth, lfo, self.buf, self.wptr, self.mix, self.feedback
        )
        return y

# -----------------------------
# Modulation (RM/AM/SSB)
# -----------------------------

class RingMod:
    """Ring modulation: y = m * c"""
    def process(self, modulator, carrier):
        modulator = np.asarray(modulator, dtype=np.float64)
        carrier = np.asarray(carrier, dtype=np.float64)
        if modulator.shape != carrier.shape:
            raise ValueError("Shapes must match.")
        return modulator * carrier

class AmplitudeMod:
    """Amplitude modulation: y = (1 + depth*m)*c; for tremolo use low-f m."""
    def __init__(self, depth=1.0):
        self.depth = float(depth)

    def process(self, modulator, carrier):
        m = np.asarray(modulator, dtype=np.float64)
        c = np.asarray(carrier, dtype=np.float64)
        if m.shape != c.shape:
            raise ValueError("Shapes must match.")
        return (1.0 + self.depth * m) * c

class SSBModulator:
    """
    Single-Sideband modulation (spectral shift) using Hilbert transforms.
    Given real modulator m and real carrier c:
      y = Re{ (m + j*H(m)) * (c + j*H(c)) } for upper sideband,
      or y = Re{ (m + j*H(m)) * (c - j*H(c)) } for lower sideband.
    """
    def __init__(self, sideband: str = "upper"):
        if sideband not in ("upper", "lower"):
            raise ValueError("sideband must be 'upper' or 'lower'")
        self.sideband = sideband

    def process(self, modulator, carrier):
        m = np.asarray(modulator, dtype=np.float64)
        c = np.asarray(carrier, dtype=np.float64)
        if m.shape != c.shape:
            raise ValueError("Shapes must match.")
        ma = _hilbert_analytic(m)
        ca = _hilbert_analytic(c)
        if self.sideband == "upper":
            y = np.real(ma * ca)
        else:
            y = np.real(ma * np.conj(ca))
        return y

# -----------------------------
# Envelope follower & demod
# -----------------------------

@njit(cache=True, fastmath=True)
def _env_ar_block(x, ga, gr, y0):
    """
    Attack/Release envelope:
      if |x| > y: y = ga*y + (1-ga)*|x|
      else:       y = gr*y + (1-gr)*|x|
    Returns (env, y_end)
    """
    N = x.shape[0]
    env = np.empty_like(x)
    y = y0
    for n in range(N):
        xn = abs(x[n])
        if xn > y:
            y = ga * y + (1.0 - ga) * xn
        else:
            y = gr * y + (1.0 - gr) * xn
        env[n] = y
    return env, y

def _time_const_to_coef(tau_s, fs):
    if tau_s <= 0:
        return 0.0
    return float(np.exp(-1.0 / (tau_s * fs)))

class EnvelopeFollower:
    """
    Attack/Release envelope follower.
    Typical: tau_a ~ 0.005 (5ms), tau_r ~ 0.3..1.5 (s) for meters.
    """
    def __init__(self, fs, tau_a=0.005, tau_r=0.3):
        self.fs = float(fs)
        self.ga = _time_const_to_coef(tau_a, self.fs)
        self.gr = _time_const_to_coef(tau_r, self.fs)
        self.y = 0.0

    def reset(self):
        self.y = 0.0

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        env, self.y = _env_ar_block(x, self.ga, self.gr, self.y)
        return env

# Convenience demod helpers
def am_demod_envelope(x, fs, tau_a=0.001, tau_r=0.03):
    """
    Basic AM demod: take magnitude of analytic signal (Hilbert),
    then smooth with attack/release.
    """
    env_inst = np.abs(_hilbert_analytic(np.asarray(x, dtype=np.float64)))
    follower = EnvelopeFollower(fs, tau_a, tau_r)
    return follower.process(env_inst)
```

## tiny usage demo

```python
if __name__ == "__main__":
    import numpy as np

    fs = 48000
    t = np.arange(0, 1.0, 1/fs)
    x = 0.3*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*330*t)

    # --- SVF sweep: 200 Hz -> 6 kHz bandpass
    svf = SVF(fs, mode="bp", normalize=True)
    fc_sweep = np.linspace(200, 6000, x.size)
    y_svf = svf.process(x, fc=fc_sweep, Q=1.0)

    # --- FIR comb: 10 ms
    comb_fir = CombFIR(fs, max_delay_s=0.1, g=0.7, mix=1.0)
    y_fir = comb_fir.process(x, delay_s=0.010)

    # --- IIR comb with LP in feedback (natural resonator flavor)
    comb_iir = CombIIR(fs, max_delay_s=0.1, g=0.8, c=0.5, lp_a=0.9, mix=1.0)
    y_iir = comb_iir.process(x, delay_s=0.020)

    # --- Modulated delay (flanger-ish): 5 ms base, 3 ms depth, 0.5 Hz sine LFO
    md = ModulatedDelay(fs, max_delay_s=0.05, mix=0.5, feedback=0.3)
    lfo = np.sin(2*np.pi*0.5*t)
    y_flange = md.process(x, base_delay_s=0.005, depth_s=0.003, lfo=lfo)

    # --- Ring mod: 700 Hz carrier
    rm = RingMod()
    carrier = np.sin(2*np.pi*700*t)
    y_rm = rm.process(x, carrier)

    # --- AM (tremolo): 5 Hz LFO @ depth 0.8
    am = AmplitudeMod(depth=0.8)
    trem = am.process(np.sin(2*np.pi*5*t), x)

    # --- SSB: shift via SSB mixing with a 200 Hz sinusoid
    ssb = SSBModulator(sideband="upper")
    y_ssb = ssb.process(x, np.sin(2*np.pi*200*t))

    # --- Envelope follower
    env = EnvelopeFollower(fs, tau_a=0.005, tau_r=0.5).process(x)

    print("ok: y_svf, y_fir, y_iir, y_flange, y_rm, trem, y_ssb, env",
          [arr.shape for arr in (y_svf, y_fir, y_iir, y_flange, y_rm, trem, y_ssb, env)])
```

### notes & tips

* **SVF stability:** the TPT form is robust. Still, avoid `fc` too close to `fs/2`. The mapping `F=2*sin(pi*fc/fs)` already flattens high-end sensitivity.
* **Comb IIR levels:** keep `|g| ≤ 1`. If you want hard safety, lower `c` (e.g., 0.5) or post-scale.
* **Feedback LP (`lp_a`)** inside `CombIIR`: 0 → bright metallic; 0.9–0.98 → more “acoustic”.
* **Fractional delays** use linear interpolation for speed. Swap to higher-order Lagrange if you need ultra-clean chorusing.
* **SSB** uses FFT Hilbert (no SciPy). It produces upper or lower sideband depending on `sideband`.

if you want me to tailor any class (e.g., stereo processing, batched frames, higher-order interpolation, or different SVF parameterization) just say the word and i’ll drop it in.

---

love this. here’s a clean unification in two parts:

# 1) The unified math (LTV state-space)

We can express **filters, delays, modulations, and demodulations** as a linear time-varying (LTV) system:
[
\begin{aligned}
\mathbf{x}_{n+1} &= \mathbf{A}[n];\mathbf{x}_n + \mathbf{B}[n];u[n] \
y[n] &= \mathbf{C}[n];\mathbf{x}_n + \mathbf{D}[n];u[n]
\end{aligned}
]

* **Static filters** → (\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}) constant.
* **Time-varying filters** (sweeps, compressors’ envelopes) → matrices vary with (n).
* **Delays/comb/reverb** → state is the **delay line**; feedback sits in (\mathbf{A}[n]).
* **Modulation** → multiply input or state by (g[n]) (lives in (\mathbf{B}[n]) or (\mathbf{A}[n])).
* **Demodulation** → multiply by a carrier (modulation), then low-pass (filter), i.e., still LTV.

For everyday coding, it’s handy to also use a **unified difference equation**:
[
\boxed{,y[n] = m[n]\cdot x[n] + \sum_{i=1}^{P} b_i[n];x[n-i] ;+; d[n];\underbrace{x[n-\tau[n]]}*{\text{frac. delay}};+; \sum*{j=1}^{Q} a_j[n];y[n-j],}
]
This single form covers:

* **AM/RM**: (m[n]) (and/or a separate carrier)
* **Biquads & SVF-as-biquad**: time-varying (b_i[n],a_j[n])
* **Flanger/Chorus/Vibrato**: fractional delay term (x[n-\tau[n]])
* **IIR/FIR combs**: long (\tau[n]) and feedback (a_j[n])

---

# 2) A practical, unified Python toolkit (NumPy + Numba)

Below is a compact implementation that gives you:

* **LTVStateSpace**: run any precomputed ({A,B,C,D}) sequences.
* **UnifiedBlock**: the “one equation to rule them all” (FIR+IIR+mod+frac-delay).
* **Small builders** to parameterize common cases (AM/RM, flanger, IIR/FIR comb, biquad via BLT).

All are **procedural classes**, vectorized where it matters, with Numba-jitted inner loops.

```python
# unified_dsp.py
from __future__ import annotations
import numpy as np
from numba import njit

# ---------------------------
# Helpers
# ---------------------------

def _as_array(x, N, dtype=np.float64):
    if np.isscalar(x):
        return np.full(N, float(x), dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    if arr.shape[0] != N:
        raise ValueError(f"Expected length {N}, got {arr.shape[0]}")
    return arr

@njit(cache=True, fastmath=True)
def _lininterp(buf, wptr, delay):
    """Fractional read from circular buffer (linear interpolation)."""
    N = buf.shape[0]
    read_pos = wptr - delay
    while read_pos < 0.0:
        read_pos += N
    i0 = int(read_pos) % N
    frac = read_pos - int(read_pos)
    i1 = (i0 + 1) % N
    return (1.0 - frac) * buf[i0] + frac * buf[i1]

@njit(cache=True, fastmath=True)
def _write_advance(buf, wptr, val):
    N = buf.shape[0]
    buf[wptr] = val
    wptr += 1
    if wptr >= N: wptr = 0
    return wptr

# ---------------------------
# 2.1 LTV State-Space runner
# x_{n+1}=A[n]x_n + B[n]u[n]; y[n]=C[n]x_n + D[n]u[n]
# ---------------------------

class LTVStateSpace:
    """
    Run a sequence of small state-space systems with time-varying A,B,C,D.
    Shapes:
      A_seq: (N, nx, nx)
      B_seq: (N, nx, 1)   (single-input)
      C_seq: (N, 1,  nx)  (single-output)
      D_seq: (N, 1,  1)
      u    : (N,)
    """
    def __init__(self, x0: np.ndarray):
        self.x = np.asarray(x0, dtype=np.float64)

    def reset(self, x0=None):
        if x0 is None:
            self.x[:] = 0.0
        else:
            self.x = np.asarray(x0, dtype=np.float64)

    def process(self, A_seq, B_seq, C_seq, D_seq, u):
        A_seq = np.asarray(A_seq, dtype=np.float64)
        B_seq = np.asarray(B_seq, dtype=np.float64)
        C_seq = np.asarray(C_seq, dtype=np.float64)
        D_seq = np.asarray(D_seq, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        y, x_end = _ltv_run(A_seq, B_seq, C_seq, D_seq, u, self.x)
        self.x = x_end
        return y

@njit(cache=True, fastmath=True)
def _ltv_run(A_seq, B_seq, C_seq, D_seq, u, x0):
    N, nx, _ = A_seq.shape
    x = x0.copy()
    y = np.empty(N, dtype=np.float64)
    for n in range(N):
        # y = C x + D u
        acc = 0.0
        for i in range(nx):
            acc += C_seq[n, 0, i] * x[i]
        y[n] = acc + D_seq[n, 0, 0] * u[n]
        # x_next = A x + B u
        x_next = np.zeros_like(x)
        for i in range(nx):
            s = 0.0
            for j in range(nx):
                s += A_seq[n, i, j] * x[j]
            s += B_seq[n, i, 0] * u[n]
            x_next[i] = s
        x = x_next
    return y, x

# ---------------------------
# 2.2 UnifiedBlock: y = m*x + Σ b_i x[n-i] + d*x[n-τ] + Σ a_j y[n-j]
# Supports time-varying taps/weights and fractional delay.
# ---------------------------

@njit(cache=True, fastmath=True)
def _unified_block(x, m, b, a, d, tau_samps, xbuf, xwptr, yhist):
    """
    x: (N,)
    m: (N,) multiplier (AM/RM-like)
    b: (N,P) feedforward taps (x history). If P==0, ignored.
    a: (N,Q) feedback taps (y history). If Q==0, ignored. Note sign convention: PLUS Σ a_j y[n-j]
    d: (N,) weight for fractional delay term
    tau_samps: (N,) fractional delay in samples (read from x-buffer)
    xbuf: circular buffer for input, sized >= max(tau)+2
    yhist: last Q values of y (most recent first): [y[n-1], y[n-2], ...]
    """
    N = x.shape[0]
    P = b.shape[1] if b.ndim == 2 else 0
    Q = a.shape[1] if a.ndim == 2 else 0

    y = np.empty_like(x)
    # past x ringbuffer already contains previous inputs; we will write current x each step
    for n in range(N):
        # base term: m[n]*x[n]
        acc = m[n] * x[n]

        # fractional delay from input buffer
        acc += d[n] * _lininterp(xbuf, xwptr, tau_samps[n])

        # FIR over past x (direct-form; x[n-i] needs ringbuffer reads)
        # We read i=1..P using integer offsets (fast path). If you want fractional taps, fold them into 'd'.
        for i in range(1, P + 1):
            # integer read: x[n-i]
            acc += b[n, i - 1] * xbuf[(xwptr - i) % xbuf.shape[0]]

        # IIR feedback over past y (y[n-j] from yhist)
        for j in range(1, Q + 1):
            acc += a[n, j - 1] * yhist[j - 1]

        # output
        y[n] = acc

        # shift y history (simple small-Q shift)
        if Q > 0:
            for j in range(Q - 1, 0, -1):
                yhist[j] = yhist[j - 1]
            yhist[0] = y[n]

        # write current input to buffer and advance
        xwptr = _write_advance(xbuf, xwptr, x[n])

    return y, xwptr, yhist

class UnifiedBlock:
    """
    Unified time-varying block:
      y[n] = m[n]*x[n] + sum_i b_i[n]*x[n-i] + d[n]*x[n-τ[n]] + sum_j a_j[n]*y[n-j]

    Notes:
      - 'a' uses PLUS sign convention; supply negative values for classic DF-II (-a_j).
      - Provide per-sample arrays for time-varying behavior, or scalars broadcasted to N.
      - τ is fractional delay on INPUT path (good for flanger/chorus/vibrato). For feedback combs,
        use nonzero 'a' and set τ via history or use a dedicated delay loop.
    """
    def __init__(self, fs, max_delay_s=2.0, Q=2, P=2):
        self.fs = float(fs)
        self.max_len = int(np.ceil(max_delay_s * self.fs)) + 3
        self.xbuf = np.zeros(self.max_len, dtype=np.float64)
        self.xwptr = 0
        self.yhist = np.zeros(Q, dtype=np.float64)
        self._Q = Q
        self._P = P

    def reset(self):
        self.xbuf[:] = 0.0
        self.xwptr = 0
        self.yhist[:] = 0.0

    def process(self, x, m=1.0, b=None, a=None, d=0.0, tau_s=0.0):
        x = np.asarray(x, dtype=np.float64)
        N = x.shape[0]

        m = _as_array(m, N)
        d = _as_array(d, N)
        tau_samps = _as_array(np.asarray(tau_s, np.float64) * self.fs, N)

        if b is None:
            b_arr = np.zeros((N, 0), dtype=np.float64)
        else:
            b = np.asarray(b, dtype=np.float64)
            if b.ndim == 1:  # (P,) -> broadcast to (N,P)
                b_arr = np.tile(b[None, :], (N, 1))
            else:
                if b.shape[0] != N:
                    raise ValueError("b must be (N,P) or (P,)")
                b_arr = b

        if a is None:
            a_arr = np.zeros((N, 0), dtype=np.float64)
        else:
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 1:
                a_arr = np.tile(a[None, :], (N, 1))
            else:
                if a.shape[0] != N:
                    raise ValueError("a must be (N,Q) or (Q,)")
                a_arr = a

        y, self.xwptr, self.yhist = _unified_block(
            x, m, b_arr, a_arr, d, tau_samps, self.xbuf, self.xwptr, self.yhist
        )
        return y

# ---------------------------
# 2.3 Small builders / recipes
# ---------------------------

def biquad_coeffs(fs, fc, Q=0.707, ftype="lp"):
    """
    Bilinear-transform RBJ-style biquad (TDF-II sign: y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2).
    We return (b, a_df2) ready for UnifiedBlock by flipping signs for 'a' to PLUS convention.
    """
    fc = float(fc); Q = float(Q)
    w0 = 2*np.pi*fc/fs
    alpha = np.sin(w0)/(2*Q)
    cosw = np.cos(w0)

    if ftype == "lp":
        b0 = (1 - cosw)/2; b1 = 1 - cosw; b2 = (1 - cosw)/2
        a0 = 1 + alpha; a1 = -2*cosw; a2 = 1 - alpha
    elif ftype == "hp":
        b0 = (1 + cosw)/2; b1 = -(1 + cosw); b2 = (1 + cosw)/2
        a0 = 1 + alpha; a1 = -2*cosw; a2 = 1 - alpha
    elif ftype == "bp":
        b0 =  alpha; b1 = 0.0; b2 = -alpha
        a0 = 1 + alpha; a1 = -2*cosw; a2 = 1 - alpha
    elif ftype == "br":
        b0 =  1; b1 = -2*cosw; b2 = 1
        a0 = 1 + alpha; a1 = -2*cosw; a2 = 1 - alpha
    else:
        raise ValueError("ftype must be lp/hp/bp/br")

    # Normalize
    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0

    # For UnifiedBlock we want PLUS Σ a_j y[n-j] => supply [-a1, -a2]
    return np.array([b0, b1, b2]), np.array([-a1, -a2])

def am_params(x, depth=1.0, lfo=None):
    """AM/Tremolo: m[n] = 1 + depth * lfo[n]."""
    N = x.shape[0]
    if lfo is None:
        lfo = np.zeros(N, dtype=np.float64)
    return 1.0 + float(depth) * np.asarray(lfo, dtype=np.float64)

def rm_apply(x, carrier):
    """Ring mod: simple multiply (or fold carrier into 'm')."""
    return x * np.asarray(carrier, dtype=np.float64)

def flanger_params(fs, base_ms=5.0, depth_ms=3.0, lfo=None, mix=0.5):
    """Flanger via fractional delay on input with weight d and direct path via m."""
    base_s = base_ms * 1e-3; depth_s = depth_ms * 1e-3
    def tau_series(N):
        t = np.linspace(0, 1, N, endpoint=False)
        if lfo is None:
            l = np.sin(2*np.pi*0.5*t)   # default 0.5 Hz
        else:
            l = np.asarray(lfo, np.float64)
        return base_s + 0.5*depth_s*(l + 1.0)
    return tau_series, mix  # use in UnifiedBlock with m=(1-mix), d=mix

def comb_fir_params(delay_s, g=0.7):
    """FIR comb: y = x + g x[n-M] -> set m=1, d=g, tau=const."""
    return dict(m=1.0, d=float(g), tau_s=float(delay_s))

def comb_iir_params(delay_s, g=0.7, c=0.5):
    """
    IIR comb in UnifiedBlock (feedforward form with feedback via a[0]).
    Approximate as: y[n] = c x[n] + g y[n-M]
    For long M, better implement dedicated loop; here we emulate short-M cases using 'a' and y-history.
    """
    # This helper is limited to short delays that fit in y-history. For long delays, build a dedicated delay loop.
    raise NotImplementedError("For long delay feedback, prefer a dedicated delay-line comb (like in your previous toolkit).")

# ---------------------------
# 2.4 Tiny demos
# ---------------------------

if __name__ == "__main__":
    fs = 48000
    N = fs // 2
    t = np.arange(N)/fs
    x = 0.3*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*330*t)

    # --- A) Static biquad lowpass (UnifiedBlock)
    b, a = biquad_coeffs(fs, fc=1000, Q=0.707, ftype="lp")
    ub = UnifiedBlock(fs, max_delay_s=0.05, Q=2, P=2)
    y_lp = ub.process(x, m=0.0, b=b, a=a, d=0.0, tau_s=0.0)

    # --- B) Tremolo (AM) at 5 Hz
    lfo = np.sin(2*np.pi*5*t)
    m = am_params(x, depth=0.8, lfo=lfo)
    ub.reset()
    y_trem = ub.process(x, m=m)  # no taps, no delay

    # --- C) Flanger: base 5 ms, depth 3 ms, mix 0.5
    tau_fun, mix = flanger_params(fs, base_ms=5.0, depth_ms=3.0, lfo=np.sin(2*np.pi*0.5*t), mix=0.5)
    tau = tau_fun(N)
    ub.reset()
    y_flang = ub.process(x, m=(1.0-mix), d=mix, tau_s=tau)

    # --- D) LTV State-Space: 2-state SVF-like toy (time-varying)
    # Example: simple integrator chain with sweep in 'F' (illustrative)
    F = 2*np.sin(np.pi*np.linspace(200, 4000, N)/fs)
    R = 1.0/(2*1.0)
    A_seq = np.zeros((N, 2, 2)); B_seq = np.zeros((N, 2, 1))
    C_seq = np.zeros((N, 1, 2)); D_seq = np.zeros((N, 1, 1))
    for n in range(N):
        g = F[n]
        # very simple 2-int chain (not a full ZDF-SVF; illustrative for LTV):
        # x1[n+1] = x1[n] + g*(u[n] - 2R*x1[n] - x2[n])
        # x2[n+1] = x2[n] + g*x1[n]
        A_seq[n] = np.array([[1.0 - 2*R*g, -g],
                             [g,            1.0]])
        B_seq[n] = np.array([[g],[0.0]])
        C_seq[n] = np.array([[0.0, 1.0]])   # output = x2 (LP)
        D_seq[n] = np.array([[0.0]])
    ltv = LTVStateSpace(x0=np.zeros(2))
    y_svf_like = ltv.process(A_seq, B_seq, C_seq, D_seq, x)

    print("ok:", y_lp.shape, y_trem.shape, y_flang.shape, y_svf_like.shape)
```

---

## How to use this “one block” for classic effects

* **Biquad filter**: compute `b,a = biquad_coeffs(...)`, then `ub.process(x, m=0.0, b=b, a=a)`.
* **Param-swept filter**: recompute `(b,a)` per sample (or per chunk) and pass as `(N,2)` arrays.
* **AM/Tremolo**: `m = 1 + depth*lfo`; `ub.process(x, m=m)`.
* **Ring mod**: just multiply: `y = rm_apply(x, carrier)`.
* **Flanger/Chorus/Vibrato**: set `m=(1-mix)`, `d=mix`, and `tau_s` as your fractional delay series.
* **FIR comb**: `ub.process(x, m=1.0, d=g, tau_s=const_delay_s)`.

> For **long IIR combs (feedback with large M)** or **reverbs**, keep using the dedicated delay-line feedback loops from your previous toolkit (they’re optimal and clearer). The `LTVStateSpace` is great for compact state models (SVF, compressors, envelope trackers), and `UnifiedBlock` is your Swiss-army knife for FIR/IIR/frac-delay/AM in one pass.

If you want, I can add:

* a ZDF-SVF **builder** that emits per-sample (A,B,C,D) (exact TPT form),
* a **Hilbert** helper to build SSB in this same framework,
* or a **feedback-delay network (FDN)** wrapper that parameterizes into `UnifiedBlock` instances.

---

heck yes — here’s a **ZDF-SVF (zero-delay feedback state-variable filter) builder** that plugs straight into the `LTVStateSpace` runner we set up. It emits per-sample (A,B,C,D) matrices for **LP/BP/HP** outputs using the classic TPT/ZDF form (stable, sweep-friendly).

Drop this into your existing `unified_dsp.py` (or alongside it).

```python
# ---------------------------
# ZDF-SVF (TPT) builder for LTVStateSpace
# State vector is s = [z1, z2] where z1,z2 are the two integrator states.
# Equations (per-sample parameters):
#   g = 2*sin(pi*fc/fs)   (TPT mapping; robust for sweeps)
#   R = 1/(2*Q)
#   D = 1 + R*g + g*g
#
#   hp = (x - (R+1)*z1 - z2) / D
#   bp = g*hp + z1
#   lp = g*bp + z2
#
#   z1' = 2*g*hp + z1
#   z2' = 2*g*bp + z2
#
# From these, we derive linear forms:
#   hp = d_hp*x + c1_hp*z1 + c2_hp*z2
#   bp = d_bp*x + c1_bp*z1 + c2_bp*z2
#   lp = d_lp*x + c1_lp*z1 + c2_lp*z2
# and state update:
#   [z1'; z2'] = A * [z1; z2] + B * x
# ---------------------------

import numpy as np

def _zdf_svf_coeffs_per_sample(fs, fc, Q):
    fc = np.asarray(fc, dtype=np.float64)
    Q  = np.asarray(Q,  dtype=np.float64)
    if fc.ndim == 0: fc = fc[None]
    if Q.ndim  == 0: Q  = Q[None]
    if fc.shape[0] != Q.shape[0]:
        raise ValueError("fc and Q must have same length (or both scalars).")
    N = fc.shape[0]

    g = 2.0 * np.sin(np.pi * fc / float(fs))   # TPT freq map
    R = 1.0 / (2.0 * (Q + 1e-12))              # guard small Q
    D = 1.0 + R*g + g*g

    # hp = (1/D)*x + (-(R+1)/D)*z1 + (-1/D)*z2
    d_hp  = 1.0 / D
    c1_hp = -(R + 1.0) / D
    c2_hp = -1.0 / D

    # bp = g*hp + z1
    #    = (g/D)*x + [1 - g*(R+1)/D]*z1 + [-g/D]*z2
    d_bp  = g / D
    c1_bp = 1.0 - g*(R + 1.0) / D
    c2_bp = -g / D

    # lp = g*bp + z2
    #    = (g^2/D)*x + [g*(1 - g*(R+1)/D)]*z1 + [1 - g^2/D]*z2
    d_lp  = (g*g) / D
    c1_lp = g * (1.0 - g*(R + 1.0) / D)
    c2_lp = 1.0 - (g*g) / D

    # State updates:
    # z1' = 2g*hp + z1  = (2g/D)*x + [1 - 2g(R+1)/D]*z1 + [-2g/D]*z2
    # z2' = 2g*bp + z2  = (2g^2/D)*x + [2g*(1 - g(R+1)/D)]*z1 + [1 - 2g^2/D]*z2
    a11 = 1.0 - 2.0*g*(R + 1.0) / D
    a12 = -2.0*g / D
    a21 = 2.0*g * (1.0 - g*(R + 1.0) / D)
    a22 = 1.0 - 2.0*(g*g) / D

    b1 = 2.0*g / D
    b2 = 2.0*(g*g) / D

    # Pack sequences
    A_seq = np.zeros((N, 2, 2), dtype=np.float64)
    B_seq = np.zeros((N, 2, 1), dtype=np.float64)

    A_seq[:, 0, 0] = a11
    A_seq[:, 0, 1] = a12
    A_seq[:, 1, 0] = a21
    A_seq[:, 1, 1] = a22

    B_seq[:, 0, 0] = b1
    B_seq[:, 1, 0] = b2

    # Output coefficient triplets for each mode
    outs = {
        "hp": (c1_hp, c2_hp, d_hp),
        "bp": (c1_bp, c2_bp, d_bp),
        "lp": (c1_lp, c2_lp, d_lp),
    }
    return A_seq, B_seq, outs

def build_zdf_svf_sequences(fs, fc, Q=0.707, mode="lp"):
    """
    Return (A_seq, B_seq, C_seq, D_seq) for an LTV ZDF-SVF with the chosen mode.
      fs : sample rate
      fc : cutoff/center freq (scalar or per-sample array)
      Q  : quality factor (scalar or per-sample array)
      mode: 'lp' | 'bp' | 'hp'
    Shapes:
      A_seq: (N,2,2), B_seq: (N,2,1), C_seq: (N,1,2), D_seq: (N,1,1)
    """
    if mode not in ("lp", "bp", "hp"):
        raise ValueError("mode must be 'lp'|'bp'|'hp'")

    # Broadcast fc,Q to arrays of length N
    fc_arr = np.atleast_1d(np.asarray(fc, dtype=np.float64))
    Q_arr  = np.atleast_1d(np.asarray(Q,  dtype=np.float64))
    if fc_arr.size == 1 and Q_arr.size > 1:
        fc_arr = np.full_like(Q_arr, float(fc_arr))
    elif Q_arr.size == 1 and fc_arr.size > 1:
        Q_arr = np.full_like(fc_arr, float(Q_arr))
    elif fc_arr.size != Q_arr.size:
        raise ValueError("fc and Q must be same length or scalars.")

    A_seq, B_seq, outs = _zdf_svf_coeffs_per_sample(fs, fc_arr, Q_arr)
    c1, c2, d = outs[mode]

    N = fc_arr.shape[0]
    C_seq = np.zeros((N, 1, 2), dtype=np.float64)
    D_seq = np.zeros((N, 1, 1), dtype=np.float64)

    C_seq[:, 0, 0] = c1
    C_seq[:, 0, 1] = c2
    D_seq[:, 0, 0] = d

    return A_seq, B_seq, C_seq, D_seq

class ZDFSVF:
    """
    Convenience wrapper: ZDF-SVF driven by LTVStateSpace.
    Choose output mode ('lp','bp','hp'). Supports per-sample fc/Q sweeps.
    """
    def __init__(self, fs, mode="lp"):
        from unified_dsp import LTVStateSpace  # or adjust import if placed together
        self.fs = float(fs)
        self.mode = mode
        self.ltv = LTVStateSpace(x0=np.zeros(2, dtype=np.float64))

    def reset(self):
        self.ltv.reset(x0=np.zeros(2, dtype=np.float64))

    def process(self, x, fc, Q=0.707):
        x = np.asarray(x, dtype=np.float64)
        N = x.shape[0]
        # Prepare per-sample matrices
        A, B, C, D = build_zdf_svf_sequences(self.fs, fc, Q, mode=self.mode)
        # Run as LTV SS
        y = self.ltv.process(A, B, C, D, x)
        return y
```

## quick usage

```python
if __name__ == "__main__":
    import numpy as np
    fs = 48000
    N  = fs
    t  = np.arange(N)/fs
    x  = 0.3*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*330*t)

    # Sweep 200 Hz -> 6 kHz bandpass with Q=1
    fc_sweep = np.linspace(200, 6000, N)
    from unified_dsp import LTVStateSpace  # if separated, ensure imports

    # Build sequences for BP output
    A, B, C, D = build_zdf_svf_sequences(fs, fc_sweep, Q=1.0, mode="bp")
    ltv = LTVStateSpace(x0=np.zeros(2))
    y_bp = ltv.process(A, B, C, D, x)

    # Or use the convenience wrapper:
    zdf = ZDFSVF(fs, mode="lp")
    y_lp = zdf.process(x, fc=fc_sweep, Q=0.8)

    print(y_bp.shape, y_lp.shape)
```

### notes

* This is the **TPT/ZDF** SVF (two trapezoidal integrators with algebraic loop solved analytically). It’s stable for wide sweeps.
* Frequency map uses (F = 2\sin(\pi f_c/f_s)), which plays nicely near Nyquist and works great for realtime control.
* You can hot-swap modes by rebuilding `C_seq, D_seq` (no need to change `A_seq, B_seq`).
* For stereo or multiband, instantiate multiple `ZDFSVF` objects (or vectorize the state).

If you want a **drive/resonance-safe loudness compensation** option (the √ζ + HF roll-off trick) baked into the builder, I can add an optional `norm='L2'` flag that scales `D_seq` / `C_seq` accordingly.

# Reference

https://dafx.de/paper-archive/1998/DUT52.PS.pdf
