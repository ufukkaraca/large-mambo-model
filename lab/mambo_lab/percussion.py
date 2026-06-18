"""Percussion path (PAPER §4.5, PAPER §4.5): onset + per-user few-shot
kick/snare/hat classification.

Spectral-flux onset detection + a per-user few-shot classifier over
onset-synchronized timbre features (the literature — Ramires LVT, Delgado 2022,
Dubler — converges on ~5-12 calibration examples per sound as what makes this
work; generic models generalize poorly across people). Classification happens
~50-100 ms after onset (Stowell & Plumbley's delayed-decision result), not at
onset. Output feeds the UIR ``percussion[]`` and an ``insert_drum_pattern`` op.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

CLASSES = ("kick", "snare", "hat")
N_FFT = 512
HOP = 128
WIN_S = 0.07  # delayed-decision window after onset


def detect_onsets(audio: np.ndarray, sr: int, *, delta: float = 0.06) -> np.ndarray:
    """Spectral-flux onset times (s)."""
    audio = np.asarray(audio, dtype=np.float32)
    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP,
                                    units="time", backtrack=True, delta=delta)
    return np.asarray(on)


def hit_features(audio: np.ndarray, sr: int, t0: float, *, win: float = WIN_S) -> np.ndarray:
    """Timbre feature vector over [t0, t0+win] — discriminates kick/snare/hat."""
    a = int(t0 * sr)
    b = min(len(audio), int((t0 + win) * sr))
    seg = np.asarray(audio[a:b], dtype=np.float32)
    if len(seg) < N_FFT:
        seg = np.pad(seg, (0, N_FFT - len(seg)))
    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP)) + 1e-9
    feats = [
        float(np.log(np.sqrt(np.mean(seg**2)) + 1e-9)),
        float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()),
        float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean()),
        float(librosa.feature.spectral_flatness(S=S).mean()),
        float(librosa.feature.zero_crossing_rate(seg, frame_length=N_FFT, hop_length=HOP).mean()),
    ]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    psd = (S**2).mean(axis=1)
    tot = psd.sum() + 1e-9
    feats += [float(psd[freqs < 200].sum() / tot),
              float(psd[(freqs >= 200) & (freqs < 3000)].sum() / tot),
              float(psd[freqs >= 5000].sum() / tot)]
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=5, n_fft=N_FFT, hop_length=HOP,
                                n_mels=24, fmax=sr / 2).mean(axis=1)
    feats += [float(x) for x in mfcc]
    return np.asarray(feats, dtype=np.float64)


@dataclass
class PercussionHit:
    t: float
    cls: str
    confidence: float


class PercussionClassifier:
    """Few-shot kNN over standardized hit features (per-user calibration)."""

    def __init__(self, k: int = 3):
        self.k = k
        self._scaler = None
        self._knn = None

    def fit(self, X: list[np.ndarray], y: list[str]) -> "PercussionClassifier":
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        Xa = np.vstack(X)
        self._scaler = StandardScaler().fit(Xa)
        self._knn = KNeighborsClassifier(n_neighbors=min(self.k, len(y))).fit(
            self._scaler.transform(Xa), y)
        return self

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        xs = self._scaler.transform([x])
        cls = str(self._knn.predict(xs)[0])
        conf = float(self._knn.predict_proba(xs)[0].max())
        return cls, conf


def analyze_percussion(audio: np.ndarray, sr: int, clf: PercussionClassifier) -> list[PercussionHit]:
    audio = np.asarray(audio, dtype=np.float32)
    hits = []
    for t in detect_onsets(audio, sr):
        cls, conf = clf.predict(hit_features(audio, sr, float(t)))
        hits.append(PercussionHit(t=round(float(t), 4), cls=cls, confidence=round(conf, 3)))
    return hits


def to_uir_percussion(hits: list[PercussionHit]) -> list[dict]:
    """PercussionHit list -> the UIR percussion[] array (mambo.utterance.v1)."""
    return [{"t": h.t, "class": h.cls, "confidence": h.confidence} for h in hits]


def train_from_calib(calib: list[tuple[np.ndarray, str]], sr: int, k: int = 3) -> PercussionClassifier:
    """Train from (audio, class) calibration clips — features at the clip's onset."""
    X, y = [], []
    for audio, cls in calib:
        ons = detect_onsets(audio, sr)
        t0 = float(ons[0]) if len(ons) else 0.05
        X.append(hit_features(audio, sr, t0))
        y.append(cls)
    return PercussionClassifier(k=k).fit(X, y)
