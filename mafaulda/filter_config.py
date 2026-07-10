import numpy as np
from scipy.signal import butter, iirfilter, lfilter, filtfilt

class TachometerProcessor:
    """
    Tachometer signal processor for calculating instantaneous RPM.
    The code is simplified adhering strictly to the Single Responsibility Principle (SRP).
    """
    def __init__(self,
                 Fs: int = int(5e4),
                 pulses_per_rev: int = 1,
                 min_edge_ratio: float = 0.7,
                 threshold_ratio: float = 0.7,
                 filter_cutoff: float = 400.0,
                 filter_order: int = 4):
        """
        Initializes the Tachometer Processing pipeline.

        Args:
            Fs (int): Sampling frequency in Hz. Defaults to 50000.
            pulses_per_rev (int): Number of signal pulses per single shaft revolution. Defaults to 1.
            min_edge_ratio (float): Ratio used to filter out noise-induced false edges. Defaults to 0.7.
            threshold_ratio (float): Amplitude percentage to trigger an edge. Defaults to 0.7.
            filter_cutoff (float): Cutoff frequency for the low-pass noise filter. Defaults to 400.0.
            filter_order (int): Order of the Butterworth low-pass filter. Defaults to 4.
        """
        self.Fs = Fs
        self.pulses_per_rev = pulses_per_rev
        self.min_edge_ratio = min_edge_ratio
        self.threshold_ratio = threshold_ratio
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.b, self.a = butter(self.filter_order, self.filter_cutoff / (self.Fs * 0.5), btype="low")

    def _apply_low_pass(self, x: np.ndarray) -> np.ndarray:
        """Applying a low-pass filter to remove high-frequency noise"""
        return filtfilt(self.b, self.a, x)

    def _robust_scale(self, x: np.ndarray) -> np.ndarray:
        """Normalizing the signal between 0 and 1 based on percentiles"""
        q1 = np.percentile(x, 1)
        q99 = np.percentile(x, 99)
        scale = q99 - q1
        if scale <= 0:
            return x - q1
        return (x - q1) / scale

    def _detect_rising_edges(self, x: np.ndarray) -> np.ndarray:
        """Finding upward threshold crossing points (rising edges)"""
        rising_mask = (x[:-1] < self.threshold_ratio) & (x[1:] >= self.threshold_ratio) & (x[1:] > x[:-1])
        return np.where(rising_mask)[0] + 1

    def _filter_close_edges(self, crosses: np.ndarray) -> np.ndarray:
        """Removing edges that are too close to each other due to noise"""
        median_diff = np.median(np.diff(crosses))
        min_sep = max(1, int(round(self.min_edge_ratio * median_diff)))

        valid = [crosses[-1]]
        for i in range(len(crosses) - 2, -1, -1):
            if valid[-1] - crosses[i] >= min_sep:
                valid.append(crosses[i])
        valid.reverse()
        return np.array(valid, dtype=int)

    def _compute_rpm(self, edge_idx: np.ndarray, total_length: int) -> np.ndarray:
        """Calculating the instantaneous RPM vector based on edge distances"""
        rpm_vec = np.zeros(total_length, dtype=float)
        
        # Calculating RPM for the first interval
        first_delta = max(1, edge_idx[1] - edge_idx[0])
        rpm_first = 60 * (self.Fs / first_delta) / self.pulses_per_rev
        rpm_vec[:edge_idx[0]] = rpm_first

        # Calculating RPM for the middle intervals
        for i in range(1, len(edge_idx)):
            delta = max(1, edge_idx[i] - edge_idx[i - 1])
            rpm_i = 60 * (self.Fs / delta) / self.pulses_per_rev
            rpm_vec[edge_idx[i - 1]:edge_idx[i]] = rpm_i

        # Generalizing the last RPM to the end of the signal
        rpm_vec[edge_idx[-1]:] = rpm_i
        
        return rpm_vec

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The main entry point of the class that executes functions in order"""
        x_filt = self._apply_low_pass(x)
        x_norm = self._robust_scale(x_filt)
        
        crosses = self._detect_rising_edges(x_norm)
        if len(crosses) < 2:
            return np.zeros(len(x))
            
        edge_idx = self._filter_close_edges(crosses)
        if len(edge_idx) < 2:
            return np.zeros(len(x))
            
        return self._compute_rpm(edge_idx, len(x))


class AccelerometerFilter:
    """
    Causal low-pass IIR filter for vibration signals.
    """
    def __init__(self,
                 fs: float = 50000.0,
                 cutoff: float = 6220.0,
                 iir_kind: str = "ellip",
                 iir_order: int = 15,
                 rp: float = 0.5,
                 rs: float = 60.0,):
        """
        Initializes the causal IIR filter configuration.

        Args:
            fs (float): Sampling frequency in Hz. Defaults to 50000.0.
            cutoff (float): Cutoff frequency in Hz. Defaults to 6220.0.
            iir_kind (str): Filter type (e.g., 'ellip', 'butter'). Defaults to 'ellip'.
            iir_order (int): Filter order. Defaults to 15.
            rp (float): Maximum ripple allowed below unity gain in the passband. Defaults to 0.5.
            rs (float): Minimum attenuation required in the stopband. Defaults to 60.0.
        """
        
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist

        # Designing the IIR filter
        self.b, self.a = iirfilter(iir_order, normalized_cutoff, rp=rp, rs=rs, 
                                   btype='low', ftype=iir_kind, output='ba')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applying the causal filter using lfilter"""
        return lfilter(self.b, self.a, x)