"""
Statistical analysis utilities with minimal dependencies.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from scipy import stats
import logging

from ..base.exceptions import StatisticsError


class BasicStatistics:
    """Basic statistical operations."""
    
    @staticmethod
    def mean_std(data: np.ndarray, axis: Optional[int] = None, 
                 robust: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and standard deviation."""
        if robust:
            return RobustStatistics.median_mad(data, axis=axis)
        else:
            return np.mean(data, axis=axis), np.std(data, axis=axis, ddof=1)
    
    @staticmethod
    def percentiles(data: np.ndarray, percentiles: np.ndarray = None,
                   axis: Optional[int] = None) -> np.ndarray:
        """Calculate percentiles."""
        if percentiles is None:
            percentiles = np.array([16, 50, 84])  # 1-sigma equivalent
        
        return np.percentile(data, percentiles, axis=axis)
    
    @staticmethod
    def moments(data: np.ndarray, axis: Optional[int] = None) -> Dict[str, float]:
        """Calculate first four statistical moments."""
        mean = np.mean(data, axis=axis)
        var = np.var(data, axis=axis, ddof=1)
        skew = stats.skew(data, axis=axis)
        kurt = stats.kurtosis(data, axis=axis)
        
        return {
            "mean": mean,
            "variance": var,
            "skewness": skew,
            "kurtosis": kurt
        }
    
    @staticmethod
    def histogram_stats(data: np.ndarray, bins: Union[int, np.ndarray] = 50,
                       density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate histogram with proper normalization."""
        counts, bin_edges = np.histogram(data, bins=bins, density=density)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts


class RobustStatistics:
    """Robust statistical estimators."""
    
    @staticmethod
    def median_mad(data: np.ndarray, axis: Optional[int] = None,
                   scale_factor: float = 1.4826) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate median and median absolute deviation."""
        median = np.median(data, axis=axis)
        
        if axis is None:
            mad = np.median(np.abs(data - median))
        else:
            mad = np.median(np.abs(data - np.expand_dims(median, axis)), axis=axis)
        
        # Scale MAD to match standard deviation for normal distribution
        mad_scaled = mad * scale_factor
        
        return median, mad_scaled
    
    @staticmethod
    def trimmed_mean(data: np.ndarray, trim_fraction: float = 0.1,
                    axis: Optional[int] = None) -> np.ndarray:
        """Calculate trimmed mean."""
        if not 0 <= trim_fraction < 0.5:
            raise StatisticsError("trim_fraction must be in [0, 0.5)")
        
        return stats.trim_mean(data, trim_fraction, axis=axis)
    
    @staticmethod
    def winsorized_stats(data: np.ndarray, limits: Tuple[float, float] = (0.05, 0.05),
                        axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate winsorized mean and standard deviation."""
        winsorized = stats.mstats.winsorize(data, limits=limits, axis=axis)
        mean = np.mean(winsorized, axis=axis)
        std = np.std(winsorized, axis=axis, ddof=1)
        return mean, std
    
    @staticmethod
    def biweight_location_scale(data: np.ndarray, c: float = 9.0,
                               axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biweight location and scale estimators."""
        # Simplified implementation - full biweight requires iterative solution
        median = np.median(data, axis=axis)
        mad = np.median(np.abs(data - np.expand_dims(median, axis) if axis is not None else median), axis=axis)
        
        # Use MAD as initial scale estimate
        scale = mad * 1.4826
        
        return median, scale


class CorrelationAnalysis:
    """Correlation and dependence analysis."""
    
    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray,
                           confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate Pearson correlation with confidence interval."""
        if x.shape != y.shape:
            raise StatisticsError("Input arrays must have same shape")
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 3:
            raise StatisticsError("Need at least 3 valid data points")
        
        corr, p_value = stats.pearsonr(x_clean, y_clean)
        
        # Calculate confidence interval using Fisher transform
        n = len(x_clean)
        z = np.arctanh(corr)
        se = 1 / np.sqrt(n - 3)
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        z_low = z - z_crit * se
        z_high = z + z_crit * se
        
        ci_low = np.tanh(z_low)
        ci_high = np.tanh(z_high)
        
        return {
            "correlation": corr,
            "p_value": p_value,
            "confidence_interval": (ci_low, ci_high),
            "n_samples": n
        }
    
    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate Spearman rank correlation."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 3:
            raise StatisticsError("Need at least 3 valid data points")
        
        corr, p_value = stats.spearmanr(x_clean, y_clean)
        
        return {
            "correlation": corr,
            "p_value": p_value,
            "n_samples": len(x_clean)
        }
    
    @staticmethod
    def cross_correlation(x: np.ndarray, y: np.ndarray,
                         maxlags: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cross-correlation function."""
        if maxlags is None:
            maxlags = min(len(x), len(y)) // 4
        
        # Normalize inputs
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
        
        # Calculate cross-correlation using FFT
        correlation = np.correlate(x_norm, y_norm, mode='full')
        correlation = correlation / len(x)
        
        # Extract relevant lags
        center = len(correlation) // 2
        lags = np.arange(-maxlags, maxlags + 1)
        correlation_subset = correlation[center - maxlags:center + maxlags + 1]
        
        return lags, correlation_subset


class CovarianceEstimator:
    """Covariance matrix estimation and analysis."""
    
    @staticmethod
    def sample_covariance(data: np.ndarray, rowvar: bool = True,
                         bias: bool = False) -> np.ndarray:
        """Calculate sample covariance matrix."""
        return np.cov(data, rowvar=rowvar, bias=bias)
    
    @staticmethod
    def shrinkage_covariance(data: np.ndarray, shrinkage: Optional[float] = None,
                           rowvar: bool = True) -> Tuple[np.ndarray, float]:
        """Calculate shrinkage covariance estimator (Ledoit-Wolf)."""
        if not rowvar:
            data = data.T
        
        n_features, n_samples = data.shape
        
        # Center the data
        data_centered = data - np.mean(data, axis=1, keepdims=True)
        
        # Sample covariance
        sample_cov = np.dot(data_centered, data_centered.T) / (n_samples - 1)
        
        if shrinkage is None:
            # Ledoit-Wolf optimal shrinkage
            trace_sample = np.trace(sample_cov)
            trace_sample_sq = np.trace(np.dot(sample_cov, sample_cov))
            
            # Target matrix (diagonal with average variance)
            target = np.eye(n_features) * trace_sample / n_features
            
            # Optimal shrinkage intensity
            numerator = trace_sample_sq - trace_sample**2 / n_features
            denominator = (n_samples + 1) * (trace_sample_sq - trace_sample**2 / n_features)
            
            if denominator == 0:
                shrinkage = 0
            else:
                shrinkage = min(1, numerator / denominator)
        
        # Shrinkage estimator
        target = np.eye(n_features) * np.trace(sample_cov) / n_features
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        return shrunk_cov, shrinkage
    
    @staticmethod
    def condition_number(cov_matrix: np.ndarray) -> float:
        """Calculate condition number of covariance matrix."""
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = eigenvals[eigenvals > 0]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return np.inf
        
        return np.max(eigenvals) / np.min(eigenvals)
    
    @staticmethod
    def effective_sample_size(cov_matrix: np.ndarray, n_samples: int) -> float:
        """Estimate effective sample size accounting for correlations."""
        # Based on the trace of the precision matrix
        try:
            precision = np.linalg.inv(cov_matrix)
            trace_precision = np.trace(precision)
            n_features = cov_matrix.shape[0]
            
            # Effective sample size correction
            eff_n = n_samples * n_features / trace_precision
            return max(1, eff_n)
        except np.linalg.LinAlgError:
            logging.warning("Singular covariance matrix, returning nominal sample size")
            return n_samples


class PowerSpectrumEstimator:
    """Power spectrum estimation utilities."""
    
    @staticmethod
    def periodogram(data: np.ndarray, sampling_rate: float = 1.0,
                   window: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate periodogram power spectrum estimate."""
        if window is not None:
            if window == 'hanning':
                window_func = np.hanning(len(data))
            elif window == 'hamming':
                window_func = np.hamming(len(data))
            elif window == 'blackman':
                window_func = np.blackman(len(data))
            else:
                window_func = np.ones(len(data))
            
            data_windowed = data * window_func
            # Correct for window power
            window_norm = np.sum(window_func**2)
        else:
            data_windowed = data
            window_norm = len(data)
        
        # Calculate FFT
        fft_data = np.fft.fft(data_windowed)
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        
        # Power spectrum (one-sided)
        power = np.abs(fft_data)**2 / (sampling_rate * window_norm)
        
        # Return positive frequencies only
        n_pos = len(data) // 2
        return freqs[:n_pos], power[:n_pos]
    
    @staticmethod
    def welch_method(data: np.ndarray, nperseg: Optional[int] = None,
                    overlap: float = 0.5, window: str = 'hanning',
                    sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Welch's method for power spectrum estimation."""
        from scipy import signal
        
        if nperseg is None:
            nperseg = len(data) // 8
        
        freqs, power = signal.welch(data, fs=sampling_rate, window=window,
                                  nperseg=nperseg, noverlap=int(nperseg * overlap))
        
        return freqs, power