"""
Statistical analysis utilities with minimal dependencies.

This module provides comprehensive statistical functions using only numpy
and scipy, avoiding heavy astronomical packages.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any, List
from scipy import stats
import logging

from ..base.exceptions import StatisticsError


class BasicStatistics:
    """Basic statistical operations with numpy."""
    
    @staticmethod
    def mean_std(data: np.ndarray, axis: Optional[int] = None, 
                 robust: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and standard deviation.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        axis : int, optional
            Axis along which to compute statistics
        robust : bool
            If True, use robust estimators (median/MAD)
            
        Returns
        -------
        tuple
            (mean, std) or (median, MAD) if robust=True
        """
        if robust:
            return RobustStatistics.median_mad(data, axis=axis)
        else:
            return np.mean(data, axis=axis), np.std(data, axis=axis, ddof=1)
    
    @staticmethod
    def percentiles(data: np.ndarray, percentiles: Optional[np.ndarray] = None,
                   axis: Optional[int] = None) -> np.ndarray:
        """Calculate percentiles.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        percentiles : np.ndarray, optional
            Percentile values to compute (default: [16, 50, 84])
        axis : int, optional
            Axis along which to compute percentiles
            
        Returns
        -------
        np.ndarray
            Computed percentiles
        """
        if percentiles is None:
            percentiles = np.array([16, 50, 84])  # 1-sigma equivalent
        
        return np.percentile(data, percentiles, axis=axis)
    
    @staticmethod
    def moments(data: np.ndarray, axis: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """Calculate first four statistical moments.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        axis : int, optional
            Axis along which to compute moments
            
        Returns
        -------
        dict
            Dictionary containing mean, variance, skewness, kurtosis
        """
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
                       density: bool = True, range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate histogram with proper normalization.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        bins : int or np.ndarray
            Number of bins or bin edges
        density : bool
            If True, normalize to probability density
        range : tuple, optional
            Range for histogram bins
            
        Returns
        -------
        tuple
            (bin_centers, counts)
        """
        counts, bin_edges = np.histogram(data, bins=bins, density=density, range=range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts
    
    @staticmethod
    def bootstrap_statistic(data: np.ndarray, statistic_func: callable,
                          n_bootstrap: int = 1000, confidence_level: float = 0.95,
                          random_state: Optional[int] = None) -> Dict[str, float]:
        """Bootstrap estimation of statistic uncertainty.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        statistic_func : callable
            Function to compute statistic
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for interval
        random_state : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Bootstrap results with confidence interval
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_data = len(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n_data, replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            "statistic": statistic_func(data),
            "bootstrap_mean": np.mean(bootstrap_stats),
            "bootstrap_std": np.std(bootstrap_stats),
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": confidence_level
        }


class RobustStatistics:
    """Robust statistical estimators."""
    
    @staticmethod
    def median_mad(data: np.ndarray, axis: Optional[int] = None,
                   scale_factor: float = 1.4826) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate median and median absolute deviation.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        axis : int, optional
            Axis along which to compute statistics
        scale_factor : float
            Factor to scale MAD to match std for normal distribution
            
        Returns
        -------
        tuple
            (median, scaled_mad)
        """
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
        """Calculate trimmed mean.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        trim_fraction : float
            Fraction of data to trim from each end
        axis : int, optional
            Axis along which to compute
            
        Returns
        -------
        np.ndarray
            Trimmed mean
        """
        if not 0 <= trim_fraction < 0.5:
            raise StatisticsError("trim_fraction must be in [0, 0.5)")
        
        return stats.trim_mean(data, trim_fraction, axis=axis)
    
    @staticmethod
    def winsorized_stats(data: np.ndarray, limits: Tuple[float, float] = (0.05, 0.05),
                        axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate winsorized mean and standard deviation.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        limits : tuple
            Fraction of data to winsorize at (lower, upper) ends
        axis : int, optional
            Axis along which to compute
            
        Returns
        -------
        tuple
            (winsorized_mean, winsorized_std)
        """
        winsorized = stats.mstats.winsorize(data, limits=limits, axis=axis)
        mean = np.mean(winsorized, axis=axis)
        std = np.std(winsorized, axis=axis, ddof=1)
        return mean, std
    
    @staticmethod
    def huber_location_scale(data: np.ndarray, k: float = 1.345,
                           max_iter: int = 100, tol: float = 1e-6) -> Tuple[float, float]:
        """Calculate Huber location and scale estimators.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (1D)
        k : float
            Tuning parameter for Huber function
        max_iter : int
            Maximum iterations for convergence
        tol : float
            Tolerance for convergence
            
        Returns
        -------
        tuple
            (huber_location, huber_scale)
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Initialize with median and MAD
        mu = np.median(data)
        sigma = np.median(np.abs(data - mu)) * 1.4826
        
        for _ in range(max_iter):
            mu_old = mu
            
            # Huber weights
            residuals = (data - mu) / sigma
            weights = np.where(np.abs(residuals) <= k, 1.0, k / np.abs(residuals))
            
            # Update location
            mu_new = np.sum(weights * data) / np.sum(weights)
            
            # Update scale
            residuals_new = (data - mu_new) / sigma
            huber_residuals = np.where(np.abs(residuals_new) <= k,
                                     residuals_new**2,
                                     2 * k * np.abs(residuals_new) - k**2)
            sigma = sigma * np.sqrt(np.mean(huber_residuals))
            
            # Check convergence
            if np.abs(mu_new - mu_old) < tol:
                mu = mu_new
                break
            
            mu = mu_new
        
        return mu, sigma


class CorrelationAnalysis:
    """Correlation and dependence analysis."""
    
    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray,
                           confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate Pearson correlation with confidence interval.
        
        Parameters
        ----------
        x, y : np.ndarray
            Input data arrays
        confidence_level : float
            Confidence level for interval
            
        Returns
        -------
        dict
            Correlation results with confidence interval
        """
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
        """Calculate Spearman rank correlation.
        
        Parameters
        ----------
        x, y : np.ndarray
            Input data arrays
            
        Returns
        -------
        dict
            Spearman correlation results
        """
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
    def kendall_tau(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate Kendall's tau correlation.
        
        Parameters
        ----------
        x, y : np.ndarray
            Input data arrays
            
        Returns
        -------
        dict
            Kendall's tau results
        """
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 3:
            raise StatisticsError("Need at least 3 valid data points")
        
        tau, p_value = stats.kendalltau(x_clean, y_clean)
        
        return {
            "tau": tau,
            "p_value": p_value,
            "n_samples": len(x_clean)
        }
    
    @staticmethod
    def cross_correlation(x: np.ndarray, y: np.ndarray,
                         maxlags: Optional[int] = None, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cross-correlation function.
        
        Parameters
        ----------
        x, y : np.ndarray
            Input signals
        maxlags : int, optional
            Maximum lag to compute
        normalize : bool
            Whether to normalize the correlation
            
        Returns
        -------
        tuple
            (lags, cross_correlation)
        """
        if maxlags is None:
            maxlags = min(len(x), len(y)) // 4
        
        if normalize:
            # Normalize inputs
            x_norm = (x - np.mean(x)) / np.std(x)
            y_norm = (y - np.mean(y)) / np.std(y)
        else:
            x_norm, y_norm = x, y
        
        # Calculate cross-correlation using FFT
        correlation = np.correlate(x_norm, y_norm, mode='full')
        
        if normalize:
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
        """Calculate sample covariance matrix.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        rowvar : bool
            If True, each row represents a variable
        bias : bool
            If False, use Bessel's correction (N-1 normalization)
            
        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        return np.cov(data, rowvar=rowvar, bias=bias)
    
    @staticmethod
    def shrinkage_covariance(data: np.ndarray, shrinkage: Optional[float] = None,
                           rowvar: bool = True) -> Tuple[np.ndarray, float]:
        """Calculate shrinkage covariance estimator (Ledoit-Wolf).
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        shrinkage : float, optional
            Shrinkage intensity (if None, use optimal Ledoit-Wolf)
        rowvar : bool
            If True, each row represents a variable
            
        Returns
        -------
        tuple
            (shrunk_covariance, shrinkage_intensity)
        """
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
        """Calculate condition number of covariance matrix.
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix
            
        Returns
        -------
        float
            Condition number
        """
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = eigenvals[eigenvals > 0]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return np.inf
        
        return np.max(eigenvals) / np.min(eigenvals)
    
    @staticmethod
    def effective_sample_size(cov_matrix: np.ndarray, n_samples: int) -> float:
        """Estimate effective sample size accounting for correlations.
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix
        n_samples : int
            Nominal sample size
            
        Returns
        -------
        float
            Effective sample size
        """
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
                   window: Optional[str] = None, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate periodogram power spectrum estimate.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series
        sampling_rate : float
            Sampling rate
        window : str, optional
            Window function ('hanning', 'hamming', 'blackman')
        detrend : bool
            Whether to remove linear trend
            
        Returns
        -------
        tuple
            (frequencies, power_spectrum)
        """
        if detrend:
            # Remove linear trend
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            data = data - np.polyval(coeffs, x)
        
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
        """Welch's method for power spectrum estimation.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series
        nperseg : int, optional
            Length of segments
        overlap : float
            Overlap between segments (0-1)
        window : str
            Window function
        sampling_rate : float
            Sampling rate
            
        Returns
        -------
        tuple
            (frequencies, power_spectrum)
        """
        from scipy import signal
        
        if nperseg is None:
            nperseg = len(data) // 8
        
        freqs, power = signal.welch(data, fs=sampling_rate, window=window,
                                  nperseg=nperseg, noverlap=int(nperseg * overlap))
        
        return freqs, power
    
    @staticmethod
    def multitaper_method(data: np.ndarray, nw: float = 4, k: Optional[int] = None,
                         sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Multitaper power spectrum estimation.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series
        nw : float
            Time-bandwidth product
        k : int, optional
            Number of tapers (default: 2*nw - 1)
        sampling_rate : float
            Sampling rate
            
        Returns
        -------
        tuple
            (frequencies, power_spectrum)
        """
        from scipy.signal import dpss
        
        n = len(data)
        if k is None:
            k = int(2 * nw - 1)
        
        # Generate DPSS tapers
        tapers, eigenvals = dpss(n, nw, k, return_ratios=True)
        
        # Apply tapers and compute periodograms
        periodograms = []
        for taper in tapers:
            windowed_data = data * taper
            fft_data = np.fft.fft(windowed_data)
            power = np.abs(fft_data)**2 / (sampling_rate * np.sum(taper**2))
            periodograms.append(power)
        
        # Average periodograms
        power_spectrum = np.mean(periodograms, axis=0)
        freqs = np.fft.fftfreq(n, 1/sampling_rate)
        
        # Return positive frequencies only
        n_pos = n // 2
        return freqs[:n_pos], power_spectrum[:n_pos]