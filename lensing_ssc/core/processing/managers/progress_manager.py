"""
Progress tracking and monitoring for processing operations.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable, Union
from contextlib import contextmanager
from pathlib import Path

from ...base.exceptions import ProcessingError

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class ProgressManager:
    """Manager for tracking and displaying processing progress.
    
    Provides both programmatic progress tracking and optional visual progress bars.
    Supports nested progress tracking for complex operations.
    
    Parameters
    ----------
    total : int, optional
        Total number of operations expected
    description : str, optional
        Description for progress display
    unit : str, optional
        Unit for progress display (default: "it")
    enable_bar : bool, optional
        Whether to show progress bar (default: True if tqdm available)
    log_interval : int, optional
        Log progress every N updates (default: 10)
    time_remaining : bool, optional
        Whether to estimate time remaining (default: True)
    """
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "it",
        enable_bar: bool = None,
        log_interval: int = 10,
        time_remaining: bool = True
    ):
        self.total = total
        self.description = description
        self.unit = unit
        self.log_interval = log_interval
        self.time_remaining = time_remaining
        
        # Auto-detect if we should show progress bar
        if enable_bar is None:
            self.enable_bar = _HAS_TQDM and total is not None
        else:
            self.enable_bar = enable_bar and _HAS_TQDM
        
        # Progress tracking
        self.current = 0
        self.start_time = None
        self.last_update_time = None
        self.last_log_count = 0
        
        # Progress bar
        self._pbar = None
        self._nested_bars = []
        
        # Callbacks
        self._update_callbacks = []
        self._completion_callbacks = []
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistics
        self.stats = {
            'updates': 0,
            'total_time': 0,
            'average_rate': 0,
            'estimated_remaining': None
        }
    
    def start(self) -> None:
        """Start progress tracking."""
        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time
        self.current = 0
        
        if self.enable_bar and self.total is not None:
            self._pbar = tqdm(
                total=self.total,
                desc=self.description,
                unit=self.unit,
                dynamic_ncols=True
            )
        
        self.logger.info(f"Started progress tracking: {self.description}")
        if self.total:
            self.logger.info(f"Total operations: {self.total}")
    
    def update(self, n: int = 1, info: str = "", **kwargs) -> None:
        """Update progress by n steps.
        
        Parameters
        ----------
        n : int
            Number of steps to advance
        info : str
            Additional information to display
        **kwargs
            Additional metadata
        """
        if self.start_time is None:
            self.start()
        
        self.current += n
        self.stats['updates'] += 1
        current_time = time.perf_counter()
        
        # Update progress bar
        if self._pbar is not None:
            self._pbar.update(n)
            if info:
                self._pbar.set_postfix_str(info)
        
        # Calculate statistics
        self._update_statistics(current_time)
        
        # Log progress periodically
        if (self.current - self.last_log_count) >= self.log_interval:
            self._log_progress(info)
            self.last_log_count = self.current
        
        # Call update callbacks
        for callback in self._update_callbacks:
            try:
                callback(self.current, self.total, info, **kwargs)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
        
        # Check for completion
        if self.total and self.current >= self.total:
            self._on_completion()
        
        self.last_update_time = current_time
    
    def set_description(self, description: str) -> None:
        """Update progress description."""
        self.description = description
        if self._pbar is not None:
            self._pbar.set_description(description)
    
    def set_postfix(self, **kwargs) -> None:
        """Set postfix information for progress bar."""
        if self._pbar is not None:
            self._pbar.set_postfix(**kwargs)
    
    def close(self) -> None:
        """Close progress tracking."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        
        # Close nested bars
        for bar in self._nested_bars:
            if bar is not None:
                bar.close()
        self._nested_bars.clear()
        
        self.logger.info(f"Completed progress tracking: {self.description}")
        if self.start_time:
            total_time = time.perf_counter() - self.start_time
            self.logger.info(f"Total time: {self._format_duration(total_time)}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    @contextmanager
    def nested(self, total: Optional[int] = None, description: str = "Subtask", **kwargs):
        """Create nested progress tracking.
        
        Parameters
        ----------
        total : int, optional
            Total for nested operation
        description : str
            Description for nested operation
        **kwargs
            Additional arguments for nested progress bar
        """
        if self.enable_bar and _HAS_TQDM:
            nested_bar = tqdm(
                total=total,
                desc=description,
                unit=kwargs.get('unit', self.unit),
                leave=False,
                position=len(self._nested_bars) + 1
            )
            self._nested_bars.append(nested_bar)
        else:
            nested_bar = None
        
        class NestedProgress:
            def __init__(self, bar, parent_logger):
                self.bar = bar
                self.logger = parent_logger
                self.current = 0
                
            def update(self, n=1, info=""):
                self.current += n
                if self.bar:
                    self.bar.update(n)
                    if info:
                        self.bar.set_postfix_str(info)
            
            def close(self):
                if self.bar:
                    self.bar.close()
        
        nested_progress = NestedProgress(nested_bar, self.logger)
        
        try:
            yield nested_progress
        finally:
            nested_progress.close()
            if nested_bar in self._nested_bars:
                self._nested_bars.remove(nested_bar)
    
    def add_update_callback(self, callback: Callable) -> None:
        """Add callback function called on each update.
        
        Parameters
        ----------
        callback : Callable
            Function with signature: callback(current, total, info, **kwargs)
        """
        self._update_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable) -> None:
        """Add callback function called on completion.
        
        Parameters
        ----------
        callback : Callable
            Function with signature: callback(stats)
        """
        self._completion_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get progress statistics.
        
        Returns
        -------
        Dict[str, Any]
            Progress statistics
        """
        current_time = time.perf_counter()
        if self.start_time:
            self.stats['total_time'] = current_time - self.start_time
        
        return self.stats.copy()
    
    def get_rate(self) -> float:
        """Get current processing rate (items per second).
        
        Returns
        -------
        float
            Processing rate
        """
        if self.start_time and self.current > 0:
            elapsed = time.perf_counter() - self.start_time
            return self.current / elapsed if elapsed > 0 else 0
        return 0
    
    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion.
        
        Returns
        -------
        Optional[float]
            Estimated seconds remaining, or None if cannot estimate
        """
        if not self.total or not self.time_remaining:
            return None
        
        rate = self.get_rate()
        if rate > 0 and self.current < self.total:
            remaining_items = self.total - self.current
            return remaining_items / rate
        
        return None
    
    def _update_statistics(self, current_time: float) -> None:
        """Update internal statistics."""
        if self.start_time:
            self.stats['total_time'] = current_time - self.start_time
            if self.stats['total_time'] > 0:
                self.stats['average_rate'] = self.current / self.stats['total_time']
        
        self.stats['estimated_remaining'] = self.get_eta()
    
    def _log_progress(self, info: str = "") -> None:
        """Log current progress."""
        if self.total:
            percent = (self.current / self.total) * 100
            rate = self.get_rate()
            eta = self.get_eta()
            
            msg = f"Progress: {self.current}/{self.total} ({percent:.1f}%) - {rate:.2f} {self.unit}/s"
            
            if eta is not None:
                msg += f" - ETA: {self._format_duration(eta)}"
            
            if info:
                msg += f" - {info}"
        else:
            rate = self.get_rate()
            msg = f"Progress: {self.current} - {rate:.2f} {self.unit}/s"
            if info:
                msg += f" - {info}"
        
        self.logger.info(msg)
    
    def _on_completion(self) -> None:
        """Handle completion."""
        final_stats = self.get_stats()
        
        # Call completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(final_stats)
            except Exception as e:
                self.logger.warning(f"Completion callback failed: {e}")
        
        self.logger.info(f"Completed: {self.description}")
        self.logger.info(f"Final stats: {final_stats}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


class MultiStageProgress:
    """Progress manager for multi-stage operations.
    
    Manages progress across multiple stages with different totals and descriptions.
    
    Parameters
    ----------
    stages : List[Dict[str, Any]]
        List of stage definitions with 'name', 'total', and optional 'weight'
    """
    
    def __init__(self, stages: list):
        self.stages = stages
        self.current_stage = 0
        self.stage_managers = []
        self.overall_progress = None
        
        # Calculate overall total with weights
        self.overall_total = 0
        for stage in stages:
            weight = stage.get('weight', 1.0)
            total = stage.get('total', 100)
            self.overall_total += total * weight
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def start(self) -> None:
        """Start multi-stage progress tracking."""
        self.overall_progress = ProgressManager(
            total=int(self.overall_total),
            description="Overall Progress",
            unit="ops"
        )
        self.overall_progress.start()
        
        self.logger.info(f"Starting multi-stage progress: {len(self.stages)} stages")
    
    def next_stage(self) -> ProgressManager:
        """Move to next stage and return its progress manager.
        
        Returns
        -------
        ProgressManager
            Progress manager for current stage
        """
        if self.current_stage >= len(self.stages):
            raise ProcessingError("No more stages available")
        
        stage = self.stages[self.current_stage]
        stage_manager = ProgressManager(
            total=stage.get('total'),
            description=stage.get('name', f"Stage {self.current_stage + 1}"),
            unit=stage.get('unit', 'it')
        )
        
        # Add callback to update overall progress
        weight = stage.get('weight', 1.0)
        def update_overall(current, total, info, **kwargs):
            if total and self.overall_progress:
                stage_contribution = (current / total) * (total * weight)
                # Update overall based on completed stages plus current stage progress
                completed_contribution = sum(
                    s.get('total', 100) * s.get('weight', 1.0) 
                    for s in self.stages[:self.current_stage]
                )
                overall_current = int(completed_contribution + stage_contribution)
                self.overall_progress.current = overall_current
                if self.overall_progress._pbar:
                    self.overall_progress._pbar.n = overall_current
                    self.overall_progress._pbar.refresh()
        
        stage_manager.add_update_callback(update_overall)
        self.stage_managers.append(stage_manager)
        
        stage_manager.start()
        self.current_stage += 1
        
        return stage_manager
    
    def close(self) -> None:
        """Close all progress managers."""
        for manager in self.stage_managers:
            manager.close()
        
        if self.overall_progress:
            self.overall_progress.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()