"""
Progress manager for tracking and reporting processing progress.

Provides comprehensive progress tracking with multiple display formats,
nested progress support, performance metrics, and thread-safe operations.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import sys
import os

from .exceptions import ProgressError
from ...config.settings import ProcessingConfig


logger = logging.getLogger(__name__)


@dataclass
class ProgressMetrics:
    """Performance metrics for progress tracking."""
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    total: Optional[int] = None
    current: int = 0
    rate: float = 0.0
    elapsed: float = 0.0
    eta: Optional[float] = None
    
    def update_metrics(self, current: int) -> None:
        """Update performance metrics."""
        now = time.time()
        self.elapsed = now - self.start_time
        
        if current > self.current:
            time_diff = now - self.last_update
            if time_diff > 0:
                self.rate = (current - self.current) / time_diff
        
        self.current = current
        self.last_update = now
        
        # Calculate ETA
        if self.total and self.rate > 0:
            remaining = self.total - current
            self.eta = remaining / self.rate


class ProgressTracker:
    """Individual progress tracker with display and metrics."""
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "items",
        show_rate: bool = True,
        show_eta: bool = True,
        update_interval: float = 0.1
    ):
        self.total = total
        self.description = description
        self.unit = unit
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.update_interval = update_interval
        
        self.metrics = ProgressMetrics(total=total)
        self._lock = threading.Lock()
        self._last_display_update = 0.0
        self._paused = False
        self._pause_start = 0.0
        self._total_pause_time = 0.0
        
        # Display state
        self._last_line_length = 0
        self._display_enabled = self._should_display()
    
    def update(self, n: int = 1, **kwargs) -> None:
        """Update progress by n steps."""
        with self._lock:
            if self._paused:
                return
                
            new_current = self.metrics.current + n
            if self.total and new_current > self.total:
                new_current = self.total
                
            self.metrics.update_metrics(new_current)
            
            # Update description if provided
            if 'description' in kwargs:
                self.description = kwargs['description']
            
            # Display update
            now = time.time()
            if (now - self._last_display_update) >= self.update_interval:
                self._display()
                self._last_display_update = now
    
    def set_current(self, current: int) -> None:
        """Set absolute progress position."""
        with self._lock:
            if self._paused:
                return
            self.metrics.update_metrics(current)
            self._display()
    
    def pause(self) -> None:
        """Pause progress tracking."""
        with self._lock:
            if not self._paused:
                self._paused = True
                self._pause_start = time.time()
    
    def resume(self) -> None:
        """Resume progress tracking."""
        with self._lock:
            if self._paused:
                self._paused = False
                self._total_pause_time += time.time() - self._pause_start
                self.metrics.start_time += self._total_pause_time
    
    def finish(self) -> None:
        """Mark progress as complete."""
        with self._lock:
            if self.total:
                self.metrics.update_metrics(self.total)
            self._display(force=True)
            if self._display_enabled:
                print()  # New line after completion
    
    def _display(self, force: bool = False) -> None:
        """Display progress information."""
        if not self._display_enabled and not force:
            return
            
        line = self._format_line()
        
        if self._display_enabled:
            # Clear previous line
            if self._last_line_length > 0:
                print('\r' + ' ' * self._last_line_length + '\r', end='')
            
            # Print new line
            print(f'\r{line}', end='', flush=True)
            self._last_line_length = len(line)
    
    def _format_line(self) -> str:
        """Format progress line for display."""
        parts = [self.description]
        
        # Progress indicator
        if self.total:
            percentage = (self.metrics.current / self.total) * 100
            bar_width = 30
            filled_width = int(bar_width * self.metrics.current / self.total)
            bar = '█' * filled_width + '░' * (bar_width - filled_width)
            parts.append(f"|{bar}| {self.metrics.current}/{self.total} ({percentage:.1f}%)")
        else:
            parts.append(f"{self.metrics.current} {self.unit}")
        
        # Rate
        if self.show_rate and self.metrics.rate > 0:
            if self.metrics.rate >= 1:
                parts.append(f"{self.metrics.rate:.1f} {self.unit}/s")
            else:
                parts.append(f"{1/self.metrics.rate:.1f} s/{self.unit}")
        
        # Elapsed time
        elapsed_str = self._format_time(self.metrics.elapsed)
        parts.append(f"[{elapsed_str}")
        
        # ETA
        if self.show_eta and self.metrics.eta is not None:
            eta_str = self._format_time(self.metrics.eta)
            parts[-1] += f"<{eta_str}"
        
        parts[-1] += "]"
        
        return " ".join(parts)
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m{seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def _should_display(self) -> bool:
        """Check if progress should be displayed."""
        return (
            sys.stdout.isatty() and 
            os.getenv('JUPYTER_RUNNING') != '1' and
            not os.getenv('PYTEST_RUNNING')
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tracker status."""
        return {
            'description': self.description,
            'total': self.total,
            'current': self.metrics.current,
            'percentage': (self.metrics.current / self.total * 100) if self.total else None,
            'rate': self.metrics.rate,
            'elapsed': self.metrics.elapsed,
            'eta': self.metrics.eta,
            'paused': self._paused,
        }


class ProgressManager:
    """Manager for multiple progress trackers with coordination."""
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "items",
        show_rate: bool = True,
        show_eta: bool = True,
        nested_indent: str = "  ",
        config: Optional[ProcessingConfig] = None
    ):
        """Initialize progress manager.
        
        Parameters
        ----------
        total : int, optional
            Total number of items to process
        description : str
            Main progress description
        unit : str
            Unit name for items
        show_rate : bool
            Show processing rate
        show_eta : bool
            Show estimated time to completion
        nested_indent : str
            Indentation for nested progress
        config : ProcessingConfig, optional
            Configuration object
        """
        # Load from config if provided
        if config:
            show_rate = getattr(config, 'enable_progress_bar', True)
            
        self.main_tracker = ProgressTracker(
            total=total,
            description=description,
            unit=unit,
            show_rate=show_rate,
            show_eta=show_eta
        )
        
        self.nested_indent = nested_indent
        self._subtrackers: Dict[str, ProgressTracker] = {}
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        
        # Global state
        self._enabled = config.enable_progress_bar if config else True
        self._log_progress = False
        
        logger.debug(f"ProgressManager initialized with total={total}")
    
    def update(self, n: int = 1, **kwargs) -> None:
        """Update main progress."""
        self.main_tracker.update(n, **kwargs)
        self._trigger_callbacks('main', self.main_tracker.get_status())
    
    def set_total(self, total: int) -> None:
        """Set or update total items."""
        self.main_tracker.total = total
        self.main_tracker.metrics.total = total
    
    def create_subtracker(
        self,
        name: str,
        total: Optional[int] = None,
        description: str = "",
        **kwargs
    ) -> ProgressTracker:
        """Create nested progress tracker."""
        with self._lock:
            if name in self._subtrackers:
                raise ProgressError(f"Subtracker '{name}' already exists")
            
            # Add indentation to description
            if description and self.nested_indent:
                description = self.nested_indent + description
            
            tracker = ProgressTracker(
                total=total,
                description=description,
                **kwargs
            )
            
            self._subtrackers[name] = tracker
            logger.debug(f"Created subtracker: {name}")
            return tracker
    
    def remove_subtracker(self, name: str) -> None:
        """Remove nested progress tracker."""
        with self._lock:
            if name in self._subtrackers:
                self._subtrackers[name].finish()
                del self._subtrackers[name]
    
    def get_subtracker(self, name: str) -> Optional[ProgressTracker]:
        """Get existing subtracker."""
        return self._subtrackers.get(name)
    
    @contextmanager
    def subprogress(self, name: str, total: Optional[int] = None, description: str = ""):
        """Context manager for nested progress."""
        tracker = self.create_subtracker(name, total, description)
        try:
            yield tracker
        finally:
            self.remove_subtracker(name)
    
    def pause_all(self) -> None:
        """Pause all progress trackers."""
        self.main_tracker.pause()
        for tracker in self._subtrackers.values():
            tracker.pause()
    
    def resume_all(self) -> None:
        """Resume all progress trackers."""
        self.main_tracker.resume()
        for tracker in self._subtrackers.values():
            tracker.resume()
    
    def finish(self) -> None:
        """Complete all progress tracking."""
        # Finish subtrackers first
        for tracker in self._subtrackers.values():
            tracker.finish()
        
        # Finish main tracker
        self.main_tracker.finish()
        
        # Clear subtrackers
        self._subtrackers.clear()
    
    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add progress update callback."""
        self._callbacks.append(callback)
    
    def enable_logging(self, log_interval: int = 10) -> None:
        """Enable progress logging."""
        self._log_progress = True
        
        def log_callback(tracker_name: str, status: Dict[str, Any]) -> None:
            if status['current'] % log_interval == 0:
                percentage = status.get('percentage', 0) or 0
                logger.info(f"{tracker_name}: {status['current']}/{status['total']} ({percentage:.1f}%)")
        
        self.add_callback(log_callback)
    
    def _trigger_callbacks(self, tracker_name: str, status: Dict[str, Any]) -> None:
        """Trigger progress callbacks."""
        for callback in self._callbacks:
            try:
                callback(tracker_name, status)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive progress status."""
        status = {
            'main': self.main_tracker.get_status(),
            'subtrackers': {
                name: tracker.get_status() 
                for name, tracker in self._subtrackers.items()
            },
            'enabled': self._enabled,
            'callback_count': len(self._callbacks),
        }
        
        # Calculate overall progress
        if status['main']['total']:
            status['overall_percentage'] = status['main']['percentage']
        else:
            status['overall_percentage'] = None
        
        return status
    
    def export_progress(self) -> Dict[str, Any]:
        """Export progress data for persistence."""
        return {
            'timestamp': time.time(),
            'main_progress': {
                'current': self.main_tracker.metrics.current,
                'total': self.main_tracker.total,
                'elapsed': self.main_tracker.metrics.elapsed,
            },
            'subtrackers': {
                name: {
                    'current': tracker.metrics.current,
                    'total': tracker.total,
                    'description': tracker.description,
                }
                for name, tracker in self._subtrackers.items()
            }
        }
    
    def import_progress(self, data: Dict[str, Any]) -> None:
        """Import progress data from persistence."""
        if 'main_progress' in data:
            main_data = data['main_progress']
            if main_data.get('current'):
                self.main_tracker.set_current(main_data['current'])
            if main_data.get('total'):
                self.set_total(main_data['total'])
        
        if 'subtrackers' in data:
            for name, tracker_data in data['subtrackers'].items():
                if name not in self._subtrackers:
                    self.create_subtracker(
                        name,
                        total=tracker_data.get('total'),
                        description=tracker_data.get('description', '')
                    )
                if tracker_data.get('current'):
                    self._subtrackers[name].set_current(tracker_data['current'])
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()