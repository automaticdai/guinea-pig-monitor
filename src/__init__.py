"""
Guinea Pig Behavior Monitoring System
Source package containing all core classes
"""

from .roi_manager import ROIManager
from .optical_flow import OpticalFlowAnalyzer
from .behavior_classifier import BehaviorClassifier
from .statistics import StatisticsAggregator
from .monitor import GuineaPigMonitor

__all__ = [
    'ROIManager',
    'OpticalFlowAnalyzer',
    'BehaviorClassifier',
    'StatisticsAggregator',
    'GuineaPigMonitor',
]

