from collections import OrderedDict
from .lamp_metrics import (
    LaMP1_metrics, 
    LaMP2_metrics,
    LaMP3_metrics,
    LaMP4_metrics,
    LaMP5_metrics,
    LaMP6_metrics,
    LaMP7_metrics
)
        
TASK_MAPPING = OrderedDict(
    [
        ('LaMP_1', LaMP1_metrics),
        ('LaMP_2', LaMP2_metrics),
        ('LaMP_3', LaMP3_metrics),
        ('LaMP_4', LaMP4_metrics),
        ('LaMP_5', LaMP5_metrics),
        ('LaMP_6', LaMP6_metrics),
        ('LaMP_7', LaMP7_metrics),
    ]
)

def build_compute_metrics_fn(task_name, config=None):

    if task_name in TASK_MAPPING:
        return TASK_MAPPING[task_name]
    else:
        raise KeyError(
            "The task name does not exist in the produced list"
        )