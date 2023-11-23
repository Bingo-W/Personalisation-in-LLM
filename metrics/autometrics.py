from collections import OrderedDict
from .lamp_metrics import LaMP1_metrics
        
TASK_MAPPING = OrderedDict(
    [
        ('LaMP_1', LaMP1_metrics),
    ]
)

def build_compute_metrics_fn(task_name, config=None):

    if task_name in TASK_MAPPING:
        return TASK_MAPPING[task_name]
    else:
        raise KeyError(
            "The task name does not exist in the produced list"
        )
