# P08_C021_R08V1 local execution

[run.py](run.py) runs the model and local performance evaluation. Its input and output arguments are documented in [the experiment README](../README.md#running-this-experiment). The source is a direct sequence of data loading, historical fitting, risk estimation, allocation and evaluation; there is no scheduler or diagnostic-launch framework.

The shared functions and their dependency graph are documented in [src](../../../src/README.md). This research entry point is separate from the self-contained contest script.
