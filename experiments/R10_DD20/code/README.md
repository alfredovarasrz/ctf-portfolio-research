# Drawdown-conditioned factor covariance

Run from the repository root with Python 3.13 and requirements-research.txt installed.

```sh
python experiments/R10_DD20/code/run.py --data /path/to/ctf-tables --output /path/to/new-r10_dd20-results --threads 10
```

The input folder contains ctff_chars.parquet, ctff_features.parquet and
ctff_daily_ret.parquet. Use a new output folder for an independent run.
The script estimates its components from those tables, constructs monthly
weights and evaluates completed returns. It writes new weights, returns,
selection records and summary statistics in the requested output folder.
Historical hyperparameter validation is part of the strategy and remains in
the scientific code. Private correctness tests and scheduling code are excluded.
The saved results.json in the experiment folder records the original full run;
executing this script does not replace that record.
