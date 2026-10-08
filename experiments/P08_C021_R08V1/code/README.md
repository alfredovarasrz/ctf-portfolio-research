# Selected combined-model research code

This folder is reserved for the runnable research implementation of the final
selected strategy, **P08-only + C02 regular + R08 expanding variance**.

Its entry point will run the complete combined method, including historical
parameter selection, expected-return forecasts, risk estimation, stock weights
and local performance evaluation. It will call the shared scientific functions
in [src](../../../src/README.md). The parent experiment folder contains the
[method and saved results](../README.md).

The runnable entry point has not yet been packaged. The existing scientific
functions are linked from the parent README; creating this folder does not
mean that a new model has been fitted or that the final export is complete.

The separate [submission folder](../../../submission/README.md) will contain
the exact self-contained contest version of this same strategy and its pinned
dependencies. That version will return stock weights through the contest's
required `main` interface, without the local performance-evaluation runner.
