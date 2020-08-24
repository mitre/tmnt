The Topic Modeling Neural Toolkit (TMNT) provides an implementation for training
topic models as neural network-based variational auto-encoders.


Documentation can be found here: https://tmnt.readthedocs.io/en/latest/

Current NOTES:
--------------

Autogluon.  Port from HPBandster generally complete.  Issues with autogluon make tuning possible in
just a few ways:

 - scheduler: hyperband, searcher: random    ==> works with CPU and GPU
 - scheduler: fifo, searcher: random         ==> works with CPU and GPU
 - scheduler: fifo, searcher: bayesopt       ==> works with CPU only and with smaller number of threads
 - scheduler: fifo, searcher: skopt          ==> works with GPU and CPU with smaller number of threads (why??)

Other configurations will hang or break.  In particular bayesopt doesn't work with hyperband.

TODO: fifo scheduler is not multi-fidelity so doesn't need reporter results on objective after each epoch - only after
final number of epochs reached.  Question: would this allow for epochs to be a real hyperparameter???

