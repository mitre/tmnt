import collections
import itertools
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = []

class LazyThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with lazy iterable collection in map().
    Implementation taken from https://github.com/python/cpython/pull/707
    """

    def map(self, fn, *iterables, timeout=None, prefetch=None):
        # pylint: disable=arguments-differ
        """Lazy apdaption of ThreadPoolExecutor.map.
        Unlike ThreadPoolExecutor.map:
        - iterables are prefetched lazily
        - if only a single iterable is specified, iter(iterables[0]) is used
          instead of zip(*iterables) to obtain a iterator over the arguments
          that are mapped to fn. This is to match the behavior of
          mxnet.gluon.Dataset.transform and gluonnlp.data.DataStream.transform
          which unpack argument tuples.
        """
        if timeout is not None:
            end_time = timeout + time.time()
        if prefetch is None:
            prefetch = self._max_workers
        if prefetch < 0:
            raise ValueError('prefetch count may not be negative')

        if len(iterables) > 1:
            argsiter = zip(*iterables)
        else:
            argsiter = iter(iterables[0])
        fs = collections.deque(
            self.submit(fn, *args)
            for args in itertools.islice(argsiter, self._max_workers +
                                         prefetch))

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def _result_iterator():
            nonlocal argsiter
            try:
                while fs:
                    res = fs[0].result() if timeout is None else fs[0].result(
                        end_time - time.time())
                    # Got a result, future needn't be cancelled
                    del fs[0]
                    # Dispatch next task before yielding to keep pipeline full
                    if argsiter:
                        try:
                            args = next(argsiter)
                        except StopIteration:
                            argsiter = None
                        else:
                            fs.append(self.submit(fn, *args))
                    yield res
            finally:
                for future in fs:
                    future.cancel()

        return _result_iterator()
