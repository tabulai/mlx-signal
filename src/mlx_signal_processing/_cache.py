"""Byte-budget LRU for materialized GPU constant vectors (twiddles, coeffs).

Entry-count LRU limits are the wrong tool for arrays whose size scales with
the transform length: 64 cached autocorrelation twiddles once retained
~257 MiB. This cache evicts least-recently-used entries until the total
payload fits a byte budget, so memory stays bounded no matter how many
distinct lengths a workload touches while repeated lengths stay warm.
"""

from __future__ import annotations

from collections import OrderedDict


class ByteBudgetCache:
    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self._entries: OrderedDict[tuple, tuple[object, int]] = OrderedDict()
        self._total = 0

    def get(self, key: tuple):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: tuple, value, nbytes: int) -> None:
        if int(nbytes) > self.budget:
            return  # oversized entries are returned to the caller uncached
        if key in self._entries:
            self._total -= self._entries.pop(key)[1]
        self._entries[key] = (value, int(nbytes))
        self._total += int(nbytes)
        while self._total > self.budget and len(self._entries) > 1:
            _, (_, freed) = self._entries.popitem(last=False)
            self._total -= freed


#: shared budget for all FFT twiddle/coefficient vectors (complex64). Sized
#: to hold the working set of one >8M-sample four-step resample (~114 MiB of
#: concurrent twiddles) with headroom for a second transform length; a sweep
#: over arbitrarily many lengths still caps here instead of growing linearly.
TWIDDLES = ByteBudgetCache(192 << 20)
