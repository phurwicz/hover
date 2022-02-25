???+ info "Caching and reading from disk"
    This guide uses [`@wrappy.memoize`](https://erniethornhill.github.io/wrappy/) in place of `@functools.lru_cache` for caching.

    -   The benefit is that `wrappy.memoize` can persist the cache to disk, speeding up code across sessions.

    Cached values for this guide have been pre-computed, making it much master to run the guide.
