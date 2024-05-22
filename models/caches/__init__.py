from . import cache

CACHES = {
    "none": cache.LMCache,
}


def get_cache(cache_name):
    return CACHES[cache_name]


def registered_caches():
    return CACHES.keys()
