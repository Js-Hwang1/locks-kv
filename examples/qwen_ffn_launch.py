import triton
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner, generate_lookup_hash_from_source_code,
)

HASH = 'ce6af60bd9142654d7aae3e92430b68fd0392a4057b84215cbd9dbb735a0705b'
CONFIG = dict(XBLOCK=4, R0_BLOCK=1024, num_warps=4, num_stages=1)
EVENTS = []


def target(tuner):
    return generate_lookup_hash_from_source_code(str(tuner.size_hints), tuner.fn.src) == HASH


def values(config):
    return dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)


def desired():
    return triton.Config(dict(XBLOCK=4, R0_BLOCK=1024), num_warps=4, num_stages=1)


def install():
    original_init = CachingAutotuner.__init__
    original_recheck = CachingAutotuner.recheck_autotune_cache

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if target(self):
            self.inductor_meta = dict(self.inductor_meta, dynamic_scale_rblock=False,
                                      coordinate_descent_tuning=False)
            before = [values(c) for c in self.configs]
            self.configs = [desired()]
            EVENTS.append(dict(phase='construction', before=before, after=CONFIG))

    def recheck(self, reload_kernel_from_src):
        if not target(self):
            return original_recheck(self, reload_kernel_from_src)
        self.inductor_meta = dict(self.inductor_meta, dynamic_scale_rblock=False,
                                  coordinate_descent_tuning=False)
        before = [values(c.config) for c in self.compile_results]
        matching = [c for c in self.compile_results if values(c.config) == CONFIG]
        if not matching:
            if self.fn.fn is None:
                self.fn = reload_kernel_from_src().fn
            matching = [self._precompile_config(desired())]
        self.compile_results = matching[:1]
        self.configs = [desired()]
        assert not self.launchers
        EVENTS.append(dict(phase='cache_restore', before=before, after=CONFIG))

    CachingAutotuner.__init__ = init
    CachingAutotuner.recheck_autotune_cache = recheck
