from typing import Callable

_CALLBACKS = []
_HOOKED = False


def on_model_loaded(fn: Callable) -> None:
    global _HOOKED
    _CALLBACKS.append(fn)
    if _HOOKED:
        return
    _HOOKED = True
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    orig = BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        model = orig(self, vllm_config, model_config, prefix=prefix)
        for cb in _CALLBACKS:
            cb(model, vllm_config)
        return model

    BaseModelLoader.load_model = load_model
