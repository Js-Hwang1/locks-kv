from __future__ import annotations
import ctypes
import glob
import math
import os
from pathlib import Path
import torch

_libcudart = None


def host_numa_node() -> int:
    value = os.environ.get("LOCKS_HOST_NUMA_NODE", "0")
    try:
        node = int(value)
    except ValueError as exc:
        raise ValueError("LOCKS_HOST_NUMA_NODE must be a nonnegative CPU NUMA node") from exc
    if node < 0:
        raise ValueError("LOCKS_HOST_NUMA_NODE must be nonnegative")
    cpus = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not cpus.is_file() or not cpus.read_text().strip():
        raise ValueError(f"LOCKS_HOST_NUMA_NODE={node} is not a NUMA node with CPUs")
    return node


def _cudart():
    global _libcudart
    if _libcudart is not None:
        return _libcudart
    last = None
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += sorted(
        glob.glob(
            os.path.join(
                os.path.dirname(torch.__file__),
                "..",
                "nvidia",
                "cuda_runtime",
                "lib",
                "libcudart.so*",
            )
        )
    )
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaHostGetDevicePointer.restype = ctypes.c_int
            lib.cudaHostGetDevicePointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            lib.cudaHostAlloc.restype = ctypes.c_int
            lib.cudaHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            lib.cudaFreeHost.restype = ctypes.c_int
            lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
            lib.cudaDeviceSynchronize.restype = ctypes.c_int
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostRegister.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            lib.cudaHostUnregister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            _libcudart = lib
            return lib
        except (OSError, AttributeError) as exc:
            last = exc
    raise RuntimeError(f"could not load libcudart: {last}")


def host_device_pointer(host_tensor: torch.Tensor) -> int:
    if not host_tensor.is_pinned():
        raise ValueError("pool tensor must be pinned (pin_memory=True)")
    dptr = ctypes.c_void_p(0)
    rc = _cudart().cudaHostGetDevicePointer(
        ctypes.byref(dptr), ctypes.c_void_p(host_tensor.data_ptr()), 0
    )
    if rc != 0:
        raise RuntimeError(
            f"cudaHostGetDevicePointer failed rc={rc} (pinned memory not device-mapped?)"
        )
    return dptr.value


class _RegisteredPinnedOwner:
    def __init__(self, nbytes, huge):
        self.pointer = ctypes.c_void_p()
        self.registered = False
        self.numa_node = host_numa_node()
        self.lib = _cudart()
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self.libc.munmap.restype = ctypes.c_int
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.madvise.restype = ctypes.c_int
        self.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        page = os.sysconf("SC_PAGE_SIZE")
        align = 536870912 if huge else page
        size = (nbytes + page - 1) // page * page
        raw = self.libc.mmap(None, size + align, 3, 34, -1, 0)
        if raw in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap CPU KV store failed")
        aligned = (raw + align - 1) // align * align
        prefix = aligned - raw
        suffix = align - prefix
        if prefix:
            assert self.libc.munmap(ctypes.c_void_p(raw), prefix) == 0
        if suffix:
            assert self.libc.munmap(ctypes.c_void_p(aligned + size), suffix) == 0
        self.pointer = ctypes.c_void_p(aligned)
        self.size = size
        numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
        numa.mbind.restype = ctypes.c_long
        numa.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        words = self.numa_node // 64 + 1
        mask = (ctypes.c_ulong * words)()
        mask[self.numa_node // 64] = 1 << (self.numa_node % 64)
        rc = numa.mbind(self.pointer, size, 2, mask, words * 64, 0)
        if rc != 0:
            raise OSError(ctypes.get_errno(), f"mbind CPU node {self.numa_node} failed")
        rc = self.libc.madvise(self.pointer, size, 14 if huge else 15)
        if rc != 0:
            raise OSError(ctypes.get_errno(), "madvise host KV policy failed")

    def register(self):
        rc = self.lib.cudaHostRegister(self.pointer, self.size, 2)
        if rc != 0:
            raise RuntimeError(f"cudaHostRegister({self.size}) failed rc={rc}")
        self.registered = True

    def __del__(self):
        if not self.pointer.value:
            return
        if self.registered:
            if self.lib.cudaDeviceSynchronize() != 0:
                return
            if self.lib.cudaHostUnregister(self.pointer) != 0:
                return
        self.libc.munmap(self.pointer, self.size)
        self.pointer.value = None


def exact_pinned_zeros(shape, dtype):
    nbytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    owner = _RegisteredPinnedOwner(nbytes, huge=True)
    buffer = (ctypes.c_ubyte * nbytes).from_address(owner.pointer.value)
    buffer._owner = owner
    tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
    tensor.zero_()
    owner.register()
    if not tensor.is_pinned():
        raise RuntimeError("CUDA host allocation was not recognized as pinned")
    return tensor


class DevPtrTensor:
    __slots__ = ("_ptr", "dtype", "device", "shape")

    def __init__(self, ptr: int, dtype: torch.dtype, device: torch.device, shape=()):
        self._ptr = int(ptr)
        self.dtype = dtype
        self.device = device
        self.shape = shape

    def data_ptr(self) -> int:
        return self._ptr


class MappedHostVPool:
    HOST_SLACK_BYTES = 32 << 30

    @staticmethod
    def _host_available_bytes() -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        n_kv: int,
        page: int,
        d: int,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
        max_pool_gb: float = 256.0,
    ):
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"pool dtype must be bf16/fp16, got {dtype}")
        (self.L, self.NB, self.n_kv) = (num_layers, num_blocks, n_kv)
        (self.page, self.d, self.dtype) = (page, d, dtype)
        elem = torch.empty(0, dtype=dtype).element_size()
        pool_bytes = num_layers * num_blocks * n_kv * page * d * elem
        if pool_bytes > max_pool_gb * (1 << 30):
            raise RuntimeError(
                f"pinned V pool would be {pool_bytes / 2**30:.1f} GiB (> cap max_pool_gb={max_pool_gb}); reduce num_gpu_blocks (num_gpu_blocks_override) or raise LocksConfig.max_pool_gb. NOTE this is HOST DRAM (the offload target), not VRAM."
            )
        avail = self._host_available_bytes()
        factor = 1.25
        need = int(pool_bytes * factor) + self.HOST_SLACK_BYTES
        if avail and need > avail:
            raise RuntimeError(
                f"pinned pool of {pool_bytes / 2**30:.1f} GiB needs ~{need / 2**30:.1f} GiB host headroom (factor {factor} + {self.HOST_SLACK_BYTES >> 30} GiB slack) but MemAvailable is {avail / 2**30:.1f} GiB -- refusing. Reduce num_gpu_blocks_override / max_model_len or free host RAM."
            )
        dev = (
            torch.device(device)
            if device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        shape = (num_layers, num_blocks, n_kv, page, d)
        self.numa_node = host_numa_node()
        self.host = exact_pinned_zeros(shape, dtype)
        self.dev_view = DevPtrTensor(
            host_device_pointer(self.host), torch.int16, dev, tuple(self.host.shape)
        )
        self.allocation_kind = "registered_huge"
        self.allocation_requested_bytes = pool_bytes
        self.guard_factor = factor
        self.pool_gib = pool_bytes / 2**30
        self.device = dev

    def page_offset(self, layer: int, blk: int, kv: int) -> int:
        return ((layer * self.NB + blk) * self.n_kv + kv) * self.page * self.d

    def vram_note(self) -> str:
        return f"host_pool={self.pool_gib:.2f}GiB (CPU NUMA node {self.numa_node}, keyed by NB={self.NB}; VRAM hot/staging bounded to working set, see residency/staging)"
