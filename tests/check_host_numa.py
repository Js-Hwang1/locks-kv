import ctypes
import gc
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locks.tier import pool


class HostNumaTests(unittest.TestCase):
    def test_measured_default(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="0-71\n"):
            self.assertEqual(pool.host_numa_node(), 0)

    def test_explicit_cpu_node(self):
        with patch.dict(os.environ, {"LOCKS_HOST_NUMA_NODE": "1"}), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="72-143\n"):
            self.assertEqual(pool.host_numa_node(), 1)

    def test_invalid_node(self):
        for value in ("-1", "gpu", "1.5", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"LOCKS_HOST_NUMA_NODE": value}):
                with self.assertRaises(ValueError):
                    pool.host_numa_node()

    def test_absent_node(self):
        with patch.dict(os.environ, {"LOCKS_HOST_NUMA_NODE": "10000"}), \
             patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(ValueError):
                pool.host_numa_node()

    def test_memory_only_node(self):
        with patch.dict(os.environ, {"LOCKS_HOST_NUMA_NODE": "1"}), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="\n"):
            with self.assertRaises(ValueError):
                pool.host_numa_node()

    def test_linux_bind_without_cuda_allocation(self):
        node = pool.host_numa_node()
        with patch.object(pool, "_cudart", return_value=None):
            owner = pool._RegisteredPinnedOwner(65536, huge=False)
            self.assertEqual(owner.numa_node, node)
            self.assertFalse(owner.registered)
            ctypes.memset(owner.pointer, 1, owner.size)
            address = format(owner.pointer.value, "x")
            mappings = Path("/proc/self/numa_maps").read_text().splitlines()
            line = next(line for line in mappings if line.split()[0] == address)
            self.assertIn(f"bind:{node}", line)
            self.assertIn(f"N{node}=", line)
            del owner
            gc.collect()


if __name__ == "__main__":
    unittest.main()
