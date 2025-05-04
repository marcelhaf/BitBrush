"""
bitbrush.py

A novel, high-performance and readable bit manipulation toolkit with optional NumPy acceleration.

This module introduces a unique approach to "bit brushing" —
conceptual and practical code to systematically explore, test,
and manipulate bits using expressive primitives. It supports
both pure-Python and NumPy-backed operations for large-scale
bit pattern generation.
"""

import numpy as np
from typing import Generator, Union, Literal


class BitBrush:
    """
    BitBrush: A high-performance utility for bit-level pattern manipulation.

    Supports both pure-Python and NumPy backends for key operations.
    """

    def __init__(self, width: int = 32, backend: Literal['python', 'numpy'] = 'python'):
        """
        Initialize BitBrush with a given bit width and backend.

        Args:
            width (int): Number of bits to operate on. Default is 32.
            backend (str): 'python' or 'numpy' implementation. Default is 'python'.
        """
        self.width = width
        self.mask = (1 << width) - 1
        self.backend = backend
        self._mirror_lut = self._build_mirror_lut()

    def _build_mirror_lut(self) -> list[int]:
        """
        Build a lookup table for 8-bit reversed values.

        Returns:
            list[int]: 256-element LUT for byte-wise bit reversal.
        """
        lut = [0] * 256
        for i in range(256):
            b, rev = i, 0
            for _ in range(8):
                rev = (rev << 1) | (b & 1)
                b >>= 1
            lut[i] = rev
        return lut

    def sweep_ones(self) -> Union[Generator[int, None, None], np.ndarray]:
        """Generate values with a single '1' sweeping from LSB to MSB."""
        if self.backend == 'numpy':
            return np.left_shift(np.uint64(1), np.arange(self.width, dtype=np.uint64))
        return (1 << i for i in range(self.width))

    def sweep_zeros(self) -> Union[Generator[int, None, None], np.ndarray]:
        """Generate values with all bits set except one '0' sweeping."""
        if self.backend == 'numpy':
            ones = np.left_shift(np.uint64(1), np.arange(self.width, dtype=np.uint64))
            return np.bitwise_xor(self.mask, ones)
        return (self.mask ^ (1 << i) for i in range(self.width))

    def toggle_sparse(self, step: int = 3) -> Union[Generator[int, None, None], np.ndarray]:
        """Generate values with bits toggled sparsely at a fixed step."""
        if self.backend == 'numpy':
            indices = np.arange(0, self.width, step, dtype=np.uint64)
            arr = np.zeros(indices.shape, dtype=np.uint64)
            for idx in range(indices.size):
                bit = np.left_shift(np.uint64(1), indices[idx])
                arr[idx] = arr[idx - 1] | bit if idx > 0 else bit
            return arr
        val = 0
        for i in range(0, self.width, step):
            val |= 1 << i
            yield val

    def mirror_mask(self, value: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Reverse bit order within the configured width."""
        bytes_needed = (self.width + 7) // 8
        if self.backend == 'numpy' and isinstance(value, np.ndarray):
            result = np.zeros_like(value, dtype=np.uint64)
            for i in range(bytes_needed):
                shift = np.right_shift(value, np.uint64(i * 8)) & np.uint64(0xFF)
                rev = np.take(self._mirror_lut, shift)
                result |= np.left_shift(rev.astype(np.uint64), np.uint64((bytes_needed - 1 - i) * 8))
            return np.right_shift(result, np.uint64(bytes_needed * 8 - self.width))
        result = 0
        for i in range(bytes_needed):
            byte = (value >> (i * 8)) & 0xFF
            rev = self._mirror_lut[byte]
            result |= rev << ((bytes_needed - 1 - i) * 8)
        return result >> (bytes_needed * 8 - self.width)

    def scan_patterns(self) -> Union[Generator[int, None, None], np.ndarray]:
        """Generate symmetric patterns expanding from center outward."""
        center = self.width // 2
        if self.backend == 'numpy':
            radii = np.arange(center + 1, dtype=np.uint64)
            left = np.left_shift(np.uint64(1), center - radii)
            right = np.where(center + radii < self.width,
                             np.left_shift(np.uint64(1), center + radii),
                             np.uint64(0))
            return np.bitwise_or(left, right)
        for radius in range(center + 1):
            left = 1 << (center - radius)
            right = 1 << (center + radius) if center + radius < self.width else 0
            yield left | right

    def count_ones(self, value: int) -> int:
        """Count number of set bits in an integer."""
        return bin(int(value) & self.mask).count('1')

    def visualize(self, value: int) -> str:
        """Return binary string of value padded to width."""
        return format(int(value) & self.mask, f'0{self.width}b')
