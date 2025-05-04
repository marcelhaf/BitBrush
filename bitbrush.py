"""
bitbrush.py

A novel, high-performance and readable bit manipulation toolkit with optional NumPy acceleration.

This module introduces a unique approach to "bit brushing" —
conceptual and practical code to systematically explore, test,
and manipulate bits using expressive primitives. It now supports
both pure-Python and NumPy-backed operations for large-scale
bit pattern generation.
"""

import numpy as np
from typing import Generator, Union, Literal

class BitBrush:
    """
    BitBrush: A high-performance utility class for bit-level pattern manipulation.

    This class provides generators and utilities for controlled bit pattern generation,
    transformation, and visualization. Operations include sweeps, toggles, mirrors,
    and analytical functions such as bit counting. You can choose a pure-Python
    backend or a NumPy-backed backend for vectorized performance.
    """

    def __init__(self, width: int = 32, backend: Literal['python', 'numpy'] = 'python'):
        """
        Initializes the BitBrush instance with a given bit width and backend.

        Args:
            width (int): Number of bits to operate on. Default is 32.
            backend (str): 'python' for generator-based implementation, or 'numpy'
                           for vectorized NumPy arrays. Default is 'python'.
        """
        self.width = width
        self.mask = (1 << width) - 1
        self.backend = backend
        self._mirror_lut = self._build_mirror_lut()

    def _build_mirror_lut(self) -> list[int]:
        """
        Builds a lookup table (LUT) for fast 8-bit reversed values.

        Returns:
            list[int]: A list of 256 integers, each containing the reversed bits of its index.
        """
        lut = [0] * 256
        for i in range(256):
            b = i
            rev = 0
            for _ in range(8):
                rev = (rev << 1) | (b & 1)
                b >>= 1
            lut[i] = rev
        return lut

    def sweep_ones(self) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generates numbers with a single '1' bit sweeping from LSB to MSB.

        Returns:
            Generator[int] or np.ndarray: Sweeping bit patterns.
        """
        if self.backend == 'numpy':
            # Vectorized left-shift: produces array [1<<0, 1<<1, ..., 1<<(width-1)]
            return np.left_shift(np.uint64(1), np.arange(self.width, dtype=np.uint64))
        # Pure-Python generator
        return (1 << i for i in range(self.width))

    def sweep_zeros(self) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generates numbers with all bits set except for a sweeping '0'.

        Returns:
            Generator[int] or np.ndarray: Bit patterns with a sweeping '0'.
        """
        if self.backend == 'numpy':
            # mask ^ (1<<i) for all i
            return self.mask ^ np.left_shift(np.uint64(1), np.arange(self.width, dtype=np.uint64))
        return (self.mask ^ (1 << i) for i in range(self.width))

    def toggle_sparse(self, step: int = 3) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generates integers with sparsely toggled bits using a given step.

        Args:
            step (int): Step size between toggled bits. Default is 3.

        Returns:
            Generator[int] or np.ndarray: Bit patterns with sparsely set bits.
        """
        if self.backend == 'numpy':
            # Build array of indices to set
            indices = np.arange(0, self.width, step, dtype=np.uint64)
            # Initialize an array to hold cumulative OR results
            result = np.zeros(indices.shape, dtype=np.uint64)
            for idx in range(indices.size):
                bit = np.uint64(1) << indices[idx]
                result[idx] = bit if idx == 0 else result[idx-1] | bit
            return result
        # Python generator fallback
        val = 0
        for i in range(0, self.width, step):
            val |= 1 << i
            yield val

    def mirror_mask(self, value: int) -> int:
        """
        Mirrors (reverses) the bit pattern of the given value within the specified width.

        This optimized version uses a precomputed 8-bit lookup table
        for fast bit reversal in blocks.

        Args:
            value (int): The integer to mirror.

        Returns:
            int: The mirrored bit pattern as an integer.
        """
        result = 0
        bytes_needed = (self.width + 7) // 8
        for i in range(bytes_needed):
            byte = (value >> (i * 8)) & 0xFF
            mirrored_byte = self._mirror_lut[byte]
            shift = (bytes_needed - 1 - i) * 8
            result |= mirrored_byte << shift
        # Align down to width
        return result >> (bytes_needed * 8 - self.width)

    def scan_patterns(self) -> Union[Generator[int, None, None], np.ndarray]:
        """
        Generates symmetric bit patterns growing from the center outward.

        Returns:
            Generator[int] or np.ndarray: Patterns sweeping from center to edges.
        """
        if self.backend == 'numpy':
            center = self.width // 2
            radii = np.arange(center + 1, dtype=np.int64)
            left = np.left_shift(np.uint64(1), center - radii)
            right = np.where(center + radii < self.width,
                             np.left_shift(np.uint64(1), center + radii),
                             np.uint64(0))
            return left | right
        center = self.width // 2
        for radius in range(center + 1):
            left = 1 << (center - radius)
            right = 1 << (center + radius) if center + radius < self.width else 0
            yield left | right

    def count_ones(self, value: int) -> int:
        """
        Counts the number of 1 bits in the given value.

        Args:
            value (int): The integer to analyze.

        Returns:
            int: Number of bits set to 1.
        """
        return bin(value & self.mask).count('1')

    def visualize(self, value: int) -> str:
        """
        Returns a binary string representation of the value, padded to width.

        Args:
            value (int): The integer to convert.

        Returns:
            str: A binary string with leading zeros up to the configured width.
        """
        return f'{value & self.mask:0{self.width}b}'
