"""
bitbrush.py

A novel, high-performance and readable bit manipulation toolkit.

This module introduces a unique approach to "bit brushing" —
a conceptual and practical code to systematically explore, test,
and manipulate bits using expressive primitives.
"""

from typing import Generator

class BitBrush:
    """
    BitBrush: A high-performance utility class for bit-level pattern manipulation.

    This class provides generators and utilities for controlled bit pattern generation,
    transformation, and visualization. The operations include sweeps, toggles,
    mirrors, and analytical functions such as bit counting.
    """

    def __init__(self, width: int = 32):
        """
        Initializes the BitBrush instance with a given bit width.

        Args:
            width (int): Number of bits to operate on. Default is 32.
        """
        self.width = width
        self.mask = (1 << width) - 1
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

    def sweep_ones(self) -> Generator[int, None, None]:
        """
        Generator that yields numbers with a sweeping '1' bit through the width.

        Yields:
            int: An integer with a single bit set, sweeping from LSB to MSB.
        """
        for i in range(self.width):
            yield 1 << i

    def sweep_zeros(self) -> Generator[int, None, None]:
        """
        Generator that yields numbers with all bits set except for a sweeping '0'.

        Yields:
            int: An integer with one bit cleared, sweeping from LSB to MSB.
        """
        for i in range(self.width):
            yield self.mask ^ (1 << i)

    def toggle_sparse(self, step: int = 3) -> Generator[int, None, None]:
        """
        Generator that yields integers with bits toggled sparsely at the given step.

        Args:
            step (int): The distance between toggled bits. Default is 3.

        Yields:
            int: An integer with sparsely distributed 1s.
        """
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
        return result >> (bytes_needed * 8 - self.width)

    def scan_patterns(self) -> Generator[int, None, None]:
        """
        Generator that yields symmetric bit patterns growing from the center outward.

        Yields:
            int: Bit patterns sweeping outward and inward from the center.
        """
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
