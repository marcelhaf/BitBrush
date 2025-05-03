# BitBrush

**BitBrush** is a novel, high-performance and readable toolkit for bit-level manipulation, pattern generation, and binary data exploration in Python.

## ✨ Overview

BitBrush introduces a unique and expressive framework to "brush" bits — a metaphor for systematic, programmatic exploration of binary patterns. It is designed for testing, teaching, and benchmarking bitwise logic, offering elegant primitives for controlled manipulation.

---

## 🚀 Features

- Fast and efficient bit pattern generators
- Optimized bit mirroring using lookup tables
- Customizable bit width (e.g., 8, 16, 32, 64 bits)
- Expressive visualization in binary format
- Ideal for simulations, hardware modeling, test cases, and educational tools

---

## 📦 Installation

Simply copy `bitbrush.py` into your Python project.

---

## 🧠 How It Works

### Core Class

```python
BitBrush(width=32)
```

Initializes a bit manipulator with a specified bit width. Internally, it maintains a bitmask and a lookup table for optimized operations.

---

## 🧰 Key Functions

| Function         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `sweep_ones()`   | Generates integers with a sweeping `1` from LSB to MSB                      |
| `sweep_zeros()`  | Like `sweep_ones()`, but bits are all `1` except for a sweeping `0`         |
| `toggle_sparse()`| Sets bits sparsely across the range using a defined step                    |
| `mirror_mask()`  | Reverses bit order across the full width using a fast lookup table          |
| `scan_patterns()`| Symmetrically expands bits from the center to edges                         |
| `count_ones()`   | Counts the number of `1`s in a binary pattern                               |
| `visualize()`    | Returns a formatted binary string of the pattern                            |

---
---

## 📊 Example

```python
from bitbrush import BitBrush

bb = BitBrush(16)
for val in bb.sweep_ones():
    print(bb.visualize(val))
```

---

## 📁 Project Structure

```
bitbrush/
├── bitbrush.py
├── bitbrush_demo.py
└── bitbrush_demo.ipynb
```
---