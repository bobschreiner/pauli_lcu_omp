# Copyright 2024 RIVERLANE LTD
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


def _uint32_t_deinterleave_lowuint32(word_val):
    """
    deinterleave https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2018/01/08/interleave.c
    """
    word_val &= 0x5555555555555555
    word_val = (word_val ^ (word_val >> 1)) & 0x3333333333333333
    word_val = (word_val ^ (word_val >> 2)) & 0x0f0f0f0f0f0f0f0f
    word_val = (word_val ^ (word_val >> 4)) & 0x00ff00ff00ff00ff
    word_val = (word_val ^ (word_val >> 8)) & 0x0000ffff0000ffff
    word_val = (word_val ^ (word_val >> 16)) & 0x00000000ffffffff
    return word_val


def _interleave_uint32_with_zeros(val):
    """
    // interleave bits with zeros, see https://lemire.me/blog/2018/01/08/how-fast-can-you-bit-interleave-32-bit-integers/
    """

    word = val
    word = (word ^ (word << 16)) & 0x0000ffff0000ffff
    word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff
    word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f
    word = (word ^ (word << 2)) & 0x3333333333333333
    word = (word ^ (word << 1)) & 0x5555555555555555
    return word


def interleave(i, j):
    return _interleave_uint32_with_zeros(i) | (_interleave_uint32_with_zeros(j) << 1)


def deinterleave(word: int):
    i = _uint32_t_deinterleave_lowuint32(word)
    j = _uint32_t_deinterleave_lowuint32(word >> 1)
    return (i, j)


def lex_indices(pauli_id: int):
    """

    Parameters
    ----------
    pauli_id : int
        id corresponding to a pauli string in lexicographical order

    Returns
    -------
    tuple[int, int]
        indices to access the same string in pauli_strings
    """
    i, j = deinterleave(pauli_id)
    return i ^ j, j


def random_matrix(num_qubits: int, seed: int = 0) -> np.ndarray:
    """Generate a random complex matrix of size 2^n x 2^n."""
    rng = np.random.default_rng(seed)
    size = 1 << num_qubits
    matrix = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return matrix
