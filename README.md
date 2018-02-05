# am_fft.h

A single-header library for fast fourier transformations.
License: MIT

This is a work in progress!

## Current limitations
- 32bit floating point only
- 1D input must be power-of-two sized
- 2D input must be power-of-two sized and square

## TODO
- Find and implement a good solution for the in-place non-square matrix transpose to get rid of the square 2D input limitation
