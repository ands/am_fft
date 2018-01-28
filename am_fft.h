/*

This file is am_fft.h - a single-header library for limited fast fourier transforms.
Copyright (c) 2018 Andreas Mantler
Distributed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef AM_FFT_H
#define AM_FFT_H


// NOTE: This library is currently limited to FFTs with power-of-two sizes and, in case of 2D dfts, a square shape.


// The complex type { real, imaginary }:
typedef float am_fft_complex_t[2];

// Plans (holds precomputed coefficients to quickly re-process data with the same layout):
typedef struct
{
	float *cos_table;
	float *sin_table;
	unsigned int *twiddle_table;
	unsigned int n;
	unsigned int __pad0;
} am_fft_plan_1d_t;

typedef struct
{
	am_fft_plan_1d_t *x;
	am_fft_plan_1d_t *y;
	am_fft_complex_t *tmp;
	void *__pad0;
} am_fft_plan_2d_t;

#define AM_FFT_FORWARD 0
#define AM_FFT_INVERSE 1

// Functions:
am_fft_plan_1d_t* am_fft_plan_1d(int direction, unsigned int n);
void              am_fft_plan_1d_free(am_fft_plan_1d_t *plan);
void              am_fft_1d(const am_fft_plan_1d_t *plan, const am_fft_complex_t *in, am_fft_complex_t *out);

am_fft_plan_2d_t* am_fft_plan_2d(int direction, unsigned int width, unsigned int height);
void              am_fft_plan_2d_free(am_fft_plan_2d_t *plan);
void              am_fft_2d(const am_fft_plan_2d_t *plan, const am_fft_complex_t *in, am_fft_complex_t *out);

#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef AM_FFT_IMPLEMENTATION
#include <assert.h>

#ifndef AM_FFT_ALLOC
#include <stdlib.h>
#define AM_FFT_ALLOC malloc
#define AM_FFT_FREE free
#endif

am_fft_plan_1d_t* am_fft_plan_1d(int direction, unsigned int n)
{
	unsigned int levels = 0;
	for (unsigned int temp = n; temp > 1; temp >>= 1)
		levels++;
	if (1U << levels != n)
		return 0;
	
	void *mem = AM_FFT_ALLOC(sizeof(am_fft_plan_1d_t) + n * sizeof(unsigned int) + n * sizeof(float));
	am_fft_plan_1d_t *plan = (am_fft_plan_1d_t*)mem;
	plan->twiddle_table = (unsigned int*)(plan + 1);
	plan->cos_table = (float*)(plan->twiddle_table + n);
	plan->sin_table = plan->cos_table + n / 2;
	plan->n = n;
	
	for (unsigned int i = 0; i < n; i++)
	{
		unsigned int j = 0;
		unsigned int bits = i;
		for (unsigned int l = 0; l < levels; l++, bits >>= 1)
			j = (j << 1) | (bits & 1);
		plan->twiddle_table[i] = j;
	}
	
	double angle_step = 2.0 * acos(-1) / (double)n * (direction == AM_FFT_FORWARD ? 1.0 : -1.0);
	for (unsigned int i = 0; i < n / 2; i++)
	{
		double angle = (double)i * angle_step;
		plan->cos_table[i] = (float)cos(angle);
		plan->sin_table[i] = (float)sin(angle);
	}
	
	return plan;
}

void am_fft_plan_1d_free(am_fft_plan_1d_t *plan)
{
	AM_FFT_FREE(plan);
}

void am_fft_1d(const am_fft_plan_1d_t *plan, const am_fft_complex_t *in, am_fft_complex_t *out)
{
	// Twiddle inputs:
	unsigned int n = plan->n;
	unsigned int *twiddle_table = plan->twiddle_table;
	for (unsigned int i = 0; i < n; i++)
	{
		unsigned int j = twiddle_table[i];
		out[j][0] = in[i][0];
		out[j][1] = in[i][1];
	}
	
	// Radix-2 FFT:
	float *cos_table = plan->cos_table;
	float *sin_table = plan->sin_table;
	for (unsigned int half = 1; half < n; half <<= 1)
	{
		unsigned int size = half << 1;
		unsigned int step = n / size;
		for (unsigned int i = 0; i < n; i += size)
		{
			for (unsigned int j = i, k = 0; j < i + half; j++, k += step)
			{
				unsigned int l = j + half;
				am_fft_complex_t c;
				c[0] =  out[l][0] * cos_table[k] + out[l][1] * sin_table[k];
				c[1] = -out[l][0] * sin_table[k] + out[l][1] * cos_table[k];
				out[l][0] = out[j][0] - c[0]; out[j][0] = out[j][0] + c[0];
				out[l][1] = out[j][1] - c[1]; out[j][1] = out[j][1] + c[1];
			}
		}
	}
}

am_fft_plan_2d_t* am_fft_plan_2d(int direction, unsigned int width, unsigned int height)
{
	void *mem = AM_FFT_ALLOC(sizeof(am_fft_plan_2d_t) + sizeof(am_fft_complex_t) * width * height);
	am_fft_plan_2d_t *plan = (am_fft_plan_2d_t*)mem;
	plan->x = am_fft_plan_1d(direction, width);
	plan->y = am_fft_plan_1d(direction, height);
	plan->tmp = (am_fft_complex_t*)(plan + 1);
	return plan;
}

void am_fft_plan_2d_free(am_fft_plan_2d_t *plan)
{
	am_fft_plan_1d_free(plan->x);
	am_fft_plan_1d_free(plan->y);
	AM_FFT_FREE(plan);
}

#define am_fft_block_size 16

static void am_fft_transpose_square(am_fft_complex_t *m, int n)
{
	if (n <= am_fft_block_size)
	{
		for (unsigned int y = 0; y < n; y++)
		{
			for (unsigned int x = 0; x < y; x++)
			{
				am_fft_complex_t tmp = { m[y * n + x][0], m[y * n + x][1] };
				m[y * n + x][0] = m[x * n + y][0];
				m[y * n + x][1] = m[x * n + y][1];
				m[x * n + y][0] = tmp[0];
				m[x * n + y][1] = tmp[1];
			}
		}
	}
	else
	{
		am_fft_complex_t tmp0[am_fft_block_size][am_fft_block_size];
		am_fft_complex_t tmp1[am_fft_block_size][am_fft_block_size];
		for (unsigned int y = 0; y < n; y += am_fft_block_size)
		{
			for (unsigned int x = 0; x < y; x += am_fft_block_size)
			{
				// Read 2 blocks:
				for (unsigned int i = 0; i < am_fft_block_size; i++)
				{
					memcpy(tmp0[i], m[(y + i) * n + x], am_fft_block_size * sizeof(am_fft_complex_t));
					memcpy(tmp1[i], m[(x + i) * n + y], am_fft_block_size * sizeof(am_fft_complex_t));
				}
				// Transpose blocks:
				for (unsigned int by = 0; by < am_fft_block_size; by++)
				{
					for (unsigned int bx = 0; bx < by; bx++)
					{
						am_fft_complex_t t0 = { tmp0[by][bx][0], tmp0[by][bx][1] };
						am_fft_complex_t t1 = { tmp1[by][bx][0], tmp1[by][bx][1] };
						tmp0[by][bx][0] = tmp0[bx][by][0];
						tmp0[by][bx][1] = tmp0[bx][by][1];
						tmp1[by][bx][0] = tmp1[bx][by][0];
						tmp1[by][bx][1] = tmp1[bx][by][1];
						tmp0[bx][by][0] = t0[0];
						tmp0[bx][by][1] = t0[1];
						tmp1[bx][by][0] = t1[0];
						tmp1[bx][by][1] = t1[1];
					}
				}
				// Write swapped blocks:
				for (unsigned int i = 0; i < am_fft_block_size; i++)
				{
					memcpy(m[(y + i) * n + x], tmp1[i], am_fft_block_size * sizeof(am_fft_complex_t));
					memcpy(m[(x + i) * n + y], tmp0[i], am_fft_block_size * sizeof(am_fft_complex_t));
				}
			}

			// Directly transpose blocks on the diagonal:
			unsigned int x = y;
			for (unsigned int by = 0; by < am_fft_block_size; by++)
			{
				for (unsigned int bx = 0; bx < by; bx++)
				{
					am_fft_complex_t tmp = { m[(y + by) * n + x + bx][0], m[(y + by) * n + x + bx][1] };
					m[(y + by) * n + x + bx][0] = m[(x + bx) * n + y + by][0];
					m[(y + by) * n + x + bx][1] = m[(x + bx) * n + y + by][1];
					m[(x + bx) * n + y + by][0] = tmp[0];
					m[(x + bx) * n + y + by][1] = tmp[1];
				}
			}
		}
	}
}

void am_fft_2d(const am_fft_plan_2d_t *plan, const am_fft_complex_t *in, am_fft_complex_t *out)
{
	am_fft_complex_t *tmp = plan->tmp;
	const am_fft_complex_t *ins[2] = { in, tmp };
	am_fft_complex_t *outs[2] = { tmp, out };
	am_fft_plan_1d_t *plans[2] = { plan->x, plan->y };
	
	for (unsigned int i = 0; i < 2; i++)
	{
		const am_fft_complex_t *current_in = ins[i];
		am_fft_complex_t *current_out = outs[i];
		unsigned int n = plans[i]->n;

		for (unsigned int y = 0; y < n; y++)
			am_fft_1d(plans[i], current_in + y * n, current_out + y * n);

		if (plans[0]->n == plans[1]->n)
		{
			am_fft_transpose_square(current_out, n);
		}
		else
		{
			// TODO: Transpose non-square matrices!
			assert(0);
		}
	}
}

#endif