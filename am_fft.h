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

// Plans (hold precomputed coefficients to quickly re-process data with the same layout):
typedef struct am_fft_plan_1d_ am_fft_plan_1d_t;
typedef struct am_fft_plan_2d_ am_fft_plan_2d_t;

// FFT directions:
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
#include <math.h>

#ifndef AM_FFT_ALLOC
#include <stdlib.h>
#define AM_FFT_ALLOC malloc
#define AM_FFT_FREE free
#endif

#ifndef AM_FFT_NO_SSE2
#include <emmintrin.h>
#endif

struct am_fft_plan_1d_
{
	float *cos_table;
	float *sin_table;
	unsigned int *twiddle_table;
	unsigned int n;
	int direction;
};

struct am_fft_plan_2d_
{
	am_fft_plan_1d_t *x;
	am_fft_plan_1d_t *y;
	am_fft_complex_t *tmp;
	void *__pad0;
};

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
	plan->direction = direction;
	
	for (unsigned int i = 0; i < n; i++)
	{
		unsigned int j = 0;
		unsigned int bits = i;
		for (unsigned int l = 0; l < levels; l++, bits >>= 1)
			j = (j << 1) | (bits & 1);
		plan->twiddle_table[i] = j;
	}
	
	const double pi = 3.14159265358979323846; // Don't rely on M_PI being defined
	const double angle_step = 2.0 * pi / (double)n * (direction == AM_FFT_FORWARD ? 1.0 : -1.0);
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

	// First passes:
	if (n >= 2)
	{
		#ifdef AM_FFT_NO_SSE2
		for (unsigned int i = 0; i < n; i += 2)
		{
			float ar = out[i + 0][0];
			float ai = out[i + 0][1];
			float br = out[i + 1][0];
			float bi = out[i + 1][1];
			out[i + 0][0] = ar + br;
			out[i + 0][1] = ai + bi;
			out[i + 1][0] = ar - br;
			out[i + 1][1] = ai - bi;
		}
		#else
		const __m128 ppnn = _mm_setr_ps(1.0f, 1.0f, -1.0f, -1.0f);
		for (unsigned int i = 0; i < n; i += 2)
		{
			__m128 aribri = _mm_loadu_ps(&out[i + 0][0]);
			__m128 ariari = _mm_shuffle_ps(aribri, aribri, _MM_SHUFFLE(1, 0, 1, 0));
			__m128 bribri = _mm_shuffle_ps(aribri, aribri, _MM_SHUFFLE(3, 2, 3, 2));
			__m128 result = _mm_add_ps(ariari, _mm_mul_ps(bribri, ppnn));
			_mm_storeu_ps(&out[i + 0][0], result);
		}
		#endif
	}

	if (n >= 4)
	{
		#ifdef AM_FFT_NO_SSE2
		if (plan->direction == AM_FFT_FORWARD)
		{
			
			for (unsigned int i = 0; i < n; i += 4)
			{
				float ar = out[i + 0][0];
				float ai = out[i + 0][1];
				float br = out[i + 1][0];
				float bi = out[i + 1][1];
				float cr = out[i + 2][0];
				float ci = out[i + 2][1];
				float dr = out[i + 3][0];
				float di = out[i + 3][1];
				out[i + 0][0] = ar + cr;
				out[i + 0][1] = ai + ci;
				out[i + 1][0] = br + di;
				out[i + 1][1] = bi - dr;
				out[i + 2][0] = ar - cr;
				out[i + 2][1] = ai - ci;
				out[i + 3][0] = br - di;
				out[i + 3][1] = bi + dr;
			}
		}
		else
		{
			for (unsigned int i = 0; i < n; i += 4)
			{
				float ar = out[i + 0][0];
				float ai = out[i + 0][1];
				float br = out[i + 1][0];
				float bi = out[i + 1][1];
				float cr = out[i + 2][0];
				float ci = out[i + 2][1];
				float dr = out[i + 3][0];
				float di = out[i + 3][1];
				out[i + 0][0] = ar + cr;
				out[i + 0][1] = ai + ci;
				out[i + 1][0] = br - di;
				out[i + 1][1] = bi + dr;
				out[i + 2][0] = ar - cr;
				out[i + 2][1] = ai - ci;
				out[i + 3][0] = br + di;
				out[i + 3][1] = bi - dr;
			}
		}
		#else
		const __m128 flip = (plan->direction == AM_FFT_FORWARD) ? _mm_setr_ps(1.0f, 1.0f, 1.0f, -1.0f) : _mm_setr_ps(1.0f, 1.0f, -1.0f, 1.0f);
		for (unsigned int i = 0; i < n; i += 4)
		{
			__m128 aribri = _mm_loadu_ps(&out[i + 0][0]);
			__m128 cridri = _mm_loadu_ps(&out[i + 2][0]);
			__m128 cridir = _mm_shuffle_ps(cridri, cridri, _MM_SHUFFLE(2, 3, 1, 0));
			__m128 flipped = _mm_mul_ps(cridir, flip);
			__m128 result0 = _mm_add_ps(aribri, flipped);
			__m128 result1 = _mm_sub_ps(aribri, flipped);
			_mm_storeu_ps(&out[i + 0][0], result0);
			_mm_storeu_ps(&out[i + 2][0], result1);
		}
		#endif
	}

	// Remaining passes of the radix-2 FFT:
	float *cos_table = plan->cos_table;
	float *sin_table = plan->sin_table;
	am_fft_complex_t c;
	for (unsigned int half = 4; half < n; half <<= 1)
	{
		unsigned int size = half << 1;
		unsigned int step = n / size;
		for (unsigned int i = 0; i < n; i += size)
		{
			for (unsigned int j = i, k = 0; j < i + half; j++, k += step)
			{
				unsigned int l = j + half;
				c[0] =  out[l][0] * cos_table[k] + out[l][1] * sin_table[k];
				c[1] = -out[l][0] * sin_table[k] + out[l][1] * cos_table[k];
				out[l][0] = out[j][0] - c[0];
				out[l][1] = out[j][1] - c[1];
				out[j][0] = out[j][0] + c[0];
				out[j][1] = out[j][1] + c[1];
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

static void am_fft_transpose_square(am_fft_complex_t *m, unsigned int n)
{
#define am_fft_block_size 16
	am_fft_complex_t block0[am_fft_block_size][am_fft_block_size];
	am_fft_complex_t block1[am_fft_block_size][am_fft_block_size];
	am_fft_complex_t tmp, tmp0, tmp1;
	
	if (n <= am_fft_block_size)
	{
		for (unsigned int y = 0; y < n; y++)
		{
			for (unsigned int x = 0; x < y; x++)
			{
				tmp[0] = m[y * n + x][0];
				tmp[1] = m[y * n + x][1];
				m[y * n + x][0] = m[x * n + y][0];
				m[y * n + x][1] = m[x * n + y][1];
				m[x * n + y][0] = tmp[0];
				m[x * n + y][1] = tmp[1];
			}
		}
	}
	else
	{
		for (unsigned int y = 0; y < n; y += am_fft_block_size)
		{
			for (unsigned int x = 0; x < y; x += am_fft_block_size)
			{
				// Read 2 blocks:
				for (unsigned int i = 0; i < am_fft_block_size; i++)
				{
					memcpy(block0[i], m[(y + i) * n + x], am_fft_block_size * sizeof(am_fft_complex_t));
					memcpy(block1[i], m[(x + i) * n + y], am_fft_block_size * sizeof(am_fft_complex_t));
				}
				// Transpose blocks:
				for (unsigned int by = 0; by < am_fft_block_size; by++)
				{
					for (unsigned int bx = 0; bx < by; bx++)
					{
						tmp0[0] = block0[by][bx][0];
						tmp0[1] = block0[by][bx][1];
						tmp1[0] = block1[by][bx][0];
						tmp1[1] = block1[by][bx][1];
						block0[by][bx][0] = block0[bx][by][0];
						block0[by][bx][1] = block0[bx][by][1];
						block1[by][bx][0] = block1[bx][by][0];
						block1[by][bx][1] = block1[bx][by][1];
						block0[bx][by][0] = tmp0[0];
						block0[bx][by][1] = tmp0[1];
						block1[bx][by][0] = tmp1[0];
						block1[bx][by][1] = tmp1[1];
					}
				}
				// Write swapped blocks:
				for (unsigned int i = 0; i < am_fft_block_size; i++)
				{
					memcpy(m[(y + i) * n + x], block1[i], am_fft_block_size * sizeof(am_fft_complex_t));
					memcpy(m[(x + i) * n + y], block0[i], am_fft_block_size * sizeof(am_fft_complex_t));
				}
			}

			// Directly transpose blocks on the diagonal:
			unsigned int x = y;
			for (unsigned int by = 0; by < am_fft_block_size; by++)
			{
				for (unsigned int bx = 0; bx < by; bx++)
				{
					tmp[0] = m[(y + by) * n + x + bx][0];
					tmp[1] = m[(y + by) * n + x + bx][1];
					m[(y + by) * n + x + bx][0] = m[(x + bx) * n + y + by][0];
					m[(y + by) * n + x + bx][1] = m[(x + bx) * n + y + by][1];
					m[(x + bx) * n + y + by][0] = tmp[0];
					m[(x + bx) * n + y + by][1] = tmp[1];
				}
			}
		}
	}
#undef am_fft_block_size
}

void am_fft_2d(const am_fft_plan_2d_t *plan, const am_fft_complex_t *in, am_fft_complex_t *out)
{
	const am_fft_complex_t *ins[2];
	am_fft_complex_t *outs[2];
	am_fft_plan_1d_t *plans[2];
	ins[0] =        in; outs[0] = plan->tmp; plans[0] = plan->x;
	ins[1] = plan->tmp; outs[1] =       out; plans[1] = plan->y;
	
	for (unsigned int i = 0; i < 2; i++)
	{
		unsigned int n = plans[i]->n;

		for (unsigned int y = 0; y < n; y++)
			am_fft_1d(plans[i], ins[i] + y * n, outs[i] + y * n);

		if (plans[0]->n == plans[1]->n)
			am_fft_transpose_square(outs[i], n);
		else
		{
			// TODO: Transpose non-square matrices!
			assert(0);
		}
	}
}

#endif