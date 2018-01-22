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


// The complex type:
typedef float am_fft_complex_t[2];


// Plans (hold precomputed data to re-process data with the same layout):
typedef struct
{
	float *cos_table;
	float *sin_table;
	unsigned int levels;
	unsigned int n;
	am_fft_complex_t *tmp;
} am_fft_plan_t;


// Precomputation interface:
am_fft_plan_t* am_fft_plan(unsigned int n);
void am_fft_plan_free(am_fft_plan_t *plan);


#define AM_FFT_FORWARD 0
#define AM_FFT_INVERSE 1


// FFT interface:
void am_fft_1d(const am_fft_plan_t *plan, int direction, const am_fft_complex_t *in, am_fft_complex_t *out);
void am_fft_2d(const am_fft_plan_t *plan, int direction, const am_fft_complex_t *in, am_fft_complex_t *out);


#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef AM_FFT_IMPLEMENTATION

#ifndef AM_FFT_ALLOC
#include <stdlib.h>
#define AM_FFT_ALLOC malloc
#define AM_FFT_FREE free
#endif

am_fft_plan_t* am_fft_plan(unsigned int n)
{
	unsigned int levels = 0;
	for (unsigned int temp = n; temp > 1; temp >>= 1)
		levels++;
	if (1U << levels != n)
		return 0;
	
	am_fft_plan_t *plan = (am_fft_plan_t*)AM_FFT_ALLOC(sizeof(am_fft_plan_t));
	plan->cos_table = (float*)AM_FFT_ALLOC(n * sizeof(float));
	plan->sin_table = plan->cos_table + n / 2;
	plan->levels = levels;
	plan->n = n;
	double pi = acos(-1);
	for (unsigned int i = 0; i < n / 2; i++)
	{
		double angle = 2.0 * pi * (double)i / (double)n;
		plan->cos_table[i] = (float)cos(angle);
		plan->sin_table[i] = (float)sin(angle);
	}
	
	plan->tmp = (am_fft_complex_t*)AM_FFT_ALLOC(sizeof(am_fft_complex_t) * n * n);
	
	return plan;
}

void am_fft_plan_free(am_fft_plan_t *plan)
{
	AM_FFT_FREE(plan->cos_table);
	AM_FFT_FREE(plan->tmp);
	AM_FFT_FREE(plan);
}

void am_fft_1d(const am_fft_plan_t *plan, int direction, const am_fft_complex_t *in, am_fft_complex_t *out)
{
	// Twiddle inputs:
	unsigned int n = plan->n;
	for (unsigned int i = 0; i < n; i++)
	{
		unsigned int j = 0;
		unsigned int bits = i;
		for (unsigned int l = 0; l < plan->levels; l++, bits >>= 1)
			j = (j << 1) | (bits & 1);
		out[j][0] = in[i][0];
		out[j][1] = in[i][1];
	}
	
	// Radix-2 FFT:
	float *cos_table = plan->cos_table;
	float *sin_table = plan->sin_table;
	
	if (direction == AM_FFT_INVERSE)
	{
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
					c[0] = -out[l][1] * sin_table[k] + out[l][0] * cos_table[k];
					c[1] =  out[l][1] * cos_table[k] + out[l][0] * sin_table[k];
					out[l][0] = out[j][0] - c[0];
					out[l][1] = out[j][1] - c[1];
					out[j][0] = out[j][0] + c[0];
					out[j][1] = out[j][1] + c[1];
				}
			}
		}
	}
	else
	{
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
					out[l][0] = out[j][0] - c[0];
					out[l][1] = out[j][1] - c[1];
					out[j][0] = out[j][0] + c[0];
					out[j][1] = out[j][1] + c[1];
				}
			}
		}
	}
}

void am_fft_2d(const am_fft_plan_t *plan, int direction, const am_fft_complex_t *in, am_fft_complex_t *out)
{
	unsigned int n = plan->n;
	am_fft_complex_t *tmp = plan->tmp;
	const am_fft_complex_t *ins[2] = { in, tmp };
	am_fft_complex_t *outs[2] = { tmp, out };
	
	for (unsigned int i = 0; i < 2; i++)
	{
		const am_fft_complex_t *current_in = ins[i];
		am_fft_complex_t *current_out = outs[i];

		for (unsigned int y = 0; y < n; y++)
			am_fft_1d(plan, direction, current_in + y * n, current_out + y * n);

		for (unsigned int y = 0; y < n; y++)
		{
			for (unsigned int x = 0; x < y; x++)
			{
				am_fft_complex_t tmp = { current_out[y * n + x][0], current_out[y * n + x][1] };
				current_out[y * n + x][0] = current_out[x * n + y][0];
				current_out[y * n + x][1] = current_out[x * n + y][1];
				current_out[x * n + y][0] = tmp[0];
				current_out[x * n + y][1] = tmp[1];
			}
		}
	}
}

#endif