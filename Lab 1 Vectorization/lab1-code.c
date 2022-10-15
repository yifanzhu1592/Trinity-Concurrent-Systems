//
// CSU33014 Lab 1
//
// Please examine version each of the following routines with names
// starting lab1. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics.
// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".
#include <immintrin.h>

#include <stdio.h>

#include "lab1-code.h"

/**************** routine 0 *******************/
// Here is an example routine that should be vectorized
void lab1_routine0(float * restrict a, float * restrict b,

  float * restrict c) {
  for (int i = 0; i < 1024; i++) {

    a[i] = b[i] * c[i];
  }
}
// here is a vectorized solution for the example above
void lab1_vectorized0(float * restrict a, float * restrict b,

  float * restrict c) {
  __m128 a4, b4, c4;

  for (int i = 0; i < 1024; i = i + 4) {

    b4 = _mm_loadu_ps( & b[i]);

    c4 = _mm_loadu_ps( & c[i]);

    a4 = _mm_mul_ps(b4, c4);

    _mm_storeu_ps( & a[i], a4);
  }
}
/***************** routine 1 *********************/
// in the following, size can have any positive value
float lab1_routine1(float * restrict a, float * restrict b,

  int size) {
  float sum = 0.0;

  for (int i = 0; i < size; i++) {

    sum = sum + a[i] * b[i];
  }
  return sum;
}
// insert vectorized code for routine1 here
float lab1_vectorized1(float * restrict a, float * restrict b,

  int size) {
  // replace the following code with vectorized code
  __m128 sum4 = _mm_setzero_ps();
  int remainder = size % 4;
  int i;
  for (i = 0; i < size - remainder; i = i + 4) {

    __m128 a4 = _mm_loadu_ps( & (a[i]));

    __m128 b4 = _mm_loadu_ps( & (b[i]));

    __m128 c4 = _mm_mul_ps(a4, b4);

    sum4 = _mm_add_ps(sum4, c4);
  }

  float temp[4];
  _mm_storeu_ps(temp, sum4);
  float sum = temp[0] + temp[1] + temp[2] + temp[3];

  for (; i < size; i++) {

    sum = sum + a[i] * b[i];
  }

  return sum;
}
/******************* routine 2 ***********************/
// in the following, size can have any positive value
void lab1_routine2(float * restrict a, float * restrict b, int size) {
  for (int i = 0; i < size; i++) {

    a[i] = 1 - (1.0 / (b[i] + 1.0));
  }
}
// in the following, size can have any positive value
void lab1_vectorized2(float * restrict a, float * restrict b, int size) {
  // replace the following code with vectorized code
  __m128 one4 = _mm_set1_ps(1);
  int remainder = size % 4;
  int i;
  for (i = 0; i < size - remainder; i = i + 4) {

    __m128 b4 = _mm_loadu_ps( & (b[i]));

    _mm_storeu_ps( & a[i], _mm_sub_ps(one4, _mm_div_ps(one4, _mm_add_ps(b4, one4))));
  }

  for (; i < size; i++) {

    a[i] = 1 - (1.0 / (b[i] + 1.0));
  }
}
/******************** routine 3 ************************/
// in the following, size can have any positive value
void lab1_routine3(float * restrict a, float * restrict b, int size) {
  for (int i = 0; i < size; i++) {

    if (a[i] < 0.0) {

      a[i] = b[i];

    }
  }
}
// in the following, size can have any positive value
void lab1_vectorized3(float * restrict a, float * restrict b, int size) {
  // replace the following code with vectorized code
  __m128 zero4 = _mm_setzero_ps();
  int remainder = size % 4;
  int i;
  for (i = 0; i < size - remainder; i = i + 4) {

    __m128 a4 = _mm_loadu_ps( & (a[i]));

    __m128 b4 = _mm_loadu_ps( & (b[i]));

    __m128 part_a, part_b, mask_a, mask_b;

    mask_b = _mm_cmplt_ps(a4, zero4);

    part_b = _mm_and_ps(b4, mask_b);

    mask_a = _mm_cmpge_ps(a4, zero4);

    part_a = _mm_and_ps(a4, mask_a);

    _mm_storeu_ps( & a[i], _mm_or_ps(part_a, part_b));
  }

  for (; i < size; i++) {

    if (a[i] < 0.0) {

      a[i] = b[i];

    }
  }
}
/********************* routine 4 ***********************/
// hint: one way to vectorize the following code might use
// vector shuffle operations
void lab1_routine4(float * restrict a, float * restrict b,

  float * restrict c) {
  for (int i = 0; i < 2048; i = i + 2) {

    a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];

    a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];
  }
}
void lab1_vectorized4(float * restrict a, float * restrict b,

  float * restrict c) {
  // replace the following code with vectorized code
  __m128 mask_negative = _mm_set_ps(1, -1, 1, -1);

  for (int i = 0; i < 2048; i = i + 4) {

    __m128 b4 = _mm_loadu_ps( & b[i]);

    __m128 c4 = _mm_loadu_ps( & c[i]);

    __m128 shuf_b_first = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(2, 2, 0, 0)); // b[i + 2], b[i + 2], b[i], b[i]

    __m128 product_first = _mm_mul_ps(shuf_b_first, c4); // b[i + 2] * c[i + 3], b[i + 2] * c[i + 2], b[i] * c[i + 1], b[i] * c[i]

    __m128 shuf_b_second = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(3, 3, 1, 1)); // b[i + 3], b[i + 3], b[i + 1], b[i + 1]

    __m128 shuf_c_second = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2, 3, 0, 1)); // c[i + 2], c[i + 3], c[i], c[i + 1]

    __m128 product_second_positive = _mm_mul_ps(shuf_b_second, shuf_c_second); // b[i + 3] * c[i + 2], b[i + 3] * c[i + 3], b[i + 1] * c[i], b
    [i + 1] * c[i + 1]

    __m128 product_second = _mm_mul_ps(product_second_positive, mask_negative); // b[i + 3] * c[i + 2], -b[i + 3] * c[i + 3], b[i + 1] * c[i], 
    -
    b[i + 1] * c[i + 1]

    __m128 sum = _mm_add_ps(product_first, product_second);

    _mm_storeu_ps( & a[i], sum);
  }
}
/********************* routine 5 ***********************/
// in the following, size can have any positive value
void lab1_routine5(unsigned char * restrict a,

  unsigned char * restrict b, int size) {
  for (int i = 0; i < size; i++) {

    a[i] = b[i];
  }
}
void lab1_vectorized5(unsigned char * restrict a,

  unsigned char * restrict b, int size) {
  // replace the following code with vectorized code
  int remainder = size % 16;
  int i;
  for (i = 0; i < size - remainder; i = i + 16) {

    __m128i b4 = _mm_loadu_si128( & (b[i]));

    _mm_storeu_si128( & a[i], b4);
  }

  for (; i < size; i++) {

    a[i] = b[i];
  }
}
/********************* routine 6 ***********************/
void lab1_routine6(float * restrict a, float * restrict b,

  float * restrict c) {
  a[0] = 0.0;
  for (int i = 1; i < 1023; i++) {

    float sum = 0.0;

    for (int j = 0; j < 3; j++) {

      sum = sum + b[i + j - 1] * c[j];

    }

    a[i] = sum;
  }
  a[1023] = 0.0;
}
void lab1_vectorized6(float * restrict a, float * restrict b,

  float * restrict c) {
  // replace the following code with vectorized code
  a[0] = 0.0;
  for (int i = 1; i < 1023; i++) {

    __m128 sum3 = _mm_setzero_ps();

    __m128 b3 = _mm_loadu_ps( & (b[i - 1]));

    __m128 c3 = _mm_loadu_ps( & (c[0]));

    __m128 d3 = _mm_mul_ps(b3, c3);

    sum3 = _mm_add_ps(sum3, d3);

    float temp[3];

    _mm_storeu_ps(temp, sum3);

    a[i] = temp[0] + temp[1] + temp[2];
  }
  a[1023] = 0.0;
}
