/* 169_Absmax.c */
#include <stdio.h>
#include <math.h>

#define N 8

/* Quantize float array to int8 using absmax */
static void quantize_absmax(const float *x, 
                            signed char *q, 
                            float *scale, int n)
{
    /* Find max absolute value */
    float absmax = 0;
    int i;
    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > absmax)
            absmax = fabsf(x[i]);

    *scale = absmax / 127.0f;

    /* Quantize: round(x / scale) */
    for (i = 0; i < n; i++) {
        float v = x[i] / (*scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (signed char)(v
            + (v >= 0 ? 0.5f : -0.5f));
    }
}

/* Dequantize: multiply by scale */
static void dequantize(const signed char *q, 
    float scale, 
                       float *out, int n)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = q[i] * scale;
}

int main(void)
{
    float x[N] = { 0.532f, -0.187f, 1.204f, -0.023f, 
                    0.891f, -0.645f, 0.078f, -1.105f };
    signed char q[N];
    float scale, x_hat[N];
    int i;

    printf("Absmax int8 quantization:\n\n");

    printf("  Original:    [");
    for (i = 0; i < N; i++)
        printf("%+.3f%s", x[i], i<N-1 ? ", " : "");
    printf("]\n");

    quantize_absmax(x, q, &scale, N);

    printf("  Quantized:   [");
    for (i = 0; i < N; i++)
        printf("%4d%s", q[i], i<N-1 ? ", " : "");
    printf("]\n");
    printf("  Scale:       %.6f\n", scale);

    dequantize(q, scale, x_hat, N);

    printf("  Dequantized: [");
    for (i = 0; i < N; i++)
        printf("%+.3f%s", x_hat[i], i<N-1 ? ", " : "");
    printf("]\n\n");

    /* Compute error */
    float max_err = 0, sum_sq = 0;
    for (i = 0; i < N; i++) {
        float err = fabsf(x[i] - x_hat[i]);
        if (err > max_err) max_err = err;
        sum_sq += err * err;
    }
    printf("  Max error:   %.6f\n", max_err);
    printf("  RMS error:   %.6f\n", sqrtf(sum_sq / N));
    printf("  Memory:      %d bytes, was %lu, %.1fx\n",
           N * 1 + 4, /* int8 plus one float scale */
           N * sizeof(float), 
           (float)(N * sizeof(float)) / (N + 4));
    printf("  The scale is 4 bytes of fixed "
           "overhead.\n");
    printf("  Across a 128 value group it costs "
           "%.1f%%\n",
           100.0f * 4 / (128 + 4));
    printf("  and the ratio reaches %.2fx.\n",
           128.0f * 4 / (128 + 4));

    return 0;
}
