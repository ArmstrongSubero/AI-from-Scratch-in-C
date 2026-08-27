/* 172_4bit.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N 16

/* Pack two 4-bit values into one byte */
static void quantize_4bit(const float *x, 
                          unsigned char *packed, 
                          float *scale, int n)
{
    float absmax = 0;
    int i;
    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > absmax) absmax = fabsf(x[i]);
    *scale = absmax / 7.0f;  /* 4-bit signed: -8 to 7 */
    if (*scale < 1e-10f) *scale = 1e-10f;

    for (i = 0; i < n; i += 2) {
        float a = x[i] / (*scale);
        int v0 = (int)(a + (a >= 0 ? 0.5f : -0.5f));
        float b = x[i+1] / (*scale);
        int v1 = (int)(b + (b >= 0 ? 0.5f : -0.5f));
        if (v0 > 7) v0 = 7;
        if (v0 < -8) v0 = -8;
        if (v1 > 7) v1 = 7;
        if (v1 < -8) v1 = -8;
        /* Pack: low nibble = v0, high nibble = v1 */
        packed[i/2] = ((v1 & 0xF) << 4) | (v0 & 0xF);
    }
}

static void dequantize_4bit(
        const unsigned char *packed, 
                            float scale, float *out, 
                            int n)
{
    int i;
    for (i = 0; i < n; i += 2) {
        int v0 = packed[i/2] & 0xF;
        int v1 = (packed[i/2] >> 4) & 0xF;
        /* Sign extend from 4 bits */
        if (v0 & 0x8) v0 |= ~0xF;
        if (v1 & 0x8) v1 |= ~0xF;
        out[i] = v0 * scale;
        out[i+1] = v1 * scale;
    }
}

int main(void)
{
    float x[N];
    unsigned char packed[N/2];
    float scale, x_hat[N];
    int i;

    srand(42);
    for (i = 0; i < N; i++)
        x[i] = (randf()*2-1) * 1.0f;

    quantize_4bit(x, packed, &scale, N);
    dequantize_4bit(packed, scale, x_hat, N);

    printf("4-bit quantization (two values per "
           "byte):\n\n");
    printf("  Original  Dequant   Error    Quant\n");
    printf("  --------  --------  -------  -----\n");
    for (i = 0; i < N; i++) {
        int val;
        if (i % 2 == 0) val = packed[i/2] & 0xF;
        else val = (packed[i/2] >> 4) & 0xF;
        if (val & 0x8) val |= ~0xF;
        printf("  %+.4f   %+.4f   %.4f  %3d\n",
               x[i], x_hat[i], 
                   fabsf(x[i] - x_hat[i]), val);
    }

    float rms = 0;
    for (i = 0; i < N; i++)
        rms += (x[i] - x_hat[i]) * (x[i] - x_hat[i]);
    rms = sqrtf(rms / N);

    printf("\n  RMS error:    %.4f\n", rms);
    printf("  Memory:       %d bytes (was %lu bytes)\n",
           N/2 + 4, N * sizeof(float));
    printf("  Compression:  %.1fx\n",
           (float)(N * sizeof(float)) / (N/2 + 4));

    /* Same vector at int8, so the claim is measured */
    float s8 = 0, rms8 = 0;
    for (i = 0; i < N; i++)
        if (fabsf(x[i]) > s8) s8 = fabsf(x[i]);
    s8 /= 127.0f;
    for (i = 0; i < N; i++) {
        float v = x[i] / s8;
        float d = (float)(int)(v
            + (v >= 0 ? 0.5f : -0.5f))
                * s8;
        rms8 += (x[i] - d) * (x[i] - d);
    }
    rms8 = sqrtf(rms8 / N);

    printf("  int8 RMS on the same vector: "
           "%.4f\n", rms8);
    printf("  4-bit error is %.0fx "
           "larger for 2x less\n",
           rms / rms8);
    printf("  memory. Only 16 levels per value.\n");
    printf("  Many LLMs run at 4-bit with little "
           "loss\n");
    printf("  because weight distributions are "
           "nearly\n");
    printf("  Gaussian and most values sit near "
           "zero.\n");

    return 0;
}
