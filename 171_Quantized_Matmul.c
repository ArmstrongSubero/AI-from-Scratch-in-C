/* 171_Quantized_Matmul.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define M 64
#define K 64
#define N_VEC 64

static void quantize_vec(const float *x, 
    signed char *q, 
                         float *scale, int n)
{
    float absmax = 0;
    int i;
    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > absmax) absmax = fabsf(x[i]);
    *scale = absmax / 127.0f;
    if (*scale < 1e-10f) *scale = 1e-10f;
    for (i = 0; i < n; i++) {
        float v = x[i] / (*scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (signed char)(v
            + (v >= 0 ? 0.5f : -0.5f));
    }
}

int main(void)
{
    /* A weight matrix and input vector */
    float W[M][K], x[K];
    int i, j;

    srand(42);
    for (i = 0; i < M; i++)
        for (j = 0; j < K; j++)
            W[i][j] = (randf()*2-1) * 0.5f;
    for (j = 0; j < K; j++)
        x[j] = (randf()*2-1) * 0.5f;

    /* Float32 matmul */
    float out_f32[M];
    for (i = 0; i < M; i++) {
        out_f32[i] = 0;
        for (j = 0; j < K; j++)
            out_f32[i] += W[i][j] * x[j];
    }

    /* Quantize W rows and x, multiply in int32,
       then scale back to float */
    signed char W_q[M][K], x_q[K];
    float w_scales[M], x_scale;

    for (i = 0; i < M; i++)
        quantize_vec(W[i], W_q[i], &w_scales[i], K);
    quantize_vec(x, x_q, &x_scale, K);

    float out_q[M];
    for (i = 0; i < M; i++) {
        int acc = 0;  /* int32 accumulator */
        for (j = 0; j < K; j++)
            acc += (int)W_q[i][j] * (int)x_q[j];
        out_q[i] = acc * w_scales[i] * x_scale;
    }

    /* Compare */
    float max_err = 0, rms_err = 0;
    for (i = 0; i < M; i++) {
        float err = fabsf(out_f32[i] - out_q[i]);
        if (err > max_err) max_err = err;
        rms_err += err * err;
    }
    rms_err = sqrtf(rms_err / M);

    printf("Quantized matrix multiply "
           "(%dx%d):\n\n", M, K);
    printf("  First 8 outputs:\n");
    printf("  pos  float32     int8_quant  error\n");
    printf("  ---  ----------  ----------  -----\n");
    for (i = 0; i < 8; i++)
        printf("  %2d   %+.4f     %+.4f     %.4f\n",
               i, out_f32[i], out_q[i], 
               fabsf(out_f32[i] - out_q[i]));

    printf("\n  Max error: %.6f\n", max_err);
    printf("  RMS error: %.6f\n", rms_err);
    float scale_out = 0;
    for (i = 0; i < M; i++)
        if (fabsf(out_f32[i]) > scale_out)
            scale_out = fabsf(out_f32[i]);
    printf("  Max error over largest output: %.2f%%\n",
           max_err / scale_out * 100);

    printf("\n  Memory comparison:\n");
    printf("    float32: %lu bytes\n",
           M * K * sizeof(float));
    printf("    int8:    %lu bytes + %d scales (%lu "
           "B)\n",
           M * K * sizeof(signed char), M, 
           M * sizeof(float));
    printf("    Ratio:   %.1fx smaller\n",
           (float)(M * K * sizeof(float)) /
           (M * K * sizeof(signed char)
            + M * sizeof(float)));

    /* Measure it rather than assert it */
    volatile float sink = 0;
    clock_t t0, t1;
    int r, reps = 200000;

    t0 = clock();
    for (r = 0; r < reps; r++)
        for (i = 0; i < M; i++) {
            float a = 0;
            for (j = 0; j < K; j++)
                a += W[i][j] * x[j];
            sink += a;
        }
    t1 = clock();
    double t_f32 = (double)(t1 - t0) / CLOCKS_PER_SEC;

    t0 = clock();
    for (r = 0; r < reps; r++)
        for (i = 0; i < M; i++) {
            int acc = 0;
            for (j = 0; j < K; j++)
                acc += (int)W_q[i][j] * (int)x_q[j];
            sink += acc * w_scales[i] * x_scale;
        }
    t1 = clock();
    double t_int8 = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("\n  Timing over %d repetitions:\n", reps);
    printf("    float32: %.3f s\n", t_f32);
    printf("    int8:    %.3f s\n", t_int8);
    printf("    speedup: %.2fx\n", t_f32 / t_int8);
    printf("\n  Dequantization runs once per "
           "output,\n");
    printf("  not once per multiply, so the inner "
           "loop\n");
    printf("  stays in integer arithmetic.\n");

    return 0;
}
