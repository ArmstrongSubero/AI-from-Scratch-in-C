/* 170_Group_Quant.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N 32
#define GROUP_SIZE 8

static void quantize_group(const float *x, 
    signed char *q, 
                           float *scales, int n, 
                           int group_size)
{
    int g, i;
    int n_groups = n / group_size;

    for (g = 0; g < n_groups; g++) {
        int offset = g * group_size;
        float absmax = 0;

        for (i = 0; i < group_size; i++)
            if (fabsf(x[offset + i]) > absmax)
                absmax = fabsf(x[offset + i]);

        scales[g] = absmax / 127.0f;
        if (scales[g] < 1e-10f) scales[g] = 1e-10f;

        for (i = 0; i < group_size; i++) {
            float v = x[offset + i] / scales[g];
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[offset + i] = (signed char)
                (v + (v >= 0 ? 0.5f : -0.5f));
        }
    }
}

static void dequantize_group(const signed char *q, 
                             const float *scales, 
                             float *out, int n, 
                             int group_size)
{
    int g, i;
    int n_groups = n / group_size;

    for (g = 0; g < n_groups; g++) {
        int offset = g * group_size;
        for (i = 0; i < group_size; i++)
            out[offset + i] = q[offset + i] * scales[g];
    }
}

int main(void)
{
    float x[N];
    signed char q[N];
    float scales[N / GROUP_SIZE];
    float x_hat[N];
    int i, g;

    srand(42);

    /* Create data with an outlier in one group */
    for (i = 0; i < N; i++)
        x[i] = (randf()*2-1) * 0.5f;
    x[3] = 10.0f;  /* outlier */

    printf("Group quantization (group_size=%d):\n\n",
           GROUP_SIZE);

    /* Per-tensor quantization */
    signed char q_tensor[N];
    float tensor_scale;
    float absmax = 0;
    for (i = 0; i < N; i++)
        if (fabsf(x[i]) > absmax) absmax = fabsf(x[i]);
    tensor_scale = absmax / 127.0f;
    for (i = 0; i < N; i++) {
        float v = x[i] / tensor_scale;
        q_tensor[i] = (signed char)
            (v + (v >= 0 ? 0.5f : -0.5f));
    }

    /* Group quantization */
    quantize_group(x, q, scales, N, GROUP_SIZE);
    dequantize_group(q, scales, x_hat, N, GROUP_SIZE);

    /* Compute errors for both */
    float err_tensor = 0, err_group = 0;
    for (i = 0; i < N; i++) {
        float deq_t = q_tensor[i] * tensor_scale;
        err_tensor += (x[i] - deq_t) * (x[i] - deq_t);
        err_group += (x[i] - x_hat[i])
            * (x[i] - x_hat[i]);
    }

    printf("  Per-tensor RMS error: %.6f\n",
           sqrtf(err_tensor / N));
    printf("  Group (g=%d) RMS error: %.6f\n",
           GROUP_SIZE, sqrtf(err_group / N));
    printf("  Improvement: %.1fx\n\n",
           sqrtf(err_tensor) / sqrtf(err_group));

    /* Where the error lands, group by group */
    printf("  grp   absmax     scale  tensor  group\n");
    for (g = 0; g < N / GROUP_SIZE; g++) {
        int o = g * GROUP_SIZE;
        float et = 0, eg = 0;
        for (i = 0; i < GROUP_SIZE; i++) {
            float dt = q_tensor[o + i] * tensor_scale;
            et += (x[o + i] - dt) * (x[o + i] - dt);
            eg += (x[o + i] - x_hat[o + i])
                * (x[o + i] - x_hat[o + i]);
        }
        printf("  %3d  %8.4f  %.6f  %.4f  %.4f\n", g,
               scales[g] * 127.0f, scales[g], 
               sqrtf(et / GROUP_SIZE), 
               sqrtf(eg / GROUP_SIZE));
    }
    printf("\n  Group 0 holds the outlier x[3]=10.0 "
           "so\n");
    printf("  its scale equals the tensor scale and "
           "it\n");
    printf("  gains nothing. The damage stops "
           "there.\n\n");

    printf("  DeepSeek-V3 uses:\n");
    printf("    1x128 tiles for activations\n");
    printf("    128x128 blocks for weights\n");
    printf("    FP8 (E4M3) format instead of int8\n");
    printf("    Online scales computed on the fly\n");

    return 0;
}
