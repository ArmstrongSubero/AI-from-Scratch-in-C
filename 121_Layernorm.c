/* 121_Layernorm.c */
#include <stdio.h>
#include <math.h>

#define DIM 6

typedef struct {
    float gamma[DIM];  /* learned scale */
    float beta[DIM];   /* learned shift */
}
LayerNorm;

static void layernorm_init(LayerNorm *ln, int dim)
{
    int i;
    for (i = 0; i < dim; i++) {
        /* start at 1, an identity scale */
        ln->gamma[i] = 1.0f;
        /* start at 0 (no shift) */
        ln->beta[i] = 0.0f;
    }
}

static void layernorm_forward(const LayerNorm *ln, 
                               const float *x, 
                               float *out, int dim, 
                               float *mean_out, 
                               float *std_inv_out)
{
    int i;
    double sum = 0;
    float eps = 1e-5f;

    /* Compute mean */
    for (i = 0; i < dim; i++) sum += x[i];
    float mean = (float)(sum / dim);

    /* Compute variance */
    double var_sum = 0;
    for (i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var_sum += d * d;
    }
    float var = (float)(var_sum / dim);
    float std_inv = 1.0f / sqrtf(var + eps);

    /* Normalize and apply gamma/beta */
    for (i = 0; i < dim; i++)
        out[i] = ln->gamma[i] * (x[i] - mean)
                 * std_inv + ln->beta[i];

    /* Save for backward pass */
    *mean_out = mean;
    *std_inv_out = std_inv;
}

int main(void)
{
    LayerNorm ln;
    layernorm_init(&ln, DIM);

    float x[DIM] = { 100.0f, 102.0f, 98.0f, 
                     101.0f, 99.0f, 103.0f };
    float out[DIM];
    float mean, std_inv;
    int i;

    printf("Before LayerNorm:\n  [");
    for (i = 0; i < DIM; i++)
        printf("%.1f%s", x[i], i<DIM-1?", ":"");
    printf("]\n");

    layernorm_forward(&ln, x, out, DIM, 
        &mean, &std_inv);

    printf("\nAfter LayerNorm:\n  [");
    for (i = 0; i < DIM; i++)
        printf("%+.4f%s", out[i], i<DIM-1?", ":"");
    printf("]\n");

    /* Verify zero mean and unit variance */
    double out_sum = 0, out_var = 0;
    for (i = 0; i < DIM; i++) out_sum += out[i];
    float out_mean = (float)(out_sum / DIM);
    for (i = 0; i < DIM; i++) {
        float d = out[i] - out_mean;
        out_var += d * d;
    }
    out_var /= DIM;

    printf("\n  Output mean: %.6f, expect ~0\n",
           out_mean);
    printf("  Output var:  %.6f, expect ~1\n",
           (float)out_var);
    printf("\n  mean=%.2f  std_inv=%.4f, saved for "
           "backward\n", mean, std_inv);

    return 0;
}
