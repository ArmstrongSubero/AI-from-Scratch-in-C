/* 122_Rmsnorm.c */
#include <stdio.h>
#include <math.h>

#define DIM 6

typedef struct {
    float gamma[DIM];
}
RMSNorm;

static void rmsnorm_init(RMSNorm *rn, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
        rn->gamma[i] = 1.0f;
}

static void rmsnorm_forward(const RMSNorm *rn, 
                             const float *x, 
                             float *out, int dim, 
                             float *rms_inv_out)
{
    int i;
    float eps = 1e-5f;

    /* Compute root mean square */
    double sum_sq = 0;
    for (i = 0; i < dim; i++)
        sum_sq += x[i] * x[i];
    float rms = sqrtf((float)(sum_sq / dim) + eps);
    float rms_inv = 1.0f / rms;

    /* Normalize and apply gamma (no beta in RMSNorm) */
    for (i = 0; i < dim; i++)
        out[i] = rn->gamma[i] * x[i] * rms_inv;

    *rms_inv_out = rms_inv;
}

int main(void)
{
    RMSNorm rn;
    rmsnorm_init(&rn, DIM);

    float x[DIM] = { 100.0f, 102.0f, 98.0f, 
                     101.0f, 99.0f, 103.0f };
    float out[DIM];
    float rms_inv;
    int i;

    printf("Before RMSNorm:\n  [");
    for (i = 0; i < DIM; i++)
        printf("%.1f%s", x[i], i<DIM-1?", ":"");
    printf("]\n");

    rmsnorm_forward(&rn, x, out, DIM, &rms_inv);

    printf("\nAfter RMSNorm:\n  [");
    for (i = 0; i < DIM; i++)
        printf("%+.4f%s", out[i], i<DIM-1?", ":"");
    printf("]\n");

    /* Verify RMS of output is ~1 */
    double out_rms = 0;
    for (i = 0; i < DIM; i++)
        out_rms += out[i] * out[i];
    out_rms = sqrt(out_rms / DIM);
    printf("\n  Output RMS: %.6f, expect ~1\n",
           out_rms);

    printf("\nRMSNorm vs LayerNorm:\n");
    printf("  LayerNorm subtracts the mean then "
           "divides by std, 2*dim params.\n");
    printf("  RMSNorm divides by RMS only, dim "
           "params, and is faster.\n");
    printf("  DeepSeek-V3 uses RMSNorm throughout.\n");

    return 0;
}
