/* 167_Merge.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define D 4
#define R 2

int main(void)
{
    float W[D][D], A[R][D], B[D][R];
    int i, j, k;
    srand(42);

    for (i = 0; i < D; i++)
        for (j = 0; j < D; j++)
            W[i][j] = (randf()*2-1) * 0.3f;
    for (i = 0; i < R; i++)
        for (j = 0; j < D; j++)
            A[i][j] = (randf()*2-1) * 0.1f;
    for (i = 0; i < D; i++)
        for (j = 0; j < R; j++)
            B[i][j] = (randf()*2-1) * 0.1f;

    /* Compute B*A */
    float BA[D][D];
    for (i = 0; i < D; i++)
        for (j = 0; j < D; j++) {
            BA[i][j] = 0;
            for (k = 0; k < R; k++)
                BA[i][j] += B[i][k] * A[k][j];
        }

    /* Merge: W_new = W + BA */
    float W_merged[D][D];
    for (i = 0; i < D; i++)
        for (j = 0; j < D; j++)
            W_merged[i][j] = W[i][j] + BA[i][j];

    /* Verify: W*x + B*A*x == W_merged*x */
    float x[D] = { 0.5f, -0.2f, 0.8f, 0.3f };
    float out_separate[D] = {0}, out_merged[D] = {0};

    /* Separate: W*x + B*(A*x) */
    for (i = 0; i < D; i++)
        for (j = 0; j < D; j++)
            out_separate[i] += W[i][j] * x[j];
    float mid[R] = {0};
    for (i = 0; i < R; i++)
        for (j = 0; j < D; j++)
            mid[i] += A[i][j] * x[j];
    for (i = 0; i < D; i++)
        for (j = 0; j < R; j++)
            out_separate[i] += B[i][j] * mid[j];

    /* Merged: W_merged*x */
    for (i = 0; i < D; i++)
        for (j = 0; j < D; j++)
            out_merged[i] += W_merged[i][j] * x[j];

    printf("LoRA weight merging:\n\n");
    printf("  W_merged = W + B*A\n\n");

    printf("  Separate:  [");
    for (i = 0; i < D; i++)
        printf("%+.4f%s", out_separate[i],
               i<D-1?", ":"");
    printf("]\n");
    printf("  Merged:    [");
    for (i = 0; i < D; i++)
        printf("%+.4f%s", out_merged[i],
               i<D-1?", ":"");
    printf("]\n\n");

    float diff = 0;
    for (i = 0; i < D; i++) {
        float d = out_separate[i] - out_merged[i];
        diff += d * d;
    }
    printf("  Difference: %.10f\n\n", sqrtf(diff));

    printf("  After merging, inference is one "
           "matrix\n");
    printf("  multiply rather than two, so the\n");
    printf("  adapter costs nothing to serve.\n\n");

    printf("  Or keep it separate and swap adapters\n");
    printf("    base + LoRA_medical = medical\n");
    printf("    base + LoRA_coding  = coding\n");
    printf("    base + LoRA_legal   = legal\n");
    printf("  One frozen base, many behaviours.\n");

    return 0;
}
