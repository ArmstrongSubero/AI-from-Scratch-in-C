/* 164_Low_Rank.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8
#define MAXR 8

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* Fit B*A to target by gradient descent on the
   squared error. An SVD would be exact and this is
   enough to show how the error falls with rank. */
static float fit(const float target[N][N], int r, 
                 int iters, float lr)
{
    float A[MAXR][N], B[N][MAXR];
    int i, j, k, it;

    for (i = 0; i < r; i++)
        for (j = 0; j < N; j++)
            A[i][j] = (randf() * 2 - 1) * 0.5f;
    for (i = 0; i < N; i++)
        for (j = 0; j < r; j++)
            B[i][j] = (randf() * 2 - 1) * 0.5f;

    for (it = 0; it < iters; it++) {
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                float p = 0;
                for (k = 0; k < r; k++)
                    p += B[i][k] * A[k][j];
                float e = p - target[i][j];
                for (k = 0; k < r; k++) {
                    float b = B[i][k], a = A[k][j];
                    B[i][k] -= lr * e * a;
                    A[k][j] -= lr * e * b;
                }
            }
    }

    /* Relative error, ||BA - M|| over ||M|| */
    float num = 0, den = 0;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            float p = 0;
            for (k = 0; k < r; k++)
                p += B[i][k] * A[k][j];
            float d = p - target[i][j];
            num += d * d;
            den += target[i][j] * target[i][j];
        }
    return sqrtf(num / den);
}

int main(void)
{
    float dense[N][N], lowrank[N][N];
    float C[N][2], D[2][N];
    int i, j, k, r;

    srand(42);

    /* A matrix with no structure at all */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            dense[i][j] = (randf() * 2 - 1);

    /* A matrix built to have rank 2, which is what a
       fine-tuning update is claimed to look like */
    for (i = 0; i < N; i++)
        for (j = 0; j < 2; j++)
            C[i][j] = (randf() * 2 - 1);
    for (i = 0; i < 2; i++)
        for (j = 0; j < N; j++)
            D[i][j] = (randf() * 2 - 1);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            lowrank[i][j] = 0;
            for (k = 0; k < 2; k++)
                lowrank[i][j] += C[i][k] * D[k][j];
        }

    printf("Approximating %dx%d by B*A\n\n", N, N);
    printf("  rank  params  vs %d  dense err  "
           "rank2 err\n", N * N);
    printf("  ----  ------  -----  ---------  "
           "---------\n");

    for (r = 1; r <= 8; r *= 2) {
        int p = 2 * N * r;
        srand(7);
        float e_dense = fit(dense, r, 4000, 0.02f);
        srand(7);
        float e_low = fit(lowrank, r, 4000, 0.02f);
        printf("  %4d  %6d  %4.2fx  %9.4f  %10.4f\n",
               r, p, (float)(N*N) / p, e_dense, e_low);
    }

    printf("\n  The last column collapses at rank 2 "
           "and\n");
    printf("  stays there, because the target "
           "really\n");
    printf("  was rank 2 and nothing is left to "
           "fit.\n");
    printf("  The dense column falls slowly and "
           "only\n");
    printf("  reaches zero at full rank.\n\n");

    printf("  That gap is the whole bet behind "
           "LoRA.\n");
    printf("  It works if a fine-tuning update "
           "looks\n");
    printf("  like the last column rather than the\n");
    printf("  one before it, which is an empirical\n");
    printf("  claim about training and not a "
           "theorem.\n");

    return 0;
}
