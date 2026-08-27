/* 045_Multi_Kernels.c */
#include <stdio.h>

#define SIG_LEN 8
#define KER_LEN 3
#define N_KERNELS 3
#define OUT_LEN (SIG_LEN - KER_LEN + 1)

int main(void)
{
    float signal[SIG_LEN] = { 0, 0, 1, 2, 1, 0, 0, 0 };

    /* Three different kernels */
    float kernels[N_KERNELS][KER_LEN] = {
        { -1,  2, -1 },   /* spike detector */
        {  1,  1,  1 },   /* smoother (average) */
        { -1,  0,  1 },   /* edge detector (rising) */
    };
    const char *names[] = { "spike", "smooth", "edge" };

    float output[N_KERNELS][OUT_LEN];
    int f, i, k;

    /* Apply each kernel */
    for (f = 0; f < N_KERNELS; f++) {
        for (i = 0; i < OUT_LEN; i++) {
            float sum = 0.0f;
            for (k = 0; k < KER_LEN; k++)
                sum += signal[i + k] * kernels[f][k];
            output[f][i] = sum;
        }
    }

    printf("Signal:  ");
    for (i = 0; i < SIG_LEN; i++)
        printf("%5.1f ", signal[i]);
    printf("\n\n");

    for (f = 0; f < N_KERNELS; f++) {
        printf("%-6s:  ", names[f]);
        for (i = 0; i < OUT_LEN; i++)
            printf("%5.1f ", output[f][i]);
        printf("\n");
    }

    return 0;
}
