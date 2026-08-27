/* 120_Stats.c */
#include <stdio.h>
#include <math.h>

#define DIM 6

static void compute_stats(const float *x, int n, 
                          float *mean, float *var)
{
    int i;
    double sum = 0, sum_sq = 0;

    for (i = 0; i < n; i++)
        sum += x[i];
    *mean = (float)(sum / n);

    for (i = 0; i < n; i++) {
        float d = x[i] - *mean;
        sum_sq += d * d;
    }
    *var = (float)(sum_sq / n);
}

int main(void)
{
    /* Two vectors with very different scales */
    float a[DIM] = { 100.0f, 102.0f, 98.0f, 
                     101.0f, 99.0f, 103.0f };
    float b[DIM] = { 0.001f, 0.003f, -0.001f, 
                     0.002f, 0.000f, 0.004f };
    float mean, var;

    compute_stats(a, DIM, &mean, &var);
    printf("Vector a: mean=%.2f  var=%.2f  "
           "std=%.2f\n", mean, var, sqrtf(var));

    compute_stats(b, DIM, &mean, &var);
    printf("Vector b: mean=%.4f  var=%.8f  "
           "std=%.6f\n", mean, var, sqrtf(var));

    printf("\nThese two vectors sit at completely "
           "different scales.\n");
    printf("A dense layer receiving both would "
           "struggle, because\n");
    printf("the same weights must span 98 to 103\n");
    printf("AND values from -0.001 to 0.004.\n");
    printf("Normalization brings both to one scale.\n");

    return 0;
}
