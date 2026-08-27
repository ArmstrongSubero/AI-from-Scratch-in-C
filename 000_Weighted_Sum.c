/* 000_Weighted_Sum.c */
#include <stdio.h>

int main(void)
{
    float x1 = 1.0f, x2 = 0.0f; /* inputs */
    float w1 = 0.5f, w2 = 0.5f; /* weights */

    float z = w1 * x1 + w2 * x2;

    printf("z = %.2f\n", z);

    return 0;
}
