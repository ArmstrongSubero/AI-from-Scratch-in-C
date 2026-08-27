/* 003_Forward_Pass.c */
#include <stdio.h>

static float step_function(float z)
{
    return z >= 0.0f ? 1.0f : 0.0f;
}

int main(void)
{
    float w1 = 1.0f, w2 = 1.0f;
    float b  = -1.5f;

    /* Example A: both inputs high */
    float x1 = 1.0f, x2 = 1.0f;
    float z  = w1 * x1 + w2 * x2 + b;
    float y  = step_function(z);
    printf("A: x1=%.0f x2=%.0f  z=%.2f  y=%.0f\n", x1, x2, z, y);

    /* Example B: one input low */
    x1 = 1.0f;
    x2 = 0.0f;
    z  = w1 * x1 + w2 * x2 + b;
    y  = step_function(z);
    printf("B: x1=%.0f x2=%.0f  z=%.2f  y=%.0f\n", x1, x2, z, y);

    return 0;
}
