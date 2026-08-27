/* 002_Activation_Function.c */
#include <stdio.h>

static float step_function(float z)
{
    return z >= 0.0f ? 1.0f : 0.0f;
}

int main(void)
{
    float x1 = 1.0f, x2 = 1.0f; /* inputs */
    float w1 = 1.0f, w2 = 1.0f; /* weights */

    float b = -1.5f; /* bias */

    float z = w1 * x1 + w2 * x2 + b;

    float y = step_function(z); /* step activation */

    printf("z = %.2f  output = %.0f\n", z, y);

    return 0;
}
