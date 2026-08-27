/* 015_Relu_Activation.c */
#include <stdio.h>

static float relu(float z)
{
    return z > 0.0f ? z : 0.0f;
}

static float relu_derivative(float z)
{
    return z > 0.0f ? 1.0f : 0.0f;
}

int main(void)
{
    float values[] = { -10.0f, -5.0f, -2.0f,
        -1.0f, 0.0f,
                        1.0f, 2.0f, 5.0f, 10.0f };
    int n = sizeof(values) / sizeof(values[0]);
    int i;

    printf("  z        relu       derivative\n");
    printf("  -------  ---------  ----------\n");
    for (i = 0; i < n; i++) {
        float r = relu(values[i]);
        float d = relu_derivative(values[i]);
        printf("  %6.1f   %9.6f  %10.6f\n", values[i],
            r, d);
    }
    return 0;
}
