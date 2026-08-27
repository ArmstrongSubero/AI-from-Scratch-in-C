/* 014_Tanh_Activation.c */
#include <stdio.h>
#include <math.h>

static float my_tanh(float z)
{
    /* Guard against overflow for very negative z */
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}

static float tanh_derivative(float tanh_output)
{
    return 1.0f - tanh_output * tanh_output;
}

int main(void)
{
    float values[] = { -10.0f, -5.0f, -2.0f,
        -1.0f, 0.0f,
                        1.0f, 2.0f, 5.0f, 10.0f };
    int n = sizeof(values) / sizeof(values[0]);
    int i;

    printf("  z        tanh       derivative\n");
    printf("  -------  ---------  ----------\n");
    for (i = 0; i < n; i++) {
        float t = my_tanh(values[i]);
        float d = tanh_derivative(t);
        printf("  %6.1f   %9.6f  %10.6f\n", values[i],
            t, d);
    }
    return 0;
}
