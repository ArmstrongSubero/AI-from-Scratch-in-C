/* 016_GELU.c */
#include <stdio.h>
#include <math.h>

static float gelu(float z)
{
    /* Approximation used in GPT and BERT */
    float c = 0.7978845608f;  /* sqrt(2/pi) */
    float inner = c * (z + 0.044715f * z * z * z);
    return 0.5f * z * (1.0f + tanhf(inner));
}

static float gelu_derivative_numerical(float z)
{
    float h = 0.0001f;
    return (gelu(z + h) - gelu(z - h)) / (2.0f * h);
}

int main(void)
{
    float values[] = { -10.0f, -5.0f, -2.0f,
        -1.0f, 0.0f,
                        1.0f, 2.0f, 5.0f, 10.0f };
    int n = sizeof(values) / sizeof(values[0]);
    int i;

    printf("  z        gelu       "
           "derivative\n");
    printf("  -------  ---------  ----------\n");
    for (i = 0; i < n; i++) {
        float g = gelu(values[i]);
        float d = gelu_derivative_numerical(values[i]);
        printf("  %6.1f   %9.6f  %10.6f\n", values[i],
            g, d);
    }
    return 0;
}
