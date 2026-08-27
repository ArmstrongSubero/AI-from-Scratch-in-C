/* 017_Comparison.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float sigmoid_d(float z)
{
    float s = sigmoid(z);
    return s * (1.0f - s);
}

static float my_tanh(float z)
{
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f*z);
    return (1.0f-e)/(1.0f+e);
}
static float tanh_d(float z)
{
    float t = my_tanh(z);
    return 1.0f - t * t;
}

static float relu(float z)
{
    return z > 0.0f ? z : 0.0f;
}
static float relu_d(float z)
{
    return z > 0.0f ? 1.0f : 0.0f;
}

static float gelu(float z)
{
    float c = 0.7978845608f;
    return 0.5f*z*(1.0f+tanhf(c*(z+0.044715f*z*z*z)));
}
static float gelu_d(float z)
{
    float h = 0.0001f;
    return (gelu(z+h)-gelu(z-h))/(2.0f*h);
}

int main(void)
{
    float values[] = { -3.0f, -2.0f, -1.0f,
        -0.5f, 0.0f,
                        0.5f, 1.0f, 2.0f, 3.0f };
    int n = sizeof(values) / sizeof(values[0]);
    int i;

    printf("--- Output values ---\n");
    printf("  z      sigmoid   tanh"
           "      relu      gelu\n");
    for (i = 0; i < n; i++) {
        float z = values[i];
        printf("  %5.1f  %7.4f"
               "   %7.4f   %7.4f   %7.4f\n",
               z, sigmoid(z), my_tanh(z),
               relu(z), gelu(z));
    }

    printf("\n--- Derivatives ---\n");
    printf("  z      sigmoid   tanh"
           "      relu      gelu\n");
    for (i = 0; i < n; i++) {
        float z = values[i];
        printf("  %5.1f  %7.4f"
               "   %7.4f   %7.4f   %7.4f\n",
               z, sigmoid_d(z), tanh_d(z),
               relu_d(z), gelu_d(z));
    }

    return 0;
}
