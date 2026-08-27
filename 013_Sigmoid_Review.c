/* 013_Sigmoid_Review.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

static float sigmoid_derivative(float sig_output)
{
    /* Derivative computed from the output, not from z
        */
    return sig_output * (1.0f - sig_output);
}

int main(void)
{
    float values[] = { -10.0f, -5.0f, -2.0f,
        -1.0f, 0.0f,
                        1.0f, 2.0f, 5.0f, 10.0f };
    int n = sizeof(values) / sizeof(values[0]);
    int i;

    printf("  z        sigmoid    derivative\n");
    printf("  -------  ---------  ----------\n");
    for (i = 0; i < n; i++) {
        float s = sigmoid(values[i]);
        float d = sigmoid_derivative(s);
        printf("  %6.1f   %9.6f  %10.6f\n", values[i],
            s, d);
    }
    return 0;
}
