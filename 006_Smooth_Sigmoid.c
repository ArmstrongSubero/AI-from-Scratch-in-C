/* 006_Smooth_Sigmoid.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

int main(void)
{
    /* Use the weights learned from step 6 */
    float w0 = 0.2f, w1 = 0.1f, b = -0.2f;

    float X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
    int s;

    printf("AND gate with sigmoid activation:\n");
    for (s = 0; s < 4; s++) {
        float z = w0 * X[s][0] + w1 * X[s][1] + b;
        float y = sigmoid(z);
        printf("  %.0f, %.0f  ->  z=%6.3f  sigmoid=%.4f\n",
               X[s][0], X[s][1], z, y);
    }

    /* Show the derivative at each point */
    printf("\nSigmoid derivative at each point:\n");
    for (s = 0; s < 4; s++) {
        float z = w0 * X[s][0] + w1 * X[s][1] + b;
        float sig = sigmoid(z);
        float dsig = sig * (1.0f - sig);
        printf("  %.0f, %.0f  ->  sigmoid=%.4f  derivative=%.4f\n",
               X[s][0], X[s][1], sig, dsig);
    }

    return 0;
}
