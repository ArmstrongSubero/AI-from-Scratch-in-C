/* 020_MSE_Problem.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

int main(void)
{
    /* Simulate the gradient that reaches a weight */
    float predictions[] =
        { 0.0001f, 0.01f, 0.1f, 0.5f, 0.9f };
    float target = 1.0f;
    int i;

    printf("MSE gradient through sigmoid (target=1.0):\n");
    printf("  predict  mse_grad  sig_deriv  eff\n");
    for (i = 0; i < 5; i++) {
        float y = predictions[i];
        float mse_grad  = -2.0f * (target - y);
        float sig_deriv = y * (1.0f - y);
        float effective = mse_grad * sig_deriv;
        printf("  %.4f    %+7.4f    %.6f   %+.6f\n",
               y, mse_grad, sig_deriv, effective);
    }

    return 0;
}
