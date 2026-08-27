/* 029_ADAM.c */
#include <stdio.h>
#include <math.h>

static void gradient(float x, float y, float *gx, float *gy)
{
    *gx = 2.0f * x;
    *gy = 20.0f * y;
}

int main(void)
{
    float x = 5.0f, y = 3.0f;
    float lr = 0.1f;
    float beta1 = 0.9f, beta2 = 0.999f;
    float eps = 1e-8f;
    float mx = 0.0f, my = 0.0f;    /* first moment */
    float vx = 0.0f, vy = 0.0f;    /* second moment */
    float b1t = 1.0f, b2t = 1.0f;
    /* beta1^t, beta2^t for bias correction */
    int i;

    printf("Adam on f(x,y) = x^2 + 10*y^2\n");
    printf("  step   x        y        f(x,y)\n");
    for (i = 0; i < 30; i++) {
        float f = x * x + 10.0f * y * y;
        if (i % 5 == 0)
            printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", i, x,
                y, f);

        float gx, gy;
        gradient(x, y, &gx, &gy);

        /* Update moments */
        mx = beta1 * mx + (1.0f - beta1) * gx;
        my = beta1 * my + (1.0f - beta1) * gy;
        vx = beta2 * vx + (1.0f - beta2) * gx * gx;
        vy = beta2 * vy + (1.0f - beta2) * gy * gy;

        /* Bias correction */
        b1t *= beta1;
        b2t *= beta2;
        float mx_hat = mx / (1.0f - b1t);
        float my_hat = my / (1.0f - b1t);
        float vx_hat = vx / (1.0f - b2t);
        float vy_hat = vy / (1.0f - b2t);

        /* Update parameters */
        x -= lr * mx_hat / (sqrtf(vx_hat) + eps);
        y -= lr * my_hat / (sqrtf(vy_hat) + eps);
    }
    float f = x * x + 10.0f * y * y;
    printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", 30, x, y, f);
    return 0;
}
