/* 028_RMSPROP.c */
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
    float decay = 0.9f;
    float eps = 1e-6f;
    float rx = 0.0f, ry = 0.0f;
    /* running avg of squared gradients */
    int i;

    printf("RMSprop on f(x,y) = x^2 + 10*y^2\n");
    printf("  step   x        y        f(x,y)\n");
    for (i = 0; i < 30; i++) {
        float f = x * x + 10.0f * y * y;
        if (i % 5 == 0)
            printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", i, x,
                y, f);

        float gx, gy;
        gradient(x, y, &gx, &gy);

        rx = decay * rx + (1.0f - decay) * gx * gx;
        ry = decay * ry + (1.0f - decay) * gy * gy;

        x -= lr / sqrtf(rx + eps) * gx;
        y -= lr / sqrtf(ry + eps) * gy;
    }
    float f = x * x + 10.0f * y * y;
    printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", 30, x, y, f);
    return 0;
}
