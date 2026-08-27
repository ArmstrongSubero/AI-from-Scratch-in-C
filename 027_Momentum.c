/* 027_Momentum.c */
#include <stdio.h>

static void gradient(float x, float y, float *gx, float *gy)
{
    *gx = 2.0f * x;
    *gy = 20.0f * y;
}

int main(void)
{
    float x = 5.0f, y = 3.0f;
    float lr = 0.05f;
    float beta = 0.9f;
    float vx = 0.0f, vy = 0.0f;   /* velocity */
    int i;

    printf("SGD with Momentum on f(x,y) = x^2 + 10*y^2\n");
    printf("  step   x        y        f(x,y)\n");
    for (i = 0; i < 30; i++) {
        float f = x * x + 10.0f * y * y;
        if (i % 5 == 0)
            printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", i, x,
                y, f);

        float gx, gy;
        gradient(x, y, &gx, &gy);

        vx = beta * vx + gx;
        vy = beta * vy + gy;

        x -= lr * vx;
        y -= lr * vy;
    }
    float f = x * x + 10.0f * y * y;
    printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", 30, x, y, f);
    return 0;
}
