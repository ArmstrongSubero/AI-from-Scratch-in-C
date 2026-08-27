/* 030_All_Optimizers_Comparison.c */
#include <stdio.h>
#include <math.h>

static void grad(float x, float y, float *gx, float *gy)
{
    *gx = 2.0f * x;
    *gy = 20.0f * y;
}

static float f(float x, float y)
{
    return x*x + 10.0f*y*y;
}

int main(void)
{
    int i;
    float gx, gy;

    /* SGD */
    {
        float x = 5.0f, y = 3.0f, lr = 0.05f;
        for (i = 0; i < 50; i++) {
            grad(x, y, &gx, &gy);
            x -= lr * gx;
            y -= lr * gy;
        }
        printf("SGD        after 50 steps: "
               "f=%.6f  x=%+.6f y=%+.6f\n",
               f(x,y), x, y);
    }

    /* Momentum */
    {
        float x = 5.0f, y = 3.0f, lr = 0.05f, beta = 0.9f;
        float vx = 0, vy = 0;
        for (i = 0; i < 50; i++) {
            grad(x, y, &gx, &gy);
            vx = beta*vx + gx;
            vy = beta*vy + gy;
            x -= lr * vx;
            y -= lr * vy;
        }
        printf("Momentum   after 50 steps: "
               "f=%.6f  x=%+.6f y=%+.6f\n",
               f(x,y), x, y);
    }

    /* RMSprop */
    {
        float x = 5.0f, y = 3.0f, lr = 0.1f;
        float decay = 0.9f, eps = 1e-6f;
        float rx = 0, ry = 0;
        for (i = 0; i < 50; i++) {
            grad(x, y, &gx, &gy);
            rx = decay*rx + (1-decay)*gx*gx;
            ry = decay*ry + (1-decay)*gy*gy;
            x -= lr / sqrtf(rx + eps) * gx;
            y -= lr / sqrtf(ry + eps) * gy;
        }
        printf("RMSprop    after 50 steps: "
               "f=%.6f  x=%+.6f y=%+.6f\n",
               f(x,y), x, y);
    }

    /* Adam */
    {
        float x = 5.0f, y = 3.0f, lr = 0.1f;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float mx = 0, my = 0, vx = 0, vy = 0;
        float b1t = 1, b2t = 1;
        for (i = 0; i < 50; i++) {
            grad(x, y, &gx, &gy);
            mx = b1*mx + (1-b1)*gx;
            my = b1*my + (1-b1)*gy;
            vx = b2*vx + (1-b2)*gx*gx;
            vy = b2*vy + (1-b2)*gy*gy;
            b1t *= b1;
            b2t *= b2;
            float mxh = mx/(1-b1t), myh = my/(1-b1t);
            float vxh = vx/(1-b2t), vyh = vy/(1-b2t);
            x -= lr * mxh / (sqrtf(vxh) + eps);
            y -= lr * myh / (sqrtf(vyh) + eps);
        }
        printf("Adam       after 50 steps: "
               "f=%.6f  x=%+.6f y=%+.6f\n",
               f(x,y), x, y);
    }

    return 0;
}
