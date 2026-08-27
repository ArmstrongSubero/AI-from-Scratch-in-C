/* 004_Anding.c */
#include <stdio.h>

typedef struct{
  float w[2];    /* weights (fixed at 2 inputs) */
  float b;       /* bias */
} Perceptron;

static float step_function(float z)
{
    return z >= 0.0f ? 1.0f : 0.0f;
}

static float forward(const Perceptron *p, float x0, float x1)
{
    float z = p->w[0] * x0 + p->w[1] * x1 + p->b;
    return step_function(z);

}

int main(void)
{
    Perceptron p = {.w = {1.0f, 1.0f}, .b = -1.5f};

    printf("0 and 0 = %.0f\n", forward(&p, 0, 0));
    printf("0 and 1 = %.0f\n", forward(&p, 0, 1));
    printf("1 and 0 = %.0f\n", forward(&p, 1, 0));
    printf("1 and 1 = %.0f\n", forward(&p, 1, 1));
    return 0;
}
