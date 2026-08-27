/* 021_BCE.c */
#include <stdio.h>
#include <math.h>

static float bce_loss(float y, float t)
{
    /* Clamp to avoid log(0) */
    float tiny = 1e-9f;
    if (y < tiny) y = tiny;
    if (y > 1.0f - tiny) y = 1.0f - tiny;

    return -(t * logf(y) + (1.0f - t) * logf(1.0f - y));
}

static float bce_derivative(float y, float t)
{
    float tiny = 1e-9f;
    if (y < tiny) y = tiny;
    if (y > 1.0f - tiny) y = 1.0f - tiny;

    return -(t / y) + (1.0f - t) / (1.0f - y);
}

int main(void)
{
    float predictions[] =
        { 0.0001f, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f };
    float target = 1.0f;
    int i, n = 6;

    printf("BCE loss and gradient (target=1.0):\n");
    printf("  predict   loss      gradient\n");
    for (i = 0; i < n; i++) {
        float l = bce_loss(predictions[i], target);
        float g = bce_derivative(predictions[i], target);
        printf("  %.4f    %7.4f   %+10.4f\n",
               predictions[i], l, g);
    }

    return 0;
}
