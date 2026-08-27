/* 019_MSE_Review.c */
#include <stdio.h>

static float mse_loss(float y, float t)
{
    float diff = t - y;
    return diff * diff;
}

static float mse_derivative(float y, float t)
{
    return -2.0f * (t - y);
}

int main(void)
{
    float targets[] = { 0.0f, 1.0f, 1.0f, 0.0f }; /* XOR */
    /* close predictions */
    float good[] = { 0.05f, 0.92f, 0.95f, 0.08f };
    /* poor predictions */
    float bad[]  = { 0.80f, 0.20f, 0.30f, 0.70f };
    int i;

    printf("--- Good predictions ---\n");
    printf("  target  predict   loss      gradient\n");
    float total_good = 0.0f;
    for (i = 0; i < 4; i++) {
        float l = mse_loss(good[i], targets[i]);
        float g = mse_derivative(good[i], targets[i]);
        total_good += l;
        printf("  %.2f    %.2f     %.4f    %+.4f\n",
               targets[i], good[i], l, g);
    }
    printf("  MSE = %.4f\n", total_good / 4.0f);

    printf("\n--- Bad predictions ---\n");
    printf("  target  predict   loss      gradient\n");
    float total_bad = 0.0f;
    for (i = 0; i < 4; i++) {
        float l = mse_loss(bad[i], targets[i]);
        float g = mse_derivative(bad[i], targets[i]);
        total_bad += l;
        printf("  %.2f    %.2f     %.4f    %+.4f\n",
               targets[i], bad[i], l, g);
    }
    printf("  MSE = %.4f\n", total_bad / 4.0f);

    return 0;
}
