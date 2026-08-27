/* 022_BCE_VS_MSE.c */
#include <stdio.h>
#include <math.h>

int main(void)
{
    float predictions[] =
        { 0.0001f, 0.01f, 0.1f, 0.5f, 0.9f };
    float target = 1.0f;
    float tiny = 1e-9f;
    int i;

    printf("Effective gradient (target=1.0):\n");
    printf("  predict   MSE*sig'     BCE*sig'\n");
    for (i = 0; i < 5; i++) {
        float y = predictions[i];
        float sig_d = y * (1.0f - y);

        /* MSE effective */
        float mse_g = -2.0f * (target - y);
        float mse_eff = mse_g * sig_d;

        /* BCE effective */
        float yc = y < tiny ? tiny
                   : (y > 1.0f - tiny ? 1.0f - tiny : y);
        float bce_g = -(target / yc)
                      + (1.0f - target) / (1.0f - yc);
        float bce_eff = bce_g * sig_d;

        printf("  %.4f    %+10.6f    %+10.6f\n",
               y, mse_eff, bce_eff);
    }

    return 0;
}
