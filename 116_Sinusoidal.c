/* 116_Sinusoidal.c */
#include <stdio.h>
#include <math.h>

#define D_MODEL 8
#define MAX_POS 10

static void sinusoidal_pe(int pos, 
    int d_model, float *pe)
{
    int i;
    for (i = 0; i < d_model; i++) {
        float e = (float)(i / 2 * 2) / d_model;
        float angle = pos / powf(10000.0f, e);
        if (i % 2 == 0)
            pe[i] = sinf(angle);
        else
            pe[i] = cosf(angle);
    }
}

int main(void)
{
    float pe[D_MODEL];
    int pos, i;

    printf("Sinusoidal positional encoding, "
           "d_model=%d\n\n", D_MODEL);
    printf("  pos  ");
    for (i = 0; i < D_MODEL; i++) printf(" dim%d  ", i);
    printf("\n  ---  ");
    for (i = 0; i < D_MODEL; i++) printf("------  ");
    printf("\n");

    for (pos = 0; pos < 6; pos++) {
        sinusoidal_pe(pos, D_MODEL, pe);
        printf("  %2d   ", pos);
        for (i = 0; i < D_MODEL; i++)
            printf("%+.3f ", pe[i]);
        printf("\n");
    }

    /* Show that nearby positions have similar
       encodings */
    printf("\nDistance between position encodings:\n");
    for (pos = 0; pos < 5; pos++) {
        float pe_a[D_MODEL], pe_b[D_MODEL];
        sinusoidal_pe(pos, D_MODEL, pe_a);
        sinusoidal_pe(pos + 1, D_MODEL, pe_b);
        float dist = 0;
        for (i = 0; i < D_MODEL; i++) {
            float d = pe_a[i] - pe_b[i];
            dist += d * d;
        }
        dist = sqrtf(dist);
        printf("  dist(%d, %d) = %.4f\n", pos,
               pos + 1, dist);
    }

    printf("\nNearby positions have similar "
           "encodings.\n");
    printf("The distance stays constant rather than "
           "growing.\n");

    return 0;
}
