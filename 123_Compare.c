/* 123_Compare.c */
#include <stdio.h>
#include <math.h>

#define DIM 5

static void layernorm(const float *x, float *out, int n)
{
    int i;
    double sum = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) sum += x[i];
    float mean = (float)(sum / n);
    double var = 0;
    for (i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float si = 1.0f / sqrtf((float)(var / n) + eps);
    for (i = 0; i < n; i++) out[i] = (x[i] - mean) * si;
}

static void rmsnorm(const float *x, float *out, int n)
{
    int i;
    double ss = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float ri = 1.0f / sqrtf((float)(ss / n) + eps);
    for (i = 0; i < n; i++) out[i] = x[i] * ri;
}

int main(void)
{
    /* Test case 1: zero-centered data */
    float a[DIM] = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
    /* Test case 2: positive-offset data */
    float b[DIM] = { 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    /* Test case 3: mixed signs */
    float c[DIM] = { 3.0f, -1.0f, 0.5f, 2.0f, -0.5f };

    float ln_out[DIM], rms_out[DIM];
    int i;

    struct { const char *name; float
        *data;
        }
        tests[] = {
        { "zero-centered", a },
        { "positive-offset", b },
        { "mixed", c },
    };

    printf("LayerNorm vs RMSNorm comparison:\n\n");

    for (int t = 0; t < 3; t++) {
        float *x = tests[t].data;
        layernorm(x, ln_out, DIM);
        rmsnorm(x, rms_out, DIM);

        printf("  %s: [", tests[t].name);
        for (i = 0; i < DIM; i++)
            printf("%.1f%s", x[i], i<DIM-1?", ":"");
        printf("]\n");

        printf("    LN:  [");
        for (i = 0; i < DIM; i++)
            printf("%+.3f%s", ln_out[i],
                   i<DIM-1?", ":"");
        printf("]\n");

        printf("    RMS: [");
        for (i = 0; i < DIM; i++)
            printf("%+.3f%s", rms_out[i],
                   i<DIM-1?", ":"");
        printf("]\n\n");
    }

    printf("For zero-centered data the two agree.\n");
    printf("For offset data, LN centers first while\n");
    printf("RMS only scales. The gap is usually\n");
    printf("small, because learned "
           "gamma compensates.\n");

    return 0;
}
