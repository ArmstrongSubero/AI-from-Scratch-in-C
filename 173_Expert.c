/* 173_Expert.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

#define DM 6
#define D_EXPERT 8

typedef struct {
    float W_up[D_EXPERT][DM];
    float W_down[DM][D_EXPERT];
}
Expert;

/* Uniform init scaled by fan in, so the expert
   neither amplifies nor erases its input */
static void expert_init(Expert *e)
{
    float a_up = sqrtf(3.0f / DM);
    float a_dn = sqrtf(3.0f / D_EXPERT);
    int i, j;

    for (i = 0; i < D_EXPERT; i++)
        for (j = 0; j < DM; j++)
            e->W_up[i][j] = (randf()*2-1) * a_up;

    for (i = 0; i < DM; i++)
        for (j = 0; j < D_EXPERT; j++)
            e->W_down[i][j] = (randf()*2-1) * a_dn;
}

static void expert_fwd(const Expert *e, 
    const float x[DM], 
                       float hid[D_EXPERT], 
                           float out[DM])
{
    int i, j;

    /* Up project, then SiLU */
    for (i = 0; i < D_EXPERT; i++) {
        hid[i] = 0;
        for (j = 0; j < DM; j++)
            hid[i] += e->W_up[i][j] * x[j];
        hid[i] = silu(hid[i]);
    }

    /* Down project */
    for (i = 0; i < DM; i++) {
        out[i] = 0;
        for (j = 0; j < D_EXPERT; j++)
            out[i] += e->W_down[i][j] * hid[j];
    }
}

static float norm(const float *v, int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

int main(void)
{
    Expert e;
    float x[DM] = { 0.5f, -0.2f, 0.8f, 0.1f, 
        -0.3f, 0.6f };
    float hid[D_EXPERT], out[DM];
    int i, suppressed = 0;

    srand(42);
    expert_init(&e);
    expert_fwd(&e, x, hid, out);

    printf("Single expert FFN (d_model=%d, "
           "d_exp=%d):\n\n",
           DM, D_EXPERT);

    printf("  Input:   [");
    for (i = 0; i < DM; i++)
        printf("%+.3f%s", x[i], i<DM-1 ? ", " : "");
    printf("]\n");

    printf("  Hidden:  [");
    for (i = 0; i < D_EXPERT; i++)
        printf("%+.3f%s", hid[i],
            i<D_EXPERT-1 ? ", " : "");
    printf("]\n");

    printf("  Output:  [");
    for (i = 0; i < DM; i++)
        printf("%+.3f%s", out[i], i<DM-1 ? ", " : "");
    printf("]\n\n");

    for (i = 0; i < D_EXPERT; i++)
        if (hid[i] < 0) suppressed++;

    printf("  SiLU pushed %d of %d hidden units "
           "negative\n",
           suppressed, D_EXPERT);
    printf("  Input norm %.3f, "
           "output %.3f, gain %.2f\n",
           norm(x, DM), norm(out, DM), 
           norm(out, DM) / norm(x, DM));

    int params = D_EXPERT * DM + DM * D_EXPERT;
    printf("\n  Parameters per expert: %d\n", params);
    printf("  With 256 experts: %d stored, %d active\n",
           params * 256, params);

    return 0;
}
