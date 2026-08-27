/* 175_Moe_Layer.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

#define DM 6
#define DE 8
#define N_SHARED 1
#define N_ROUTED 8
#define TOP_K 2

typedef struct {
    float W_up[DE][DM];
    float W_down[DM][DE];
}
Expert;

typedef struct {
    Expert shared[N_SHARED];
    Expert routed[N_ROUTED];
    float centroids[N_ROUTED][DM];
}
MoELayer;

static void expert_init(Expert *e)
{
    float a_up = sqrtf(3.0f / DM);
    float a_dn = sqrtf(3.0f / DE);
    int i, j;

    for (i = 0; i < DE; i++)
        for (j = 0; j < DM; j++)
            e->W_up[i][j] = (randf()*2-1) * a_up;
    for (i = 0; i < DM; i++)
        for (j = 0; j < DE; j++)
            e->W_down[i][j] = (randf()*2-1) * a_dn;
}

static void expert_fwd(const Expert *e, 
    const float x[DM], 
                       float out[DM])
{
    float hid[DE];
    int i, j;

    for (i = 0; i < DE; i++) {
        hid[i] = 0;
        for (j = 0; j < DM; j++)
            hid[i] += e->W_up[i][j] * x[j];
        hid[i] = silu(hid[i]);
    }
    for (i = 0; i < DM; i++) {
        out[i] = 0;
        for (j = 0; j < DE; j++)
            out[i] += e->W_down[i][j] * hid[j];
    }
}

static void moe_init(MoELayer *m)
{
    int k, j;

    for (k = 0; k < N_SHARED; k++)
        expert_init(&m->shared[k]);
    for (k = 0; k < N_ROUTED; k++)
        expert_init(&m->routed[k]);
    for (k = 0; k < N_ROUTED; k++)
        for (j = 0; j < DM; j++)
            m->centroids[k][j] = (randf()*2-1) * 0.3f;
}

/* Returns residual plus shared plus gated routed, and
   reports the norm each of the three contributed */
static void moe_fwd(const MoELayer *m, 
    const float u[DM], 
                    float out[DM], float part[3], 
                    int sel[TOP_K], float gate[TOP_K])
{
    float sh[DM] = {0}, rt[DM] = {0};
    float s[N_ROUTED], sum = 0;
    int i, j, k, t;

    for (k = 0; k < N_SHARED; k++) {
        float e[DM];
        expert_fwd(&m->shared[k], u, e);
        for (i = 0; i < DM; i++) sh[i] += e[i];
    }

    for (k = 0; k < N_ROUTED; k++) {
        float dot = 0;
        for (j = 0; j < DM; j++)
            dot += u[j] * m->centroids[k][j];
        s[k] = sigmoid(dot);
    }

    for (t = 0; t < TOP_K; t++) {
        int best = -1;
        float bs = -1;
        for (k = 0; k < N_ROUTED; k++) {
            int used = 0;
            for (j = 0; j < t; j++)
                if (sel[j] == k) used = 1;
            if (!used && s[k] > bs) {
                best = k;
                bs = s[k];
                }
        }
        sel[t] = best;
        gate[t] = s[best];
    }
    for (t = 0; t < TOP_K; t++) sum += gate[t];
    for (t = 0; t < TOP_K; t++) gate[t] /= sum;

    for (t = 0; t < TOP_K; t++) {
        float e[DM];
        expert_fwd(&m->routed[sel[t]], u, e);
        for (i = 0; i < DM; i++)
            rt[i] += gate[t] * e[i];
    }

    for (i = 0; i < DM; i++)
        out[i] = u[i] + sh[i] + rt[i];

    for (i = 0; i < 3; i++) part[i] = 0;
    for (i = 0; i < DM; i++) {
        part[0] += u[i] * u[i];
        part[1] += sh[i] * sh[i];
        part[2] += rt[i] * rt[i];
    }
    for (i = 0; i < 3; i++) part[i] = sqrtf(part[i]);
}

int main(void)
{
    MoELayer moe;
    float x[DM] = { 0.5f, -0.2f, 0.8f, 0.1f, 
        -0.3f, 0.6f };
    float out[DM], part[3], gate[TOP_K];
    int sel[TOP_K], i;

    srand(42);
    moe_init(&moe);
    moe_fwd(&moe, x, out, part, sel, gate);

    printf("DeepSeekMoE layer:\n");
    printf("  %d shared expert always "
           "active\n", N_SHARED);
    printf("  %d routed experts, top-%d per token\n\n",
           N_ROUTED, TOP_K);

    printf("  Input:  [");
    for (i = 0; i < DM; i++)
        printf("%+.3f%s", x[i], i<DM-1 ? ", " : "");
    printf("]\n  Output: [");
    for (i = 0; i < DM; i++)
        printf("%+.3f%s", out[i], i<DM-1 ? ", " : "");
    printf("]\n\n");

    printf("  Routed to %d and %d, gates %.3f %.3f\n",
           sel[0], sel[1], gate[0], gate[1]);
    printf("  Contribution by norm:\n");
    printf("    residual  %.3f\n", part[0]);
    printf("    shared    %.3f\n", part[1]);
    printf("    routed    %.3f\n\n", part[2]);

    int per_expert = DE * DM + DM * DE;
    int router = N_ROUTED * DM;
    int total = per_expert
        * (N_ROUTED + N_SHARED) + router;
    int active = per_expert
        * (TOP_K + N_SHARED) + router;

    printf("  Parameters:\n");
    printf("    Per expert    %d\n", per_expert);
    printf("    Router        %d\n", router);
    printf("    Stored        %d\n", total);
    printf("    Active/token  %d\n", active);
    printf("    Ratio         %.2fx "
           "stored per active\n",
           (float)total / active);
    printf("    Experts only  %.2fx\n",
           (float)(per_expert * (N_ROUTED + N_SHARED))
           / (per_expert * (TOP_K + N_SHARED)));

    return 0;
}
