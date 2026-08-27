/* 176_Load_Balance.c */
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

#define DM 4
#define N_EXPERTS 6
#define TOP_K 2
#define N_TOKENS 600
#define ROUNDS 200

static float centroids[N_EXPERTS][DM];

/* Route one fresh batch, filling load[] */
static void batch(const float bias[N_EXPERTS], 
                  int load[N_EXPERTS])
{
    int t, i, j, k;

    for (i = 0; i < N_EXPERTS; i++) load[i] = 0;

    for (t = 0; t < N_TOKENS; t++) {
        float x[DM], s[N_EXPERTS];
        int sel[TOP_K];

        for (j = 0; j < DM; j++)
            x[j] = (randf()*2-1) * 0.5f;

        for (k = 0; k < N_EXPERTS; k++) {
            float dot = 0;
            for (j = 0; j < DM; j++)
                dot += x[j] * centroids[k][j];
            /* Bias enters the routing decision only */
            s[k] = sigmoid(dot) + bias[k];
        }

        for (i = 0; i < TOP_K; i++) {
            int best = -1;
            float bs = -1e9f;
            for (k = 0; k < N_EXPERTS; k++) {
                int used = 0;
                for (j = 0; j < i; j++)
                    if (sel[j] == k) used = 1;
                if (!used && s[k] > bs) {
                    best = k;
                    bs = s[k];
                }
            }
            sel[i] = best;
        }
        for (i = 0; i < TOP_K; i++) load[sel[i]]++;
    }
}

static float imbalance(const int load[N_EXPERTS])
{
    int i, lo = load[0], hi = load[0];
    for (i = 1; i < N_EXPERTS; i++) {
        if (load[i] < lo) lo = load[i];
        if (load[i] > hi) hi = load[i];
    }
    return lo ? (float)hi / lo : 999.0f;
}

/* Imbalance of an evenly random assignment, which is
   the best any balancing scheme could reach */
static float floor_imbalance(void)
{
    int load[N_EXPERTS], i, t, r;
    float acc = 0;

    for (r = 0; r < 50; r++) {
        for (i = 0; i < N_EXPERTS; i++) load[i] = 0;
        for (t = 0; t < N_TOKENS * TOP_K; t++)
            load[rand() % N_EXPERTS]++;
        acc += imbalance(load);
    }
    return acc / 50;
}

int main(void)
{
    float bias[N_EXPERTS] = {0};
    float gamma = 0.001f;
    int load[N_EXPERTS];
    int i, j, r;
    float off = 0, on = 0;

    srand(42);
    for (i = 0; i < N_EXPERTS; i++)
        for (j = 0; j < DM; j++)
            centroids[i][j] = (randf()*2-1) * 0.3f;

    printf("Auxiliary-loss-free load balancing\n");
    printf("  %d tokens per batch, top-%d of %d "
           "experts\n",
           N_TOKENS, TOP_K, N_EXPERTS);
    printf("  expected load %d, bias step %.3f\n\n",
           N_TOKENS * TOP_K / N_EXPERTS, gamma);

    /* Control: the bias stays at zero throughout */
    for (r = 0; r < ROUNDS; r++) {
        batch(bias, load);
        if (r >= ROUNDS - 50) off
                += imbalance(load) / 50;
        if (r == 0) {
            printf("  bias off, round 0:   ");
            for (i = 0; i < N_EXPERTS; i++)
                printf("%4d", load[i]);
            printf("   %.2f\n", imbalance(load));
        }
    }

    /* Treatment: the same router, bias now updating */
    for (r = 0; r < ROUNDS; r++) {
        batch(bias, load);
        if (r >= ROUNDS - 50) on
                += imbalance(load) / 50;
        if (r == 0 || r == ROUNDS - 1) {
            printf("  bias on,  round %-3d: ", r);
            for (i = 0; i < N_EXPERTS; i++)
                printf("%4d", load[i]);
            printf("   %.2f\n", imbalance(load));
        }
        for (i = 0; i < N_EXPERTS; i++) {
            if (load[i] > N_TOKENS * TOP_K / N_EXPERTS)
                bias[i] -= gamma;
            else
                bias[i] += gamma;
        }
    }

    printf("\n  mean max/min over the last 50 "
           "rounds\n");
    printf("    bias held at zero   %.3f\n", off);
    printf("    bias updating       %.3f\n", on);
    printf("    even random routing %.3f\n",
           floor_imbalance());

    printf("\n  final bias  ");
    for (i = 0; i < N_EXPERTS; i++)
        printf("%+.3f ", bias[i]);
    printf("\n\n");

    printf("  The bias enters the top-K comparison "
           "and\n");
    printf("  nothing else, so gate values still "
           "come\n");
    printf("  from the unbiased score and no loss "
           "term\n");
    printf("  is involved. That is equation 16.\n");

    return 0;
}
