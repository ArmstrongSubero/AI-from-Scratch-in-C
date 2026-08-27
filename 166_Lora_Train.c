/* 166_Lora_Train.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define D 4
#define R 4   /* the largest rank swept below */

typedef struct {
    float W[D][D];     /* frozen */
    float A[R][D];     /* trainable */
    float B[D][R];     /* trainable */
}
LoRA;

static void lora_fwd(const LoRA *l, int r, 
                     const float x[D], float out[D], 
                     float mid[R])
{
    int i, j;
    for (i = 0; i < D; i++) {
        out[i] = 0;
        for (j = 0; j < D; j++)
            out[i] += l->W[i][j] * x[j];
    }
    for (i = 0; i < r; i++) {
        mid[i] = 0;
        for (j = 0; j < D; j++)
            mid[i] += l->A[i][j] * x[j];
    }
    for (i = 0; i < D; i++)
        for (j = 0; j < r; j++)
            out[i] += l->B[i][j] * mid[j];
}

static float train_step(LoRA *l, int r, 
    const float x[D], 
                        const float target[D], float lr)
{
    float out[D], mid[R];
    int i, j, k;

    lora_fwd(l, r, x, out, mid);

    float loss = 0;
    float d_out[D];
    for (i = 0; i < D; i++) {
        d_out[i] = 2.0f * (out[i] - target[i]);
        loss += (out[i] - target[i])
                * (out[i] - target[i]);
    }

    /* d_mid reads B, so it is computed before B is
       updated. Differentiating through the new
       values would give a wrong gradient without
       any warning that it had. */
    float d_mid[R];
    for (i = 0; i < r; i++) {
        d_mid[i] = 0;
        for (k = 0; k < D; k++)
            d_mid[i] += d_out[k] * l->B[k][i];
    }

    for (i = 0; i < D; i++)
        for (j = 0; j < r; j++)
            l->B[i][j] -= lr * d_out[i] * mid[j];

    for (i = 0; i < r; i++)
        for (j = 0; j < D; j++)
            l->A[i][j] -= lr * d_mid[i] * x[j];

    /* W is never touched */
    return loss;
}

int main(void)
{
    /* Reverse the order of a 4-vector. W is frozen
       and random, so the LoRA update has to supply
       the whole difference between W and the
       permutation, which is a general 4x4 matrix. */
    float inputs[D][D] = {
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 
            0}, {0, 0, 0, 1}
    };
    float targets[D][D] = {
        {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 
            0}, {1, 0, 0, 0}
    };

    float W0[D][D];
    int i, j, r, ep, p;

    srand(42);
    for (i = 0; i < D; i++)
        for (j = 0; j < D; j++)
            W0[i][j] = (randf()*2-1) * 0.3f;

    printf("LoRA at four ranks, W frozen "
           "throughout\n\n");
    printf("  rank  trained  final loss  worst "
           "output\n");
    printf("  ----  -------  ----------  "
           "------------\n");

    for (r = 1; r <= D; r++) {
        LoRA l;
        for (i = 0; i < D; i++)
            for (j = 0; j < D; j++)
                l.W[i][j] = W0[i][j];
        srand(7);
        for (i = 0; i < R; i++)
            for (j = 0; j < D; j++)
                l.A[i][j] = (randf()*2-1) * 0.5f;
        /* B starts at zero so the adapter begins as
           a no-op and the model is unchanged */
        for (i = 0; i < D; i++)
            for (j = 0; j < R; j++) l.B[i][j] = 0;

        float loss = 0;
        for (ep = 0; ep < 4000; ep++) {
            loss = 0;
            for (p = 0; p < D; p++)
                loss += train_step(&l, r, inputs[p], 
                                   targets[p], 0.02f);
            loss /= D;
        }

        /* How far is the worst position from its
           target across all four test vectors? */
        float worst = 0;
        for (p = 0; p < D; p++) {
            float out[D], mid[R];
            lora_fwd(&l, r, inputs[p], out, mid);
            for (i = 0; i < D; i++) {
                float e = fabsf(out[i] - targets[p][i]);
                if (e > worst) worst = e;
            }
        }

        printf("  %4d  %7d  %10.4f  %12.4f\n",
               r, 2 * D * r, loss, worst);
    }

    printf("\n  Rank 4 drives the loss to nothing. "
           "The\n");
    printf("  lower ranks cannot, and the reason is\n");
    printf("  structural rather than a matter of "
           "more\n");
    printf("  training. W is frozen, so the adapter\n");
    printf("  must supply the entire difference\n");
    printf("  between W and the permutation, and "
           "that\n");
    printf("  difference is a general 4x4 matrix of\n");
    printf("  rank 4. A rank-2 update cannot reach "
           "it\n");
    printf("  however long it runs.\n\n");

    printf("  At this size rank 4 also costs 32\n");
    printf("  parameters against the 16 in W, so "
           "LoRA\n");
    printf("  is a loss here. It pays only when d "
           "is\n");
    printf("  large and the useful rank stays "
           "small.\n");

    return 0;
}
