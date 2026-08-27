/* 059_Embeddings_Learn.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* Embedding + simple predictor: given two parts,
   predict
   if they appear in the same circuit (1) or not (0) */

#define N_PARTS 6
#define EMBED_DIM 3

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

/* Predict similarity: dot product of embeddings ->
   sigmoid */
static float predict(
        const float embed[N_PARTS][EMBED_DIM],
                     int part_a, int part_b)
{
    float dot = 0;
    int i;
    for (i = 0; i < EMBED_DIM; i++)
        dot += embed[part_a][i] * embed[part_b][i];
    return sigmoid(dot);
}

int main(void)
{
    float embed[N_PARTS][EMBED_DIM];
    const char *names[] = { "resistor",
        "capacitor", "inductor",
                            "LED", "motor", "servo" };

    /* Training pairs: (part_a, part_b, co-occur?) */
    int pairs[][3] = {
        /* resistor + capacitor: yes (RC filter) */
        {0, 1, 1}, 
        /* resistor + inductor: yes (RL filter) */
        {0, 2, 1}, 
        /* capacitor + inductor: yes (LC tank) */
        {1, 2, 1}, 
        /* resistor + LED: yes (current limiting) */
        {0, 3, 1}, 
        {4, 5, 1},  /* motor + servo: yes (robotics) */
        {0, 4, 0},  /* resistor + motor: rarely */
        {1, 5, 0},  /* capacitor + servo: rarely */
        {2, 4, 0},  /* inductor + motor: rarely */
        {3, 5, 0},  /* LED + servo: rarely */
    };
    int n_pairs = 9;

    float lr = 0.5f;
    int epoch, p, i;

    srand(42);
    for (i = 0; i < N_PARTS * EMBED_DIM; i++)
        ((float*)embed)[i] = randf() * 0.4f - 0.2f;

    /* Train */
    for (epoch = 0; epoch < 500; epoch++) {
        for (p = 0; p < n_pairs; p++) {
            int a = pairs[p][0], b = pairs[p][1];
            float t = (float)pairs[p][2];
            float y = predict(embed, a, b);
            float err = y - t;

            /* Gradient: d_loss/d_embed[a][i] = err *
               y*(1-y) * embed[b][i]
                         d_loss/d_embed[b][i] = err * y
                             *(1-y) * embed[a][i] */
            float d = err * y * (1.0f - y);
            for (i = 0; i < EMBED_DIM; i++) {
                float ga = d * embed[b][i];
                float gb = d * embed[a][i];
                embed[a][i] -= lr * ga;
                embed[b][i] -= lr * gb;
            }
        }
    }

    /* Print learned embeddings */
    printf("Learned embeddings "
           "(%d-dimensional):\n\n", EMBED_DIM);
    for (i = 0; i < N_PARTS; i++)
        printf("  %-10s [%+.3f, %+.3f, %+.3f]\n",
               names[i], embed[i][0], embed[i][1], 
                   embed[i][2]);

    /* Print distances */
    printf("\nDistances between parts:\n\n");
    int a, b;
    for (a = 0; a < N_PARTS; a++) {
        for (b = a + 1; b < N_PARTS; b++) {
            float dist = 0;
            for (i = 0; i < EMBED_DIM; i++) {
                float d = embed[a][i] - embed[b][i];
                dist += d * d;
            }
            dist = sqrtf(dist);
            printf("  %-10s <-> %-10s  dist=%.3f\n",
                   names[a], names[b], dist);
        }
    }

    return 0;
}
