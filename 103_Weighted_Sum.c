/* 103_Weighted_Sum.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

#define DIM 4
#define N_KEYS 4

static float dot(const float *a, const float *b, int n)
{
    float sum = 0;
    int i;
    for (i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void softmax(float *x, int n)
{
    float mx = -FLT_MAX, s = 0;
    int i;
    for (i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

int main(void)
{
    /* Decoder state (the query) */
    float query[DIM] = { 0.5f, 0.8f, -0.2f, 0.3f };

    /* Encoder hidden states at each position */
    float keys[N_KEYS][DIM] = {
        { 0.1f, 0.9f, -0.1f, 0.2f },   /* "the" */
        { 0.7f, 0.2f,  0.5f, 0.8f },   /* "cat" */
        { 0.3f, 0.1f, -0.4f, 0.1f },   /* "sat" */
        { 0.0f, 0.5f,  0.2f, -0.1f },  /* "on"  */
    };
    /* Values (same as keys in basic attention) */
    float values[N_KEYS][DIM];
    int i, j;
    for (i = 0; i < N_KEYS; i++)
        for (j = 0; j < DIM; j++)
            values[i][j] = keys[i][j];

    const char *words[] = { "the", "cat", "sat", "on" };

    /* Step 1: compute scores */
    float scores[N_KEYS];
    for (i = 0; i < N_KEYS; i++)
        scores[i] = dot(query, keys[i], DIM);

    /* Step 2: softmax */
    softmax(scores, N_KEYS);

    /* Step 3: weighted sum of values */
    float context[DIM] = { 0 };
    for (i = 0; i < N_KEYS; i++)
        for (j = 0; j < DIM; j++)
            context[j] += scores[i] * values[i][j];

    printf("Attention mechanism step by step:\n\n");
    printf("1. Compute similarity (dot product):\n");
    for (i = 0; i < N_KEYS; i++)
        printf("   query . key[\"%s\"]"
               " = %.3f\n", words[i],
               dot(query, keys[i], DIM));

    printf("\n2. Apply softmax (get weights):\n");
    for (i = 0; i < N_KEYS; i++)
        printf("   weight[\"%s\"] = %.3f\n",
               words[i], scores[i]);

    printf("\n3. Weighted sum of values, the "
           "context vector\n");
    printf("   context = ");
    for (i = 0; i < N_KEYS; i++)
        printf("%.3f*v[\"%s\"] %s", scores[i], words[i],
               i < N_KEYS-1 ? "+ " : "");
    printf("\n   context = [%.3f, %.3f, %.3f, %.3f]\n",
           context[0], context[1], 
               context[2], context[3]);

    printf("\nThe context vector is a blend of all "
           "encoder states,\n");
    printf("weighted by relevance to the current "
           "decoder step.\n");
    printf("It replaces the single bottleneck vector "
           "from Chapter 18.\n");

    return 0;
}
