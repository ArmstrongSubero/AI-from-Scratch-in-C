/* 108_Why_Qkv.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

#define N 3
#define D 4

static float dot(const float *a, const float *b, int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += a[i] * b[i];
    return s;
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
    /* Three vectors that are geometrically similar but
       semantically different */
    float X[N][D] = {
        { 0.9f, 0.1f, 0.8f, 0.2f },   /* word A */
        /* word B, similar to A */
        { 0.8f, 0.2f, 0.7f, 0.3f }, 
        /* word C, different */
        { 0.1f, 0.9f, 0.2f, 0.8f }, 
    };

    printf("Raw dot-product attention, "
           "no projections\n\n");
    int i, j;
    for (i = 0; i < N; i++) {
        float scores[N];
        for (j = 0; j < N; j++)
            scores[j] = dot(X[i], X[j], D);
        softmax(scores, N);
        printf("  Word %c attends to: "
               "A=%.3f  B=%.3f  C=%.3f\n",
               'A' + i, scores[0], 
                   scores[1], scores[2]);
    }

    printf("\nA and B always attend to each other, "
           "being similar.\n");
    printf("The network cannot learn a different "
           "pattern.\n\n");

    /* With projections, A can attend to C */
    printf("With learned Q/K projections:\n\n");

    /* W_Q designed so A's query aligns with C's key */
    float W_Q[D][D] = {
        { 0, 0, 0, 1 }, 
        { 0, 0, 1, 0 }, 
        { 0, 1, 0, 0 }, 
        { 1, 0, 0, 0 }, 
    };
    /* W_K = identity (keys unchanged) */
    float W_K[D][D] = {
        { 1, 0, 0, 0 }, 
        { 0, 1, 0, 0 }, 
        { 0, 0, 1, 0 }, 
        { 0, 0, 0, 1 }, 
    };

    float Q[N][D], K[N][D];
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            Q[i][j] = 0;
            K[i][j] = 0;
            int k;
            for (k = 0; k < D; k++) {
                Q[i][j] += W_Q[j][k] * X[i][k];
                K[i][j] += W_K[j][k] * X[i][k];
            }
        }
    }

    for (i = 0; i < N; i++) {
        float scores[N];
        for (j = 0; j < N; j++)
            scores[j] = dot(Q[i], K[j], D);
        softmax(scores, N);
        printf("  Word %c attends to: "
               "A=%.3f  B=%.3f  C=%.3f\n",
               'A' + i, scores[0], 
                   scores[1], scores[2]);
    }

    printf("\nThe projection reversed the query "
           "dimensions.\n");
    printf("Now A's query matches C's key better "
           "than B's.\n");
    printf("The network learned to look for "
           "something different\n");
    printf("from what the raw vectors "
           "would suggest.\n");

    return 0;
}
