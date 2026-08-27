/* 107_Qkv_Sequence.c */
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 4
#define D_MODEL 4
#define D_K 3

static void softmax(float *x, int n)
{
    float mx = x[0], s = 0;
    int i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

/* out = W * x, with W stored as d_out rows of d_in */
static void project(const float W[D_K][D_MODEL], 
                    const float x[D_MODEL], 
                    float out[D_K])
{
    int i, j;
    for (i = 0; i < D_K; i++) {
        out[i] = 0;
        for (j = 0; j < D_MODEL; j++)
            out[i] += W[i][j] * x[j];
    }
}

int main(void)
{
    float X[SEQ_LEN][D_MODEL] = {
        {  1.0f,  0.0f,  0.5f, -0.2f },   /* the  */
        {  0.3f,  0.8f, -0.1f,  0.6f },   /* cat  */
        { -0.2f,  0.1f,  0.9f,  0.3f },   /* sat  */
        {  0.5f, -0.3f,  0.2f,  0.7f },   /* down */
    };
    const char *words[SEQ_LEN] = {
        "the", "cat", "sat", "down"
    };

    /* Chosen rather than trained, so the pattern is
       readable. Training would find its own. */
    float W_Q[D_K][D_MODEL] = {
        { -0.54f, +0.23f, +1.14f, +0.16f }, 
        { +1.01f, +0.59f, +3.72f, -0.67f }, 
        { +1.27f, +0.30f, -1.02f, +3.80f }, 
    };
    float W_K[D_K][D_MODEL] = {
        { +0.92f, -0.40f, +0.21f, +0.11f }, 
        { +0.16f, +0.92f, -0.18f, +0.33f }, 
        { -0.26f, -0.38f, +0.83f, +0.78f }, 
    };

    float Q[SEQ_LEN][D_K], K[SEQ_LEN][D_K];
    float w[SEQ_LEN];
    int i, j, k, best;

    for (i = 0; i < SEQ_LEN; i++) {
        project(W_Q, X[i], Q[i]);
        project(W_K, X[i], K[i]);
    }

    printf("Q/K/V attention on a 4 word sequence\n\n");
    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < SEQ_LEN; j++) {
            w[j] = 0;
            for (k = 0; k < D_K; k++)
                w[j] += Q[i][k] * K[j][k];
        }
        softmax(w, SEQ_LEN);

        best = 0;
        for (j = 1; j < SEQ_LEN; j++)
            if (w[j] > w[best]) best = j;

        printf("  \"%-4s\" attends to: ", words[i]);
        for (j = 0; j < SEQ_LEN; j++)
            printf("%s=%.2f ", words[j], w[j]);
        printf("\n      strongest: \"%s\"\n\n",
               words[best]);
    }

    printf("Every position builds its own query, so\n");
    printf("every row differs. W_Q and W_K are\n");
    printf("shared across positions, so it is the\n");
    printf("input that differs, not the weights.\n");

    return 0;
}
