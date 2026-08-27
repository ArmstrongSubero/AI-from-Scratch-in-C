/* 111_Single_Head.c */
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 4
#define D_MODEL 6
#define D_HEAD 3

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

/* One attention head. Called once per head with that
   head's own projection matrices. */
static void head(const float X[SEQ_LEN][D_MODEL], 
                 const float W_Q[D_HEAD][D_MODEL], 
                 const float W_K[D_HEAD][D_MODEL], 
                 float weights[SEQ_LEN][SEQ_LEN])
{
    float Q[SEQ_LEN][D_HEAD], K[SEQ_LEN][D_HEAD];
    float scale = 1.0f / sqrtf((float)D_HEAD);
    int i, j, k, m;

    for (i = 0; i < SEQ_LEN; i++)
        for (k = 0; k < D_HEAD; k++) {
            Q[i][k] = 0;
            K[i][k] = 0;
            for (m = 0; m < D_MODEL; m++) {
                Q[i][k] += W_Q[k][m] * X[i][m];
                K[i][k] += W_K[k][m] * X[i][m];
            }
        }

    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < SEQ_LEN; j++) {
            weights[i][j] = 0;
            for (k = 0; k < D_HEAD; k++)
                weights[i][j] += Q[i][k] * K[j][k];
            weights[i][j] *= scale;
        }
        softmax(weights[i], SEQ_LEN);
    }
}

static void show(const char *title, const char *w[], 
                 float W[SEQ_LEN][SEQ_LEN])
{
    int i, j;
    printf("%s\n", title);
    printf("          ");
    for (j = 0; j < SEQ_LEN; j++) printf("%-6s", w[j]);
    printf("\n");
    for (i = 0; i < SEQ_LEN; i++) {
        printf("  %-6s  ", w[i]);
        for (j = 0; j < SEQ_LEN; j++)
            printf("%.2f  ", W[i][j]);
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float X[SEQ_LEN][D_MODEL] = {
        { 1.0f, 0.2f, -0.3f, 0.5f, -0.1f, 0.8f }, 
        { 0.3f, 0.9f, 0.1f, -0.2f, 0.6f, 0.4f }, 
        { -0.2f, 0.4f, 0.8f, 0.3f, -0.5f, 0.1f }, 
        { 0.6f, -0.1f, 0.2f, 0.9f, 0.3f, -0.4f }, 
    };
    const char *words[SEQ_LEN] = {
        "The", "cat", "sat", "down"
    };

    /* Head 0, chosen to pick out syntactic links */
    float W_Q0[D_HEAD][D_MODEL] = {
        { -0.29f, +0.54f, +1.15f, +0.43f, 
            -0.91f, +0.28f }, 
        { +1.26f, +1.03f, +2.02f, +2.22f, 
            -3.58f, +2.71f }, 
        { +1.40f, +2.66f, +1.56f, +1.52f, 
            +4.53f, -2.21f }, 
    };
    float W_K0[D_HEAD][D_MODEL] = {
        { +0.54f, -0.09f, -0.11f, +0.47f, 
            -0.13f, +0.24f }, 
        { +0.06f, +0.63f, +0.14f, -0.19f, 
            +0.60f, +0.02f }, 
        { -0.01f, +0.28f, +0.76f, +0.55f, 
            -0.26f, -0.15f }, 
    };
    /* Head 1, chosen to look at the previous
       position */
    float W_Q1[D_HEAD][D_MODEL] = {
        { +2.02f, +2.52f, -0.38f, -0.18f, 
            +1.01f, +2.57f }, 
        { -0.73f, +1.36f, +2.86f, +1.06f, 
            -2.26f, +0.70f }, 
        { +1.14f, -0.16f, +0.91f, +2.38f, 
            +1.84f, -2.30f }, 
    };
    float W_K1[D_HEAD][D_MODEL] = {
        { +0.57f, -0.09f, -0.09f, +0.52f, 
            -0.09f, +0.19f }, 
        { +0.23f, +0.60f, +0.28f, +0.18f, 
            +0.88f, -0.34f }, 
        { -0.16f, +0.30f, +0.64f, +0.24f, 
            -0.50f, +0.15f }, 
    };

    float W0[SEQ_LEN][SEQ_LEN], W1[SEQ_LEN][SEQ_LEN];

    head(X, W_Q0, W_K0, W0);
    head(X, W_Q1, W_K1, W1);

    show("Head 0 attention weights", words, W0);
    show("Head 1 attention weights", words, W1);

    printf("Same input, same code, "
           "different weights.\n");
    printf("Head 0 links each word to a syntactic\n");
    printf("partner. Head 1 looks "
           "one position back.\n");
    printf("A single head has to pick one of these.\n");

    return 0;
}
