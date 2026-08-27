/* 089_Gru_Cell.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N 4

typedef struct {
    float W_z[N], U_z[N][N], b_z[N];  /* update gate */
    float W_r[N], U_r[N][N], b_r[N];  /* reset gate */
    float W_s[N], U_s[N][N], b_s[N];  /* candidate */
    float W_y[N], b_y;                 /* output */
}
GRU;

static void gru_step(const GRU *g, float x, 
                      const float h_prev[N], 
                          float h_new[N])
{
    float z[N], r[N], s[N];
    int i, j;

    for (i = 0; i < N; i++) {
        /* Update gate */
        float zz = g->b_z[i] + g->W_z[i] * x;
        /* Reset gate */
        float zr = g->b_r[i] + g->W_r[i] * x;

        for (j = 0; j < N; j++) {
            zz += g->U_z[i][j] * h_prev[j];
            zr += g->U_r[i][j] * h_prev[j];
        }

        z[i] = sigmoid(zz);
        r[i] = sigmoid(zr);

        /* Candidate: uses r * h_prev */
        float zs = g->b_s[i] + g->W_s[i] * x;
        for (j = 0; j < N; j++)
            zs += g->U_s[i][j] * (r[i] * h_prev[j]);
        s[i] = my_tanh(zs);

        /* New state: interpolation */
        h_new[i] = z[i] * h_prev[i]
            + (1.0f - z[i]) * s[i];
    }
}

static void gru_init(GRU *g)
{
    int i, j;
    for (i = 0; i < N; i++) {
        g->W_z[i] = randf()*0.2f-0.1f;
        g->b_z[i] = 0;
        g->W_r[i] = randf()*0.2f-0.1f;
        g->b_r[i] = 0;
        g->W_s[i] = randf()*0.2f-0.1f;
        g->b_s[i] = 0;
        g->W_y[i] = randf()*0.4f-0.2f;
        for (j = 0; j < N; j++) {
            g->U_z[i][j] = randf()*0.2f-0.1f;
            g->U_r[i][j] = randf()*0.2f-0.1f;
            g->U_s[i][j] = randf()*0.2f-0.1f;
        }
    }
    g->b_y = 0;
}

int main(void)
{
    GRU g;
    float h[N] = {0}, h_new[N];
    float seq[] = { 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
    int len = 10;
    int t, i;

    srand(42);
    gru_init(&g);

    printf("GRU cell processing a sequence:\n\n");
    printf("  t  x  h[0]    h[1]    h[2]    h[3]\n");
    printf("  -- -- ------  ------  ------  ------\n");

    for (t = 0; t < len; t++) {
        gru_step(&g, seq[t], h, h_new);
        printf("  %2d  %.0f  %+.3f  %+.3f  %+.3f  "
               "%+.3f\n",
               t, seq[t], h_new[0], h_new[1], 
               h_new[2], h_new[3]);
        for (i = 0; i < N; i++) h[i] = h_new[i];
    }

    printf("\nThe GRU has one state vector, with no "
           "separate cell state.\n");
    printf("The update gate z controls retention "
           "vs replacement.\n");

    return 0;
}
