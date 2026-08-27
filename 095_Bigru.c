/* 095_Bigru.c */
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

#define N 3
#define SEQ_LEN 8

typedef struct {
    float W_z[N], U_z[N][N], b_z[N];
    float W_r[N], U_r[N][N], b_r[N];
    float W_s[N], U_s[N][N], b_s[N];
}
GRUCell;

static void gru_step(const GRUCell *g, float x, 
                      const float hp[N], float hn[N])
{
    int i, j;
    for (i = 0; i < N; i++) {
        float zz = g->b_z[i] + g->W_z[i] * x;
        float zr = g->b_r[i] + g->W_r[i] * x;
        for (j = 0; j < N; j++) {
            zz += g->U_z[i][j] * hp[j];
            zr += g->U_r[i][j] * hp[j];
        }
        float z = sigmoid(zz), r = sigmoid(zr);
        float zs = g->b_s[i] + g->W_s[i] * x;
        for (j = 0; j < N; j++)
            zs += g->U_s[i][j] * (r * hp[j]);
        float s = my_tanh(zs);
        hn[i] = z * hp[i] + (1 - z) * s;
    }
}

static void gru_init(GRUCell *g)
{
    int i, j;
    for (i = 0; i < N; i++) {
        g->W_z[i] = randf()*0.4f-0.2f;
        g->b_z[i] = 0;
        g->W_r[i] = randf()*0.4f-0.2f;
        g->b_r[i] = 0;
        g->W_s[i] = randf()*0.4f-0.2f;
        g->b_s[i] = 0;
        for (j = 0; j < N; j++) {
            g->U_z[i][j] = randf()*0.2f-0.1f;
            g->U_r[i][j] = randf()*0.2f-0.1f;
            g->U_s[i][j] = randf()*0.2f-0.1f;
        }
    }
}

int main(void)
{
    GRUCell fwd_cell, bwd_cell;
    float seq[SEQ_LEN] = { 0.5f, -0.2f, 0.8f, 0.1f, 
                           -0.5f, 0.3f, 0.9f, -0.1f };

    float h_fwd[SEQ_LEN][N], h_bwd[SEQ_LEN][N];
    float h_zero[N] = {0};
    int t, i;

    srand(42);
    gru_init(&fwd_cell);
    gru_init(&bwd_cell);

    /* Forward GRU */
    for (t = 0; t < SEQ_LEN; t++)
        gru_step(&fwd_cell, seq[t], 
                 t == 0 ? h_zero : h_fwd[t-1], 
                 h_fwd[t]);

    /* Backward GRU */
    for (t = SEQ_LEN - 1; t >= 0; t--)
        gru_step(&bwd_cell, seq[t], 
                 t == SEQ_LEN-1 ? h_zero : h_bwd[t+1], 
                 h_bwd[t]);

    printf("Bidirectional GRU (BiGRU):\n\n");
    printf("  t  input   fwd             bwd     "
           "        |fwd|  |bwd|\n");
    printf("  -- ------  --------------- ---------"
           "------ ------ ------\n");

    for (t = 0; t < SEQ_LEN; t++) {
        float nf = 0, nb = 0;
        for (i = 0; i < N; i++) {
            nf += h_fwd[t][i] * h_fwd[t][i];
            nb += h_bwd[t][i] * h_bwd[t][i];
        }
        printf("  %d  %+4.1f   [%+.2f,%+.2f,%+.2f] "
               "[%+.2f,%+.2f,%+.2f] %.3f  %.3f\n",
               t, seq[t], 
               h_fwd[t][0], h_fwd[t][1], h_fwd[t][2], 
               h_bwd[t][0], h_bwd[t][1], h_bwd[t][2], 
               sqrtf(nf), sqrtf(nb));
    }

    printf("\nEach position has %d forward plus %d "
           "backward = %d combined.\n", N, N, 2*N);
    printf("The GRU gates handle long-range "
           "dependencies.\n");
    printf("The bidirectional structure handles "
           "both directions.\n");
    printf("Together they give long-range, "
           "full-context representations.\n");

    return 0;
}
