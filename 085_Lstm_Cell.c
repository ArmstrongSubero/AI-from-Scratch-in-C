/* 085_Lstm_Cell.c */
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

#define N 4  /* cell/hidden size */

typedef struct {
    /* Four gates, each with an input weight,
       a recurrent weight and a bias */
    float W_f[N], U_f[N][N], b_f[N];  /* forget */
    float W_i[N], U_i[N][N], b_i[N];  /* input */
    float W_o[N], U_o[N][N], b_o[N];  /* output */
    float W_g[N], U_g[N][N], b_g[N];  /* candidate */
    /* Output projection */
    float W_y[N];
    float b_y;
}
LSTM;

static void lstm_step(const LSTM *l, float x, 
                       const float h_prev[N], 
                       const float c_prev[N], 
                       float h_new[N], float c_new[N])
{
    float f[N], ig[N], o[N], g[N];
    int i, j;

    for (i = 0; i < N; i++) {
        float zf = l->b_f[i] + l->W_f[i] * x;
        float zi = l->b_i[i] + l->W_i[i] * x;
        float zo = l->b_o[i] + l->W_o[i] * x;
        float zg = l->b_g[i] + l->W_g[i] * x;

        for (j = 0; j < N; j++) {
            zf += l->U_f[i][j] * h_prev[j];
            zi += l->U_i[i][j] * h_prev[j];
            zo += l->U_o[i][j] * h_prev[j];
            zg += l->U_g[i][j] * h_prev[j];
        }

        f[i] = sigmoid(zf);
        ig[i] = sigmoid(zi);
        o[i] = sigmoid(zo);
        g[i] = my_tanh(zg);

        /* Cell update: c = f * c_old + i * g */
        c_new[i] = f[i] * c_prev[i] + ig[i] * g[i];

        /* Hidden state: h = o * tanh(c) */
        h_new[i] = o[i] * my_tanh(c_new[i]);
    }
}

static float lstm_output(const LSTM *l, 
    const float h[N])
{
    float y = l->b_y;
    int i;
    for (i = 0; i < N; i++)
        y += l->W_y[i] * h[i];
    return y;
}

static void lstm_init(LSTM *l)
{
    int i, j;
    for (i = 0; i < N; i++) {
        /* forget bias = 1 */
        l->W_f[i] = randf()*0.2f-0.1f;
        l->b_f[i] = 1.0f;
        l->W_i[i] = randf()*0.2f-0.1f;
        l->b_i[i] = 0.0f;
        l->W_o[i] = randf()*0.2f-0.1f;
        l->b_o[i] = 0.0f;
        l->W_g[i] = randf()*0.2f-0.1f;
        l->b_g[i] = 0.0f;
        l->W_y[i] = randf()*0.4f-0.2f;
        for (j = 0; j < N; j++) {
            l->U_f[i][j] = randf()*0.2f-0.1f;
            l->U_i[i][j] = randf()*0.2f-0.1f;
            l->U_o[i][j] = randf()*0.2f-0.1f;
            l->U_g[i][j] = randf()*0.2f-0.1f;
        }
    }
    l->b_y = 0;
}

int main(void)
{
    LSTM l;
    float h[N] = {0}, c[N] = {0};
    float h_new[N], c_new[N];
    float seq[] = { 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
    int len = 10;
    int t, i;

    srand(42);
    lstm_init(&l);

    printf("LSTM cell processing a sequence:\n");
    printf("(First bit is 1, then zeros, "
           "then another 1)\n\n");
    printf("  t  x  h[0]    c[0]    c[1]    output\n");
    printf("  -- -- ------  ------  ------  ------\n");

    for (t = 0; t < len; t++) {
        lstm_step(&l, seq[t], h, c, h_new, c_new);
        float y = lstm_output(&l, h_new);

        printf("  %2d  %.0f  %+.3f  %+.3f  %+.3f  "
               "%+.3f\n",
               t, seq[t], h_new[0], c_new[0], 
                   c_new[1], y);

        for (i = 0; i < N; i++) {
            h[i] = h_new[i];
            c[i] = c_new[i];
        }
    }

    printf("\nWatch the cell state (c[0], c[1]). "
           "Unlike the basic\n");
    printf("RNN hidden state, the cell state "
           "decays far more slowly\n");
    printf("across zero-input steps, because the "
           "forget gate starts\n");
    printf("at sigmoid(1.0), which is 0.73 rather "
           "than 0.\n");

    return 0;
}
