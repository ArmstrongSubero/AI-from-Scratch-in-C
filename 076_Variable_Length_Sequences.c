/* 076_Variable_Length_Sequences.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_HID 4
#define MAX_LEN 20

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

typedef struct {
    float W_x[N_HID];
    float W_h[N_HID][N_HID];
    float b_h[N_HID];
    float W_y[N_HID];
    float b_y;
}
RNN;

static void rnn_forward(const RNN *n, 
                         const float *x, int len, 
                         float h[MAX_LEN+1][N_HID], 
                         float *y)
{
    int t, i, j;
    for (i = 0; i < N_HID; i++) h[0][i] = 0;
    for (t = 0; t < len; t++) {
        for (i = 0; i < N_HID; i++) {
            float z = n->b_h[i] + n->W_x[i] * x[t];
            for (j = 0; j < N_HID; j++)
                z += n->W_h[i][j] * h[t][j];
            h[t+1][i] = my_tanh(z);
        }
        y[t] = n->b_y;
        for (j = 0; j < N_HID; j++)
            y[t] += n->W_y[j] * h[t+1][j];
    }
}

static void rnn_backward(const RNN *n, 
                          const float *x, 
                          const float *tgt, int len, 
              const float h[MAX_LEN+1][N_HID],
                          const float *y, RNN *g)
{
    int t, i, j;
    float dh_next[N_HID];
    memset(g, 0, sizeof(RNN));
    memset(dh_next, 0, sizeof(dh_next));
    for (t = len-1; t >= 0; t--) {
        float dy = 2*(y[t] - tgt[t]), 
            dh[N_HID], dz[N_HID];
        for (j = 0; j < N_HID; j++)
            g->W_y[j] += dy * h[t+1][j];
        g->b_y += dy;
        for (i = 0; i < N_HID; i++)
            dh[i] = dy * n->W_y[i] + dh_next[i];
        for (i = 0; i < N_HID; i++) {
            float hv = h[t+1][i];
            dz[i] = dh[i] * (1 - hv*hv);
        }
        for (i = 0; i < N_HID; i++)
            g->W_x[i] += dz[i] * x[t];
        for (i = 0; i < N_HID; i++)
            for (j = 0; j < N_HID; j++)
                g->W_h[i][j] += dz[i] * h[t][j];
        for (i = 0; i < N_HID; i++) g->b_h[i] += dz[i];
        memset(dh_next, 0, sizeof(dh_next));
        for (j = 0; j < N_HID; j++)
            for (i = 0; i < N_HID; i++)
                dh_next[j] += dz[i] * n->W_h[i][j];
    }
}

static void update(RNN *n, const RNN *g, float lr) {
    float *w = (float*)n;
    const float *gv = (const float*)g;
    int sz = sizeof(RNN)/sizeof(float);
    int i;
    for (i = 0; i < sz; i++) w[i] -= lr * gv[i];
    }

int main(void)
{
    RNN net, grad;
    int i, j;
    srand(42);
    for (i = 0; i < N_HID; i++) {
        net.W_x[i] = randf()*0.4f-0.2f;
        net.b_h[i] = 0;
        net.W_y[i] = randf()*0.4f-0.2f;
        for (j = 0; j < N_HID; j++)
            net.W_h[i][j] = randf()*0.4f-0.2f;
    }
    net.b_y = 0;

    /* Train on length-6 sequences */
    float X[8][6] = {{0, 0, 0, 0, 0, 0}, {1, 0, 
        0, 0, 0, 0}, 
                     {1, 1, 0, 0, 0, 0}, {1, 0, 
                         1, 0, 0, 0}, 
                     {1, 1, 1, 0, 0, 0}, {1, 0, 
                         1, 0, 1, 0}, 
                     {1, 1, 1, 1, 0, 0}, {1, 1, 
                         1, 1, 1, 1}};
    float T[8][6];
    int s, t;
    for (s = 0; s < 8; s++) {
        float c = 0;
        for (t = 0; t < 6; t++) {
            c += X[s][t];
            T[s][t] = c;
        }
    }

    for (int epoch = 0; epoch < 3000; epoch++) {
        for (s = 0; s < 8; s++) {
            float h[MAX_LEN+1][N_HID], y[MAX_LEN];
            rnn_forward(&net, X[s], 6, h, y);
            rnn_backward(&net, X[s], T[s], 6, 
                h, y, &grad);
            update(&net, &grad, 0.005f);
        }
    }

    /* Test on DIFFERENT lengths */
    printf("Trained on length 6, testing on "
           "other lengths:\n\n");

    struct { float seq[20]; int len; } tests[] = {
        {{ 1, 0, 1 }, 3}, 
        {{ 1, 1, 0, 1 }, 4}, 
        {{ 1, 0, 1, 1, 0, 1, 0, 1 }, 8}, 
        {{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 10}, 
    };
    int n_tests = 4;

    for (int test = 0; test < n_tests; test++) {
        float h[MAX_LEN+1][N_HID], y[MAX_LEN];
        int len = tests[test].len;
        rnn_forward(&net, tests[test].seq, len, h, y);

        float count = 0;
        printf("  Length %2d: ", len);
        for (t = 0; t < len; t++)
            count += tests[test].seq[t];
        printf("predicted=%.1f  actual=%.0f  ",
               y[len-1], count);
        printf("error=%.2f\n", fabsf(y[len-1] - count));
    }

    printf("\nThe same weights work for any "
           "sequence length.\n");
    printf("Accuracy degrades for lengths far "
           "from training (6).\n");

    return 0;
}
