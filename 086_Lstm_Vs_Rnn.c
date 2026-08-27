/* 086_Lstm_Vs_Rnn.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

#define N 8
#define MAX_LEN 50

/* --- Basic RNN --- */
typedef struct {
    float W_x[N], W_h[N][N], b_h[N], W_y[N], b_y;
}
BasicRNN;

static float rnn_run(const BasicRNN *r, 
                     const float *x, int len)
{
    float h[N] = {0}, hn[N];
    int t, i, j;
    for (t = 0; t < len; t++) {
        for (i = 0; i < N; i++) {
            float z = r->b_h[i] + r->W_x[i] * x[t];
            for (j = 0; j < N; j++)
                z += r->W_h[i][j] * h[j];
            hn[i] = my_tanh(z);
        }
        memcpy(h, hn, sizeof(h));
    }
    float y = r->b_y;
    for (i = 0; i < N; i++) y += r->W_y[i] * h[i];
    return y;
}

/* --- LSTM --- */
typedef struct {
    float W_f[N], U_f[N][N], b_f[N];
    float W_i[N], U_i[N][N], b_i[N];
    float W_o[N], U_o[N][N], b_o[N];
    float W_g[N], U_g[N][N], b_g[N];
    float W_y[N], b_y;
}
LSTMNet;

static float lstm_run(const LSTMNet *l, 
                      const float *x, int len)
{
    float h[N] = {0}, c[N] = {0}, hn[N], cn[N];
    int t, i, j;
    for (t = 0; t < len; t++) {
        for (i = 0; i < N; i++) {
            float zf = l->b_f[i] + l->W_f[i] * x[t];
            float zi = l->b_i[i] + l->W_i[i] * x[t];
            float zo = l->b_o[i] + l->W_o[i] * x[t];
            float zg = l->b_g[i] + l->W_g[i] * x[t];
            for (j = 0; j < N; j++) {
                zf += l->U_f[i][j] * h[j];
                zi += l->U_i[i][j] * h[j];
                zo += l->U_o[i][j] * h[j];
                zg += l->U_g[i][j] * h[j];
            }
            float f = sigmoid(zf), ig = sigmoid(zi);
            float o = sigmoid(zo), g = my_tanh(zg);
            cn[i] = f * c[i] + ig * g;
            hn[i] = o * my_tanh(cn[i]);
        }
        memcpy(h, hn, sizeof(h));
        memcpy(c, cn, sizeof(c));
    }
    float y = l->b_y;
    for (i = 0; i < N; i++) y += l->W_y[i] * h[i];
    return y;
}

int main(void)
{
    /* Check how well each architecture preserves
       the first input */
    printf("Information preservation test:\n");
    printf("  Input: 1 at t=0, then zeros. Does the "
           "final hidden\n");
    printf("  state still carry information about "
           "the first input?\n\n");

    int lengths[] = { 5, 10, 20, 30, 50 };
    int n_lengths = 5;
    int li, trial;

    printf("  length   RNN h-norm   LSTM h-norm   "
           "ratio\n");
    printf("  ------   ----------   -----------   "
           "-----\n");

    for (li = 0; li < n_lengths; li++) {
        int len = lengths[li];
        float rnn_norm_sum = 0, lstm_norm_sum = 0;
        int n_trials = 20;

        for (trial = 0; trial < n_trials; trial++) {
            srand(42 + trial);

            /* Initialize RNN */
            BasicRNN r;
            int i, j;
            for (i = 0; i < N; i++) {
                r.W_x[i] = randf()*0.4f-0.2f;
                r.b_h[i] = 0;
                r.W_y[i] = randf()*0.4f-0.2f;
                for (j = 0; j < N; j++)
                    r.W_h[i][j] = randf()*0.4f-0.2f;
            }
            r.b_y = 0;

            /* Initialize LSTM */
            LSTMNet l;
            for (i = 0; i < N; i++) {
                l.W_f[i] = randf()*0.2f-0.1f;
                l.b_f[i] = 1.0f;
                l.W_i[i] = randf()*0.2f-0.1f;
                l.b_i[i] = 0;
                l.W_o[i] = randf()*0.2f-0.1f;
                l.b_o[i] = 0;
                l.W_g[i] = randf()*0.2f-0.1f;
                l.b_g[i] = 0;
                l.W_y[i] = randf()*0.4f-0.2f;
                for (j = 0; j < N; j++) {
                    l.U_f[i][j] = randf()*0.2f-0.1f;
                    l.U_i[i][j] = randf()*0.2f-0.1f;
                    l.U_o[i][j] = randf()*0.2f-0.1f;
                    l.U_g[i][j] = randf()*0.2f-0.1f;
                }
            }
            l.b_y = 0;

            /* Sequence: 1 followed by zeros */
            float x[MAX_LEN] = {0};
            x[0] = 1.0f;

            float ry = rnn_run(&r, x, len);
            float ly = lstm_run(&l, x, len);

            rnn_norm_sum += fabsf(ry);
            lstm_norm_sum += fabsf(ly);
        }

        float rnn_avg = rnn_norm_sum / n_trials;
        float lstm_avg = lstm_norm_sum / n_trials;
        printf("  %3d      %.3e    %.3e   ", len,
               rnn_avg, lstm_avg);
        if (rnn_avg > 0)
            printf("%.0fx\n", lstm_avg / rnn_avg);
        else
            printf("RNN is zero\n");
    }

    printf("\nThe LSTM output norm stays much higher "
           "across long\n");
    printf("sequences. It preserves information "
           "from t=0 because\n");
    printf("the forget gate (bias=1) slows the "
           "decay dramatically.\n");

    return 0;
}
