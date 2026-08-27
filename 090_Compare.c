/* 090_Compare.c */
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

/* --- Basic RNN --- */
static void rnn_run(const float *x, int len, 
    float *h_final)
{
    float W_x[N], W_h[N][N], b[N], h[N] = {0}, hn[N];
    int t, i, j;
    for (i = 0; i < N; i++) {
        W_x[i] = randf()*0.4f-0.2f;
        b[i] = 0;
        for (j = 0; j < N; j++)
            W_h[i][j] = randf()*0.4f-0.2f;
    }
    for (t = 0; t < len; t++) {
        for (i = 0; i < N; i++) {
            float z = b[i] + W_x[i] * x[t];
            for (j = 0; j < N; j++)
                z += W_h[i][j] * h[j];
            hn[i] = my_tanh(z);
        }
        memcpy(h, hn, sizeof(h));
        }
    memcpy(h_final, h, sizeof(h));
}

/* --- LSTM --- */
static void lstm_run(const float *x, int len, 
                     float *h_final)
{
    float W[4][N], U[4][N][N], b[4][N];
    float h[N] = {0}, c[N] = {0}, hn[N], cn[N];
    int t, i, j, g;
    for (g = 0; g < 4; g++) for (i = 0; i < N; i++) {
        W[g][i] = randf()*0.2f-0.1f;
        /* gate 0 is forget, and it is the only
           one that starts open */
        b[g][i] = (g == 0) ? 1.0f : 0.0f;
        for (j = 0; j < N; j++)
            U[g][i][j] = randf()*0.2f-0.1f;
    }
    for (t = 0; t < len; t++) {
        for (i = 0; i < N; i++) {
            float zz[4];
            for (g = 0; g < 4; g++) {
                zz[g] = b[g][i] + W[g][i] * x[t];
                for (j = 0; j < N; j++)
                    zz[g] += U[g][i][j] * h[j];
            }
            float fi = sigmoid(zz[0]);
            float ii = sigmoid(zz[1]);
            float oi = sigmoid(zz[2]);
            float gi = my_tanh(zz[3]);
            cn[i] = fi * c[i] + ii * gi;
            hn[i] = oi * my_tanh(cn[i]);
        }
        memcpy(h, hn, sizeof(h));
        memcpy(c, cn, sizeof(c));
    }
    memcpy(h_final, h, sizeof(h));
}

/* --- GRU --- */
static void gru_run(const float *x, int len, 
    float *h_final)
{
    float W_z[N], U_z[N][N], b_z[N];
    float W_r[N], U_r[N][N], b_r[N];
    float W_s[N], U_s[N][N], b_s[N], h[N] = {0}, hn[N];
    int t, i, j;
    for (i = 0; i < N; i++) {
        W_z[i] = randf()*0.2f-0.1f;
        b_z[i] = 0;
        W_r[i] = randf()*0.2f-0.1f;
        b_r[i] = 0;
        W_s[i] = randf()*0.2f-0.1f;
        b_s[i] = 0;
        for (j = 0; j < N; j++) {
            U_z[i][j] = randf()*0.2f-0.1f;
            U_r[i][j] = randf()*0.2f-0.1f;
            U_s[i][j] = randf()*0.2f-0.1f;
        }
    }
    for (t = 0; t < len; t++) {
        for (i = 0; i < N; i++) {
            float zz = b_z[i]+W_z[i]*x[t];
            float zr = b_r[i]+W_r[i]*x[t];
            for (j = 0; j < N; j++) {
                zz += U_z[i][j]*h[j];
                zr += U_r[i][j]*h[j];
            }
            float z = sigmoid(zz), r = sigmoid(zr);
            float zs = b_s[i] + W_s[i]*x[t];
            for (j = 0; j < N; j++)
                zs += U_s[i][j]*(r*h[j]);
            float s = my_tanh(zs);
            hn[i] = z * h[i] + (1-z) * s;
            }
        memcpy(h, hn, sizeof(h));
        }
    memcpy(h_final, h, sizeof(h));
}

int main(void)
{
    int lengths[] = { 5, 10, 20, 50 };
    int n_lengths = 4;
    int li, trial, i;

    printf("Information preservation, input 1 at "
           "t=0 then zeros.\n");
    printf("Measure norm of final hidden state.\n\n");
    printf("  length   RNN         GRU         "
           "LSTM\n");
    printf("  ------   ---------   ---------   "
           "---------\n");

    for (li = 0; li < n_lengths; li++) {
        int len = lengths[li];
        float rnn_sum = 0, gru_sum = 0, lstm_sum = 0;
        int n_trials = 30;

        for (trial = 0; trial < n_trials; trial++) {
            srand(42 + trial);

            float x[100] = {0};
            x[0] = 1.0f;
            float h[N];
            float norm;

            srand(42 + trial);
            rnn_run(x, len, h);
            norm = 0;
            for (i = 0; i < N; i++)
                norm += h[i]*h[i];
            rnn_sum += sqrtf(norm);

            srand(42 + trial);
            gru_run(x, len, h);
            norm = 0;
            for (i = 0; i < N; i++)
                norm += h[i]*h[i];
            gru_sum += sqrtf(norm);

            srand(42 + trial);
            lstm_run(x, len, h);
            norm = 0;
            for (i = 0; i < N; i++)
                norm += h[i]*h[i];
            lstm_sum += sqrtf(norm);
        }

        printf("  %3d      %.3e   %.3e   %.3e\n",
               len, rnn_sum/n_trials, 
               gru_sum/n_trials, lstm_sum/n_trials);
    }

    printf("\nRNN decays fastest, having no gates.\n");
    printf("GRU retains more, since the update "
           "gate allows pass-through.\n");
    printf("LSTM retains most, with a dedicated "
           "cell state and forget bias 1.\n");

    return 0;
}
