/* 080_Gradient_Flow.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_HID 4
#define SEQ_LEN 30

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

int main(void)
{
    RNN net;
    int i, j, t;

    srand(42);
    for (i = 0; i < N_HID; i++) {
        net.W_x[i] = randf() * 0.4f - 0.2f;
        net.b_h[i] = 0;
        net.W_y[i] = randf() * 0.4f - 0.2f;
        for (j = 0; j < N_HID; j++)
            net.W_h[i][j] = randf() * 0.4f - 0.2f;
    }
    net.b_y = 0;

    /* Generate sequence */
    float x[SEQ_LEN], tgt[SEQ_LEN];
    float count = 0;
    srand(100);
    for (t = 0; t < SEQ_LEN; t++) {
        x[t] = (randf() > 0.5f) ? 1.0f : 0.0f;
        count += x[t];
        tgt[t] = count;
    }

    /* Forward pass */
    float h[SEQ_LEN + 1][N_HID], y[SEQ_LEN];
    for (i = 0; i < N_HID; i++) h[0][i] = 0;
    for (t = 0; t < SEQ_LEN; t++) {
        for (i = 0; i < N_HID; i++) {
            float z = net.b_h[i] + net.W_x[i] * x[t];
            for (j = 0; j < N_HID; j++)
                z += net.W_h[i][j] * h[t][j];
            h[t+1][i] = my_tanh(z);
        }
        y[t] = net.b_y;
        for (j = 0; j < N_HID; j++)
            y[t] += net.W_y[j] * h[t+1][j];
    }

    /* Backward pass with gradient norm at each step */
    float dh_next[N_HID];
    memset(dh_next, 0, sizeof(dh_next));

    float norms[SEQ_LEN];

    for (t = SEQ_LEN - 1; t >= 0; t--) {
        float dy = 2.0f * (y[t] - tgt[t]);
        float dh[N_HID], dz[N_HID];

        for (i = 0; i < N_HID; i++)
            dh[i] = dy * net.W_y[i] + dh_next[i];
        for (i = 0; i < N_HID; i++) {
            float hv = h[t+1][i];
            dz[i] = dh[i] * (1 - hv * hv);
        }

        float norm = 0;
        for (i = 0; i < N_HID; i++)
            norm += dz[i] * dz[i];
        norms[t] = sqrtf(norm);

        memset(dh_next, 0, sizeof(dh_next));
        for (j = 0; j < N_HID; j++)
            for (i = 0; i < N_HID; i++)
                dh_next[j] += dz[i] * net.W_h[i][j];
    }

    /* Print as a bar chart */
    printf("Gradient norm at each time step "
           "(length %d):\n\n", SEQ_LEN);
    printf("  t  norm        bar\n");

    float max_norm = 0;
    for (t = 0; t < SEQ_LEN; t++)
        if (norms[t] > max_norm) max_norm = norms[t];

    for (t = SEQ_LEN - 1; t >= 0; t--) {
        int bar_len = (max_norm > 0)
            ? (int)(norms[t] / max_norm * 40) : 0;
        printf("  %2d  %.6f  ", t, norms[t]);
        for (i = 0; i < bar_len; i++) printf("#");
        printf("\n");
    }

    printf("\n  The gradient is strong at the end "
           "and vanishes\n");
    printf("  toward the beginning. Early inputs "
           "get almost\n");
    printf("  no learning signal.\n");

    return 0;
}
