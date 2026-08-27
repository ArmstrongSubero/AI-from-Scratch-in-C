/* 077_Gradient_Decay.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_HID 4
#define MAX_LEN 60

static float my_tanh(float z)
{
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
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

static void forward(const RNN *n, 
    const float *x, int len, 
              float h[MAX_LEN + 1][N_HID],
              float *y)
{
    int t, i, j;
    for (i = 0; i < N_HID; i++) h[0][i] = 0;
    for (t = 0; t < len; t++) {
        for (i = 0; i < N_HID; i++) {
            float z = n->b_h[i] + n->W_x[i] * x[t];
            for (j = 0; j < N_HID; j++)
                z += n->W_h[i][j] * h[t][j];
            h[t + 1][i] = my_tanh(z);
        }
        y[t] = n->b_y;
        for (j = 0; j < N_HID; j++)
            y[t] += n->W_y[j] * h[t + 1][j];
    }
}

/* Backward pass, returning the gradient norm
   at each time step */
static void backward_with_norms(
        const RNN *n, 
        const float *x, 
        const float *tgt, 
        int len, 
        const float h[MAX_LEN + 1][N_HID], 
                                 const float *y, 
        float *grad_norms)
{
    int t, i, j;
    float dh_next[N_HID];
    memset(dh_next, 0, sizeof(dh_next));

    for (t = len - 1; t >= 0; t--) {
        float dy = 2.0f * (y[t] - tgt[t]);
        float dh[N_HID], dz[N_HID];

        for (i = 0; i < N_HID; i++)
            dh[i] = dy * n->W_y[i] + dh_next[i];

        for (i = 0; i < N_HID; i++) {
            float hv = h[t + 1][i];
            dz[i] = dh[i] * (1.0f - hv * hv);
        }

        /* Compute gradient norm at this step */
        float norm = 0.0f;
        for (i = 0; i < N_HID; i++)
            norm += dz[i] * dz[i];
        grad_norms[t] = sqrtf(norm);

        /* Propagate to previous step */
        memset(dh_next, 0, sizeof(dh_next));
        for (j = 0; j < N_HID; j++)
            for (i = 0; i < N_HID; i++)
                dh_next[j] += dz[i] * n->W_h[i][j];
    }
}

int main(void)
{
    RNN net;
    int i, j;

    srand(42);
    for (i = 0; i < N_HID; i++) {
        net.W_x[i] = randf() * 0.4f - 0.2f;
        net.b_h[i] = 0;
        net.W_y[i] = randf() * 0.4f - 0.2f;
        for (j = 0; j < N_HID; j++)
            net.W_h[i][j] = randf() * 0.4f - 0.2f;
    }
    net.b_y = 0;

    /* Test gradient decay across several
       sequence lengths */
    int lengths[] = { 5, 10, 20, 40 };
    int n_lengths = 4;
    int li;

    printf("Gradient norm at each time step "
           "(backward from end):\n\n");

    for (li = 0; li < n_lengths; li++) {
        int len = lengths[li];
        float x[MAX_LEN], tgt[MAX_LEN];
        float h[MAX_LEN + 1][N_HID], y[MAX_LEN];
        float grad_norms[MAX_LEN];
        float count;
        int t;

        /* Generate a sequence of random bits */
        srand(100);
        count = 0;
        for (t = 0; t < len; t++) {
            x[t] = (randf() > 0.5f) ? 1.0f : 0.0f;
            count += x[t];
            tgt[t] = count;
        }

        forward(&net, x, len, h, y);
        backward_with_norms(&net, x, tgt, len, h, y, 
                            grad_norms);

        printf("  Length %d:\n", len);
        printf("    Last step (t=%d):  grad_norm = "
               "%.6f\n",
               len - 1, grad_norms[len - 1]);
        printf("    Mid  step (t=%d):  grad_norm = "
               "%.6f\n",
               len / 2, grad_norms[len / 2]);
        printf("    First step (t=0): grad_norm = "
               "%.6f\n",
               grad_norms[0]);
        if (grad_norms[len - 1] > 0)
            printf("    Ratio first/last: %.8f\n",
                   grad_norms[0] / grad_norms[len - 1]);
        printf("\n");
    }

    return 0;
}
