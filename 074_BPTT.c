/* 074_BPTT.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_IN 1
#define N_HID 4
#define N_OUT 1
#define MAX_LEN 8

static float my_tanh(float z)
{
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}

typedef struct {
    float W_x[N_HID][N_IN];
    float W_h[N_HID][N_HID];
    float b_h[N_HID];
    float W_y[N_OUT][N_HID];
    float b_y[N_OUT];
}
RNN;

/* Forward pass: store all hidden states for BPTT */
static void forward(const RNN *net, 
    const float *x, int len, 
                    float h_all[MAX_LEN + 1][N_HID], 
                    float y_all[MAX_LEN])
{
    int t, i, j;

    /* h_all[0] = initial hidden state (zeros) */
    for (i = 0; i < N_HID; i++)
        h_all[0][i] = 0.0f;

    for (t = 0; t < len; t++) {
        /* Hidden state update */
        for (i = 0; i < N_HID; i++) {
            float z = net->b_h[i] + net
                ->W_x[i][0] * x[t];
            for (j = 0; j < N_HID; j++)
                z += net->W_h[i][j] * h_all[t][j];
            h_all[t + 1][i] = my_tanh(z);
        }

        /* Output */
        y_all[t] = net->b_y[0];
        for (j = 0; j < N_HID; j++)
            y_all[t] += net->W_y[0][j]
                * h_all[t + 1][j];
    }
}

/* Backward pass through time */
static void backward(const RNN *net, 
                      const float *x, 
                      const float *targets, 
                      int len, 
              const float h_all[MAX_LEN + 1][N_HID],
                      const float y_all[MAX_LEN], 
                      RNN *grad)
{
    int t, i, j;
    /* gradient flowing back from the next step */
    float dh_next[N_HID];

    /* Zero gradients */
    memset(grad, 0, sizeof(RNN));
    memset(dh_next, 0, sizeof(dh_next));

    /* Walk backward through time */
    for (t = len - 1; t >= 0; t--) {
        /* Output gradient for MSE,
           d_loss/dy = 2*(y - target) */
        float dy = 2.0f * (y_all[t] - targets[t]);

        /* W_y gradient */
        for (j = 0; j < N_HID; j++)
            grad->W_y[0][j] += dy * h_all[t + 1][j];
        grad->b_y[0] += dy;

        /* Gradient into the hidden state from the
           output AND from the next step */
        float dh[N_HID];
        for (i = 0; i < N_HID; i++) {
            dh[i] = dy * net->W_y[0][i] + dh_next[i];
        }

        /* Through tanh: d_tanh = 1 - tanh^2 */
        float dz[N_HID];
        for (i = 0; i < N_HID; i++) {
            float h = h_all[t + 1][i];
            dz[i] = dh[i] * (1.0f - h * h);
        }

        /* W_x gradient */
        for (i = 0; i < N_HID; i++)
            grad->W_x[i][0] += dz[i] * x[t];

        /* W_h gradient */
        for (i = 0; i < N_HID; i++)
            for (j = 0; j < N_HID; j++)
                grad->W_h[i][j] += dz[i] * h_all[t][j];

        /* b_h gradient */
        for (i = 0; i < N_HID; i++)
            grad->b_h[i] += dz[i];

        /* Propagate gradient to previous hidden
           state */
        memset(dh_next, 0, sizeof(dh_next));
        for (j = 0; j < N_HID; j++)
            for (i = 0; i < N_HID; i++)
                dh_next[j] += dz[i] * net->W_h[i][j];
    }
}

/* Simple SGD update */
static void update(RNN *net, const RNN *grad, float lr)
{
    float *w = (float *)net;
    const float *g = (const float *)grad;
    int n = sizeof(RNN) / sizeof(float);
    int i;
    for (i = 0; i < n; i++)
        w[i] -= lr * g[i];
}

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

int main(void)
{
    RNN net, grad;
    int i, j;

    srand(42);

    /* Initialize weights */
    for (i = 0; i < N_HID; i++) {
        net.W_x[i][0] = randf() * 0.4f - 0.2f;
        net.b_h[i] = 0.0f;
        net.W_y[0][i] = randf() * 0.4f - 0.2f;
        for (j = 0; j < N_HID; j++)
            net.W_h[i][j] = randf() * 0.4f - 0.2f;
    }
    net.b_y[0] = 0.0f;

    /* Training data: count 1s in 6-bit sequences */
    float X[8][6] = {
        {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, 
        {1, 1, 0, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, 
        {1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 1, 0}, 
        {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1}, 
    };
    /* Target: running count at each step */
    float T[8][6];
    int s, t;
    for (s = 0; s < 8; s++) {
        float count = 0;
        for (t = 0; t < 6; t++) {
            count += X[s][t];
            T[s][t] = count;
        }
    }

    float lr = 0.01f;

    printf("Training RNN to count 1s "
           "(sequence-to-sequence):\n\n");

    for (int epoch = 0; epoch < 2000; epoch++) {
        float total_loss = 0.0f;

        for (s = 0; s < 8; s++) {
            float h_all[MAX_LEN + 1][N_HID];
            float y_all[MAX_LEN];

            forward(&net, X[s], 6, h_all, y_all);
            backward(&net, X[s], T[s], 6, 
                     h_all, y_all, &grad);
            update(&net, &grad, lr);

            for (t = 0; t < 6; t++) {
                float diff = y_all[t] - T[s][t];
                total_loss += diff * diff;
            }
        }

        if ((epoch + 1) % 500 == 0)
            printf("  epoch %4d  loss=%.4f\n",
                   epoch + 1, total_loss / (8 * 6));
    }

    /* Test on a new sequence */
    printf("\nTest on new sequence "
           "[1, 0, 1, 1, 0, 1]:\n\n");
    float test[] = { 1, 0, 1, 1, 0, 1 };
    float h_all[MAX_LEN + 1][N_HID];
    float y_all[MAX_LEN];
    forward(&net, test, 6, h_all, y_all);

    float count = 0;
    printf("  t  input  predicted  actual\n");
    for (t = 0; t < 6; t++) {
        count += test[t];
        printf("  %d    %.0f     %5.2f     %.0f\n",
               t, test[t], y_all[t], count);
    }

    return 0;
}
