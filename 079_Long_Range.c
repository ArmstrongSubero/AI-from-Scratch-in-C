/* 079_Long_Range.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_HID 8
#define MAX_LEN 60

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

static float run_seq(const RNN *n, 
    const float *x, int len)
{
    float h[N_HID] = {0};
    float h_new[N_HID];
    int t, i, j;

    for (t = 0; t < len; t++) {
        for (i = 0; i < N_HID; i++) {
            float z = n->b_h[i] + n->W_x[i] * x[t];
            for (j = 0; j < N_HID; j++)
                z += n->W_h[i][j] * h[j];
            h_new[i] = my_tanh(z);
        }
        for (i = 0; i < N_HID; i++) h[i] = h_new[i];
    }
    float y = n->b_y;
    for (i = 0; i < N_HID; i++) y += n->W_y[i] * h[i];
    return y;
}

static void train_step(RNN *n, const float *x, 
                       int len, float target, float lr)
{
    /* Full BPTT, simplified to use numerical
       gradients for clarity */
    float *w = (float *)n;
    int nw = sizeof(RNN) / sizeof(float);
    int i;
    float h = 0.001f;

    float y0 = run_seq(n, x, len);
    float loss0 = (y0 - target) * (y0 - target);

    for (i = 0; i < nw; i++) {
        float orig = w[i];
        w[i] = orig + h;
        float y1 = run_seq(n, x, len);
        float loss1 = (y1 - target) * (y1 - target);
        float grad = (loss1 - loss0) / h;
        w[i] = orig - lr * grad;
    }
}

int main(void)
{
    RNN net;
    int i, j;

    srand(42);
    for (i = 0; i < N_HID; i++) {
        net.W_x[i] = randf() * 0.2f - 0.1f;
        net.b_h[i] = 0;
        net.W_y[i] = randf() * 0.2f - 0.1f;
        for (j = 0; j < N_HID; j++)
            net.W_h[i][j] = randf() * 0.2f - 0.1f;
    }
    net.b_y = 0;

    printf("Long-range dependency test:\n");
    printf("  First bit determines the sign of "
           "the output.\n");
    printf("  The network must remember the first "
           "bit through\n");
    printf("  the entire sequence.\n\n");

    int test_lens[] = { 5, 10, 20 };
    int n_tests = 3;
    int li;

    for (li = 0; li < n_tests; li++) {
        int len = test_lens[li];

        /* Reset network */
        srand(42);
        for (i = 0; i < N_HID; i++) {
            net.W_x[i] = randf() * 0.2f - 0.1f;
            net.b_h[i] = 0;
            net.W_y[i] = randf() * 0.2f - 0.1f;
            for (j = 0; j < N_HID; j++)
                net.W_h[i][j] = randf() * 0.2f - 0.1f;
        }
        net.b_y = 0;

        /* Train */
        int epoch;
        for (epoch = 0; epoch < 200; epoch++) {
            int s;
            for (s = 0; s < 20; s++) {
                float x[MAX_LEN];
                int t;
                float count = 0;

                /* Generate random sequence */
                for (t = 0; t < len; t++) {
                    x[t] = (randf() > 0.5f)
                        ? 1.0f
                        : 0.0f;
                    count += x[t];
                }

                /* Positive count when the first bit
                   is 1, negative otherwise */
                float target = 
                    (x[0] > 0.5f) ? count : -count;
                train_step(&net, x, len, 
                    target, 0.001f);
            }
        }

        /* Test: does it get the sign right? */
        int correct = 0, total = 50;
        for (i = 0; i < total; i++) {
            float x[MAX_LEN];
            int t;
            float count = 0;
            for (t = 0; t < len; t++) {
                x[t] = (randf() > 0.5f) ? 1.0f : 0.0f;
                count += x[t];
            }
            float target = (x[0] > 0.5f)
                ? count
                : -count;
            float y = run_seq(&net, x, len);

            /* Check if sign matches */
            if ((y > 0 && target > 0) ||
                (y < 0 && target < 0) ||
                (y == 0 && target == 0))
                correct++;
        }

        printf("  Length %2d: sign accuracy = "
               "%d/%d (%.0f%%)\n",
               len, correct, total, 
               100.0f * correct / total);
    }

    printf("\n  At short lengths, the RNN can "
           "remember the first bit.\n");
    printf("  As length increases, accuracy drops "
           "toward 50%% (random).\n");
    printf("  The gradient from the first step "
           "has vanished.\n");

    return 0;
}
