/* 018_XOR_Comparisons.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- Activation functions and their derivatives --- */

static float act_sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float der_sigmoid(float z, float out) {
    (void)z;
    return out * (1.0f - out);
}

static float act_tanh(float z) {
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}
static float der_tanh(float z, float out) {
    (void)z;
    return 1.0f - out * out;
}

static float act_relu(float z)
{
    return z > 0.0f ? z : 0.0f;
}
static float der_relu(float z, float out) {
    (void)out;
    return z > 0.0f ? 1.0f : 0.0f;
}

/* --- Network --- */

typedef struct {
    float wh[2][2], bh[2];
    float wo[2], bo;
} Net;

typedef float (*act_fn)(float);
typedef float (*der_fn)(float, float);

static float forward(const Net *n, const float x[2],
    float h[2],
                     float zh[2], float *zo, act_fn act)
{
    int i, j;
    for (i = 0; i < 2; i++) {
        zh[i] = n->bh[i];
        for (j = 0; j < 2; j++)
            zh[i] += n->wh[i][j] * x[j];
        h[i] = act(zh[i]);
    }
    *zo = n->bo;
    for (i = 0; i < 2; i++)
        *zo += n->wo[i] * h[i];
    return act(*zo);
}

static void backward(Net *n, const float x[2],
    const float h[2],
                     const float zh[2], float zo, 
                     float y, float t, float lr, 
                     der_fn der)
{
    int i, j;
    float delta_out = -2.0f * (t - y) * der(zo, y);
    float delta_h[2];

    for (i = 0; i < 2; i++)
        delta_h[i] = delta_out * n->wo[i] * der(zh[i],
            h[i]);

    for (i = 0; i < 2; i++)
        n->wo[i] -= lr * delta_out * h[i];
    n->bo -= lr * delta_out;

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++)
            n->wh[i][j] -= lr * delta_h[i] * x[j];
        n->bh[i] -= lr * delta_h[i];
    }
}

static void init_random(Net *n)
{
    int i, j;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++)
            n->wh[i][j] = ((float)rand() / RAND_MAX)
                * 2.0f - 1.0f;
        n->bh[i] = ((float)rand() / RAND_MAX) * 2.0f
            - 1.0f;
        n->wo[i] = ((float)rand() / RAND_MAX) * 2.0f
            - 1.0f;
    }
    n->bo = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static void train_xor(const char *name, act_fn act,
    der_fn der)
{
    float X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    float T[4] = { 0, 1, 1, 0 };
    Net net;
    float lr = 1.0f;
    int epoch, s;

    srand(42);
    init_random(&net);

    for (epoch = 0; epoch < 10000; epoch++) {
        float total_loss = 0.0f;
        for (s = 0; s < 4; s++) {
            float h[2], zh[2], zo, y, diff;
            y = forward(&net, X[s], h, zh, &zo, act);
            diff = T[s] - y;
            total_loss += diff * diff;
            backward(&net, X[s], h, zh, zo, y, T[s], lr,
                der);
        }
        if (epoch == 9999)
            printf("%-8s  final loss=%.6f  ", name,
                total_loss / 4.0f);
    }

    /* Final check */
    for (s = 0; s < 4; s++) {
        float h[2], zh[2], zo;
        float y = forward(&net, X[s], h, zh, &zo, act);
        printf("%.2f ", y);
    }
    printf("\n");
}

int main(void)
{
    printf("Activation  Final Loss"
           "    Outputs (00 01 10 11)\n");
    printf("-------------------------"
           "--------------------------\n");
    train_xor("sigmoid", act_sigmoid, der_sigmoid);
    train_xor("tanh",    act_tanh,    der_tanh);
    train_xor("relu",    act_relu,    der_relu);
    return 0;
}
