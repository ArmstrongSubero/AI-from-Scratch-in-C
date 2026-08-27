/* 012_Watching_It_Learn.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

typedef struct {
    float wh[2][2];
    float bh[2];
    float wo[2];
    float bo;
} Net;

static float forward(const Net *n,
                     const float x[2], float h[2])
{
    int i, j;
    for (i = 0; i < 2; i++) {
        float z = n->bh[i];
        for (j = 0; j < 2; j++)
            z += n->wh[i][j] * x[j];
        h[i] = sigmoid(z);
    }
    {
        float z = n->bo;
        for (i = 0; i < 2; i++)
            z += n->wo[i] * h[i];
        return sigmoid(z);
    }
}

static void backward(Net *n, const float x[2],
                      const float h[2],
                      float y, float t, float lr)
{
    int i, j;
    float delta_out, delta_h[2];

    delta_out = -2.0f * (t - y) * y * (1.0f - y);

    for (i = 0; i < 2; i++)
        delta_h[i] = delta_out * n->wo[i]
                     * h[i] * (1.0f - h[i]);

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
            n->wh[i][j] =
                ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        n->bh[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        n->wo[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    n->bo = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(void)
{
    float X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
    float T[4]    = {  0,     1,     1,     0    };
    Net net;
    float lr = 1.0f;
    int epoch, s;

    srand(42);
    init_random(&net);

    for (epoch = 0; epoch < 10000; epoch++) {
        for (s = 0; s < 4; s++) {
            float h[2], y;
            y = forward(&net, X[s], h);
            backward(&net, X[s], h, y, T[s], lr);
        }
    }

    /* Inspect what the hidden layer learned */
    printf("Hidden layer representations:\n");
    printf("  input     h0      h1      output  target\n");
    for (s = 0; s < 4; s++) {
        float h[2];
        float y = forward(&net, X[s], h);
        printf("  (%.0f,%.0f)  %6.4f %6.4f %6.4f %.0f\n",
               X[s][0], X[s][1], h[0], h[1], y, T[s]);
    }

    printf("\nLearned weights:\n");
    printf("  hidden neuron 0: w=(%.3f, %.3f) b=%.3f\n",
           net.wh[0][0], net.wh[0][1], net.bh[0]);
    printf("  hidden neuron 1: w=(%.3f, %.3f) b=%.3f\n",
           net.wh[1][0], net.wh[1][1], net.bh[1]);
    printf("  output neuron:   w=(%.3f, %.3f) b=%.3f\n",
           net.wo[0], net.wo[1], net.bo);

    return 0;
}
