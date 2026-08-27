/* 011_Backpropagation.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

typedef struct {
    float wh[2][2];   /* hidden weights [neuron][input] */
    float bh[2];      /* hidden biases */
    float wo[2];      /* output weights [hidden_neuron] */
    float bo;         /* output bias */
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

    /* Output delta: d_loss/dy * dy/dz_out */
    delta_out = -2.0f * (t - y) * y * (1.0f - y);

    /* Hidden deltas: error flows back through wo */
    for (i = 0; i < 2; i++)
        delta_h[i] = delta_out * n->wo[i]
                     * h[i] * (1.0f - h[i]);

    /* Update output weights and bias */
    for (i = 0; i < 2; i++)
        n->wo[i] -= lr * delta_out * h[i];
    n->bo -= lr * delta_out;

    /* Update hidden weights and biases */
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
        float total_loss = 0.0f;

        for (s = 0; s < 4; s++) {
            float h[2], y, diff;

            y = forward(&net, X[s], h);
            diff = T[s] - y;
            total_loss += diff * diff;
            backward(&net, X[s], h, y, T[s], lr);
        }

        if ((epoch + 1) % 2000 == 0)
            printf("epoch %5d  loss=%.6f\n",
                   epoch + 1,
                   total_loss / 4.0f);
    }

    /* Final results */
    printf("\nFinal results:\n");
    for (s = 0; s < 4; s++) {
        float h[2];
        float y = forward(&net, X[s], h);
        printf("  %.0f XOR %.0f = %.4f  (target %.0f)\n",
               X[s][0], X[s][1], y, T[s]);
    }

    return 0;
}
