/* 039_XOR_Hidden.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

typedef struct { float wh[2][2], 
    bh[2];
    float wo[2], bo;
    }
    Net;

static float forward(const Net *n, 
    const float x[2], float h[2])
{
    int i, j;
    for (i = 0; i < 2; i++) {
        float z = n->bh[i];
        for (j = 0; j < 2; j++) z += n->wh[i][j] * x[j];
        h[i] = sigmoid(z);
    }
    { float z = n->bo;
      for (i = 0; i < 2; i++) z += n->wo[i] * h[i];
      return sigmoid(z);
      }
}

static void backward(Net *n, const float x[2], 
    const float h[2], 
                      float y, float t, float lr)
{
    int i, j;
    float d_out = -2.0f * (t - y) * y * (1.0f - y);
    float d_h[2];
    for (i = 0; i < 2; i++)
        d_h[i] = d_out * n->wo[i] * h[i]
            * (1.0f - h[i]);
    for (i = 0; i < 2; i++)
        n->wo[i] -= lr * d_out * h[i];
    n->bo -= lr * d_out;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++)
            n->wh[i][j] -= lr * d_h[i] * x[j];
        n->bh[i] -= lr * d_h[i];
    }
}

int main(void)
{
    float X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    float T[4] = { 0, 1, 1, 0 };
    Net net;
    int epoch, s, i, j;

    srand(42);
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++)
            net.wh[i][j] = ((float)rand()/RAND_MAX)*2-1;
        net.bh[i] = ((float)rand()/RAND_MAX)*2-1;
        net.wo[i] = ((float)rand()/RAND_MAX)*2-1;
    }
    net.bo = ((float)rand()/RAND_MAX)*2-1;

    for (epoch = 0; epoch < 10000; epoch++) {
        for (s = 0; s < 4; s++) {
            float h[2], y;
            y = forward(&net, X[s], h);
            backward(&net, X[s], h, y, T[s], 1.0f);
        }
    }

    printf("XOR hidden layer representations:\n\n");
    printf("  input     h0      h1      output  "
           "target\n");
    printf("  ------    ------  ------  ------  "
           "------\n");
    for (s = 0; s < 4; s++) {
        float h[2], y;
        y = forward(&net, X[s], h);
        printf("  (%.0f, %.0f)   %6.4f  %6.4f  %6.4f  "
               "%.0f\n",
               X[s][0], X[s][1], h[0], h[1], y, T[s]);
    }

    printf("\nNow look at just h0 and h1 as a 2D "
           "space:\n");
    printf("  The four input pairs map to four points "
           "in hidden space.\n");
    printf("  In hidden space, class 1 (XOR=1) is "
           "linearly separable\n");
    printf("  from class 0 (XOR=0).\n");

    return 0;
}
