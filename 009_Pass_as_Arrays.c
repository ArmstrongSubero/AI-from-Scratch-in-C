/* 009_Pass_as_Arrays.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

/* 2 input, 2 hidden, 1 output network */
typedef struct {
    float wh[2][2];   /* wh[neuron][input] */
    float bh[2];
    float wo[2];      /* output weights, one per hidden */
    float bo;
} Net;

/* Forward pass. Stores hidden outputs in h[]. */
static float forward(const Net *n,
                     const float x[2], float h[2])
{
    int i, j;

    /* Hidden layer */
    for (i = 0; i < 2; i++) {
        float z = n->bh[i];
        for (j = 0; j < 2; j++)
            z += n->wh[i][j] * x[j];
        h[i] = sigmoid(z);
    }

    /* Output layer */
    {
        float z = n->bo;
        for (i = 0; i < 2; i++)
            z += n->wo[i] * h[i];
        return sigmoid(z);
    }
}

int main(void)
{
    Net net = {
        .wh = {{ 20, 20}, { 20, 20}},
        .bh = {-10, -30},
        .wo = { 20, -20},
        .bo = -10
    };

    float X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
    float T[4]    = {  0,     1,     1,     0    };
    float h[2];
    int s;

    for (s = 0; s < 4; s++) {
        float y = forward(&net, X[s], h);
        printf("x=(%.0f,%.0f)  y=%.4f  target=%.0f\n",
               X[s][0], X[s][1], y, T[s]);
    }
    return 0;
}
