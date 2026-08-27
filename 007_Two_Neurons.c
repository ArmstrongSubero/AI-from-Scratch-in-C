/* 007_Two_Neurons.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

int main(void)
{
    /* Hidden neuron 0: learns "x0 OR x1"*/
    float wh0[2] = {20.0f, 20.0f};
    float bh0    = -10.0f;

    /* Hidden neuron 1: learns "x0 and x1" */
    float wh1[2] = {20.0f, 20.0f};
    float bh1    = -30.0f;

    float X[4][2] = { {0, 0}, {0, 1}, {1,0}, {1, 1}};
    int s;

    printf("Hidden neuron outputs:\n");

    for (s = 0; s < 4; s++)
    {
        float z0 = wh0[0]*X[s][0] + wh0[1]*X[s][1] + bh0;
        float h0 = sigmoid(z0);

        float z1 = wh1[0]*X[s][0] + wh1[1]*X[s][1] + bh1;
        float h1 = sigmoid(z1);

        printf(" x=(%.0f, %.0f)"
               " h0=%.4f h1=%.4f\n",
               X[s][0], X[s][1],
               h0, h1);
    }

    return 0;
}
