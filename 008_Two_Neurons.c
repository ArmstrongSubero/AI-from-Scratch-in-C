/* 008_Two_Neurons.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

int main(void)
{
    /* Hidden layer weights */
    float wh0[2] = {20.0f, 20.0f}; float bh0 = -10.0f;
    float wh1[2] = {20.0f, 20.0f}; float bh1 = -30.0f;

    /* Output neuron: fires when h0 is on, h1 is off */
    float wo[2] = {20.0f, -20.0f};
    float bo    =  -10.0f;

    float X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1}};
    float T[4]    = {  0,     1,     1,     0   };
    int s;

    printf("Full network XOR:\n");


    for (s = 0; s < 4; s++)
    {
        /* Forward pass: hidden layer */
        float h0 = sigmoid(wh0[0]*X[s][0]
                          + wh0[1]*X[s][1] + bh0);
        float h1 = sigmoid(wh1[0]*X[s][0]
                          + wh1[1]*X[s][1] + bh1);

        /* Forward pass: output layer */
        float y = sigmoid(wo[0]*h0 + wo[1]*h1 + bo);

        printf(" x=(%.0f, %.0f) h0=%.4f"
               " h1=%.4f y=%.4f t=%.0f\n",
               X[s][0], X[s][1],
               h0, h1, y, T[s]);
    }

    return 0;
}
