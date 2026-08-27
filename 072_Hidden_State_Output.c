/* 072_Hidden_State_Output.c */
#include <stdio.h>
#include <math.h>

#define N_IN 1
#define N_HID 3

static float my_tanh(float z)
{
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}

/* h_new = tanh(W_x * x + W_h * h_old + b) */
static void rnn_step(const float W_x[N_HID][N_IN], 
                     const float W_h[N_HID][N_HID], 
                     const float b[N_HID], 
                     float x, 
                     const float h_old[N_HID], 
                     float h_new[N_HID])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = b[i] + W_x[i][0] * x;
        for (j = 0; j < N_HID; j++)
            z += W_h[i][j] * h_old[j];
        h_new[i] = my_tanh(z);
    }
}

int main(void)
{
    /* Hand-picked weights that demonstrate
       the state update */
    float W_x[N_HID][N_IN] = 
        {{ 1.0f }, { 0.5f }, { -0.5f }};
    float W_h[N_HID][N_HID] = {
        { 0.5f, 0.0f, 0.0f }, 
        { 0.0f, 0.5f, 0.0f }, 
        { 0.0f, 0.0f, 0.5f }, 
    };
    float b[N_HID] = { 0, 0, 0 };

    /* initial hidden state */
    float h[N_HID] = { 0, 0, 0 };
    float h_new[N_HID];
    float seq[] = { 1, 0, 1, 1, 0 };
    int len = 5;
    int t, i;

    printf("RNN state update (3 hidden units):\n\n");
    printf("  t  input  h0      h1      h2\n");
    printf("  -- -----  ------  ------  ------\n");

    /* Print initial state */
    printf("  --  init  %6.3f  %6.3f  %6.3f\n",
           h[0], h[1], h[2]);

    for (t = 0; t < len; t++) {
        rnn_step(W_x, W_h, b, seq[t], h, h_new);
        printf("  %d    %.0f    %6.3f  %6.3f  %6.3f\n",
               t, seq[t], h_new[0], h_new[1], h_new[2]);

        for (i = 0; i < N_HID; i++)
            h[i] = h_new[i];
    }

    printf("\nThe hidden state changes at every "
           "step.\n");
    printf("h0 responds strongly to input 1 "
           "(W_x[0] = 1.0).\n");
    printf("h2 responds negatively (W_x[2] = -0.5).\n");
    printf("The W_h diagonal (0.5) gives each "
           "unit memory of itself.\n");

    return 0;
}
