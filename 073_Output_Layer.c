/* 073_Output_Layer.c */
#include <stdio.h>
#include <math.h>

#define N_IN 1
#define N_HID 3
#define N_OUT 1

static float my_tanh(float z)
{
    if (z < -20.0f) return -1.0f;
    float e = expf(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}

static void rnn_step(const float W_x[N_HID][N_IN], 
                     const float W_h[N_HID][N_HID], 
                     const float b_h[N_HID], 
                     float x, 
                     const float h_old[N_HID], 
                     float h_new[N_HID])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = b_h[i] + W_x[i][0] * x;
        for (j = 0; j < N_HID; j++)
            z += W_h[i][j] * h_old[j];
        h_new[i] = my_tanh(z);
    }
}

static float rnn_output(const float W_y[N_OUT][N_HID], 
                         const float b_y[N_OUT], 
                         const float h[N_HID])
{
    float y = b_y[0];
    int j;
    for (j = 0; j < N_HID; j++)
        y += W_y[0][j] * h[j];
    return y;
}

int main(void)
{
    float W_x[N_HID][N_IN] = 
        {{ 0.8f }, { 0.3f }, { -0.2f }};
    float W_h[N_HID][N_HID] = {
        { 0.5f, 0.1f, 0.0f }, 
        { 0.0f, 0.5f, 0.1f }, 
        { 0.1f, 0.0f, 0.5f }, 
    };
    float b_h[N_HID] = { 0, 0, 0 };
    float W_y[N_OUT][N_HID] = {{ 1.0f, 1.0f, 1.0f }};
    float b_y[N_OUT] = { 0 };

    float h[N_HID] = { 0, 0, 0 };
    float h_new[N_HID];
    float seq[] = { 1, 0, 1, 1, 0, 1, 0, 0 };
    int len = 8;
    int t, i;

    /* Track running count for comparison */
    int count = 0;

    printf("RNN with output layer "
           "(trying to count 1s):\n\n");
    printf("  t  input  output   ideal   "
           "h0      h1      h2\n");
    printf("  -- -----  ------   -----   "
           "------  ------  ------\n");

    for (t = 0; t < len; t++) {
        rnn_step(W_x, W_h, b_h, seq[t], h, h_new);
        float y = rnn_output(W_y, b_y, h_new);
        count += (int)seq[t];

        printf("  %d    %.0f    %6.3f   %5.1f   "
               "%6.3f  %6.3f  %6.3f\n",
               t, seq[t], y, (float)count, 
               h_new[0], h_new[1], h_new[2]);

        for (i = 0; i < N_HID; i++)
            h[i] = h_new[i];
    }

    printf("\nThe output does not match the "
           "ideal count.\n");
    printf("The weights are random. Training "
           "will fix this.\n");

    return 0;
}
