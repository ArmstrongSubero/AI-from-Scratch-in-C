/* 092_Two_Directions.c */
#include <stdio.h>
#include <math.h>

static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}

#define N_HID 2
#define SEQ_LEN 5

/* Simple RNN step,
   h_new = tanh(w_x * x + w_h * h_old + b) */
static void rnn_step(const float w_x[N_HID], 
                     const float w_h[N_HID][N_HID], 
                     const float b[N_HID], float x, 
                     const float h_old[N_HID], 
                     float h_new[N_HID])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = b[i] + w_x[i] * x;
        for (j = 0; j < N_HID; j++)
            z += w_h[i][j] * h_old[j];
        h_new[i] = my_tanh(z);
    }
}

int main(void)
{
    /* A short sequence of values */
    float seq[SEQ_LEN] = { 1.0f, 0.0f, -1.0f, 
        0.5f, 2.0f };

    /* Forward RNN weights */
    float fw_x[N_HID] = { 0.5f, -0.3f };
    float fw_h[N_HID][N_HID] = 
        {{ 0.4f, 0.1f }, { -0.1f, 0.4f }};
    float fw_b[N_HID] = { 0, 0 };

    /* Backward RNN weights (different from forward) */
    float bw_x[N_HID] = { -0.2f, 0.6f };
    float bw_h[N_HID][N_HID] = 
        {{ 0.3f, -0.1f }, { 0.2f, 0.3f }};
    float bw_b[N_HID] = { 0, 0 };

    /* Storage for hidden states */
    float h_fwd[SEQ_LEN][N_HID];
    float h_bwd[SEQ_LEN][N_HID];
    float h_prev[N_HID] = { 0, 0 };
    int t, i;

    /* Forward pass: left to right */
    for (t = 0; t < SEQ_LEN; t++) {
        rnn_step(fw_x, fw_h, fw_b, seq[t], 
                 t == 0 ? (float[]){0, 0} : h_fwd[t-1], 
                 h_fwd[t]);
    }

    /* Backward pass: right to left */
    for (t = SEQ_LEN - 1; t >= 0; t--) {
        rnn_step(bw_x, bw_h, bw_b, seq[t], 
                 t == SEQ_LEN-1
                     ? (float[]){0, 0} : h_bwd[t+1], 
                 h_bwd[t]);
    }

    printf("Forward and backward hidden states:\n\n");
    printf("  t  input   fwd[0]  fwd[1]   "
           "bwd[0]  bwd[1]\n");
    printf("  -- ------  ------  ------   "
           "------  ------\n");
    for (t = 0; t < SEQ_LEN; t++) {
        printf("  %d  %+4.1f   %+.3f  %+.3f   "
               "%+.3f  %+.3f\n",
               t, seq[t], 
               h_fwd[t][0], h_fwd[t][1], 
               h_bwd[t][0], h_bwd[t][1]);
    }

    printf("\nAt each position, fwd has seen "
           "everything to the left,\n");
    printf("bwd has seen everything to the right.\n");

    return 0;
}
