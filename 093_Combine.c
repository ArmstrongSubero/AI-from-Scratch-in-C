/* 093_Combine.c */
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
#define N_COMBINED (2 * N_HID)  /* forward + backward */

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
    float seq[SEQ_LEN] = { 1.0f, 0.0f, -1.0f, 
        0.5f, 2.0f };

    float fw_x[N_HID] = { 0.5f, -0.3f };
    float fw_h[N_HID][N_HID] = 
        {{ 0.4f, 0.1f }, { -0.1f, 0.4f }};
    float fw_b[N_HID] = { 0, 0 };

    float bw_x[N_HID] = { -0.2f, 0.6f };
    float bw_h[N_HID][N_HID] = 
        {{ 0.3f, -0.1f }, { 0.2f, 0.3f }};
    float bw_b[N_HID] = { 0, 0 };

    float h_fwd[SEQ_LEN][N_HID], h_bwd[SEQ_LEN][N_HID];
    float combined[SEQ_LEN][N_COMBINED];
    int t, i;

    /* Forward pass */
    for (t = 0; t < SEQ_LEN; t++)
        rnn_step(fw_x, fw_h, fw_b, seq[t], 
                 t == 0 ? (float[]){0, 0} : h_fwd[t-1], 
                 h_fwd[t]);

    /* Backward pass */
    for (t = SEQ_LEN - 1; t >= 0; t--)
        rnn_step(bw_x, bw_h, bw_b, seq[t], 
                 t == SEQ_LEN-1
                     ? (float[]){0, 0} : h_bwd[t+1], 
                 h_bwd[t]);

    /* Concatenate */
    for (t = 0; t < SEQ_LEN; t++) {
        for (i = 0; i < N_HID; i++) {
            combined[t][i] = h_fwd[t][i];
            combined[t][N_HID + i] = h_bwd[t][i];
        }
    }

    printf("Combined bidirectional "
           "representations:\n\n");
    printf("  t  input   combined[0..3]\n");
    printf("  -- ------  ------------------------"
           "--------\n");
    for (t = 0; t < SEQ_LEN; t++) {
        printf("  %d  %+4.1f   [%+.3f, %+.3f | "
               "%+.3f, %+.3f]\n",
               t, seq[t], 
               combined[t][0], combined[t][1], 
               combined[t][2], combined[t][3]);
    }

    printf("\nThe | separates forward (left) from "
           "backward (right).\n");
    printf("Each position has a %d-dimensional "
           "representation\n", N_COMBINED);
    printf("that encodes context from both "
           "directions.\n");

    printf("\nThe output layer takes the combined "
           "vector.\n");
    printf("  y_t = W_out * combined_t + b_out\n");
    printf("  W_out has %d columns instead of %d.\n",
           N_COMBINED, N_HID);

    return 0;
}
