/* 096_Encoder.c */
#include <stdio.h>
#include <math.h>

static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}

#define N_HID 4

static void gru_step(const float wx[N_HID], 
                      const float wh[N_HID][N_HID], 
                      const float bh[N_HID], float x, 
                      const float hp[N_HID], 
                      float hn[N_HID])
{
    /* Simplified to a tanh RNN for clarity. The
       principle is the same with a GRU. */
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = bh[i] + wx[i] * x;
        for (j = 0; j < N_HID; j++)
            z += wh[i][j] * hp[j];
        hn[i] = my_tanh(z);
    }
}

int main(void)
{
    /* Encode "hello" as normalized character IDs */
    /* h, e, l, l, o */
    float input[] = { 0.8f, 0.5f, 1.2f, 1.2f, 1.5f };
    int input_len = 5;

    float wx[N_HID] = { 0.5f, -0.3f, 0.2f, 0.4f };
    float wh[N_HID][N_HID] = {
        { 0.3f, 0.1f, 0, 0 }, 
        { 0, 0.3f, 0.1f, 0 }, 
        { 0, 0, 0.3f, 0.1f }, 
        { 0.1f, 0, 0, 0.3f }, 
    };
    float bh[N_HID] = { 0, 0, 0, 0 };

    float h[N_HID] = { 0, 0, 0, 0 };
    float h_new[N_HID];
    int t, i;

    printf("Encoder: reading input sequence\n\n");
    printf("  t  input  hidden state\n");

    for (t = 0; t < input_len; t++) {
        gru_step(wx, wh, bh, input[t], h, h_new);
        printf("  %d  %.1f    "
               "[%+.3f, %+.3f, %+.3f, %+.3f]\n",
               t, input[t], h_new[0], h_new[1], 
               h_new[2], h_new[3]);
        for (i = 0; i < N_HID; i++) h[i] = h_new[i];
    }

    printf("\nContext vector (final hidden state):\n");
    printf("  [%+.3f, %+.3f, %+.3f, %+.3f]\n",
           h[0], h[1], h[2], h[3]);
    printf("\nThis single vector must capture "
           "everything about\n");
    printf("\"hello\" that the decoder needs to "
           "produce \"bonjour\".\n");

    return 0;
}
