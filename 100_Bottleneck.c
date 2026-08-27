/* 100_Bottleneck.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N_HID 4
#define MAX_LEN 50

/* Encode a sequence and return the final hidden
   state */
static void encode(const float wx[N_HID], 
                   const float wh[N_HID][N_HID], 
                   const float bh[N_HID], 
                   const float *x, int len, 
                   float out[N_HID])
{
    float h[N_HID] = {0}, hn[N_HID];
    int t, i, j;
    for (t = 0; t < len; t++) {
        for (i = 0; i < N_HID; i++) {
            float z = bh[i] + wx[i] * x[t];
            for (j = 0; j < N_HID; j++)
                z += wh[i][j] * h[j];
            hn[i] = my_tanh(z);
        }
        for (i = 0; i < N_HID; i++) h[i] = hn[i];
    }
    for (i = 0; i < N_HID; i++) out[i] = h[i];
}

static float dist(const float a[N_HID], 
                  const float b[N_HID])
{
    float s = 0;
    int i;
    for (i = 0; i < N_HID; i++)
        s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrtf(s);
}

static float norm(const float a[N_HID])
{
    float s = 0;
    int i;
    for (i = 0; i < N_HID; i++) s += a[i] * a[i];
    return sqrtf(s);
}

int main(void)
{
    float wx[N_HID], wh[N_HID][N_HID], bh[N_HID];
    float seq[MAX_LEN], alt[MAX_LEN];
    float ctx[N_HID], ctx_alt[N_HID];
    int lengths[] = { 3, 5, 10, 20, 50 };
    int n_lengths = 5;
    int li, t, i, j;

    srand(42);
    for (i = 0; i < N_HID; i++) {
        wx[i] = randf()*0.4f-0.2f;
        bh[i] = 0;
        for (j = 0; j < N_HID; j++)
            wh[i][j] = randf()*0.4f-0.2f;
    }

    printf("Does the first token still affect the "
           "context\n");
    printf("vector after the whole sequence is "
           "read?\n\n");
    printf("  length   |context|   change from   "
           "relative\n");
    printf("           at end      flipping x0   "
           "change\n");
    printf("  ------   ---------   -----------   "
           "---------\n");

    for (li = 0; li < n_lengths; li++) {
        int len = lengths[li];

        /* One random sequence, and a copy whose only
           difference is the very first token */
        srand(7);
        for (t = 0; t < len; t++) {
            seq[t] = randf() * 2.0f - 1.0f;
            alt[t] = seq[t];
        }
        alt[0] = -seq[0];

        encode(wx, wh, bh, seq, len, ctx);
        encode(wx, wh, bh, alt, len, ctx_alt);

        float n0 = norm(ctx);
        float d = dist(ctx, ctx_alt);
        printf("  %4d     %.5f     %.3e     %.3e\n",
               len, n0, d, n0 > 0 ? d / n0 : 0.0f);
    }

    printf("\nFlipping the sign of the first token "
           "barely\n");
    printf("moves the context once the sequence is "
           "long.\n");
    printf("The encoder has not saturated, since the "
           "state\n");
    printf("norm stays small throughout. The "
           "information is\n");
    printf("simply gone, overwritten by everything "
           "after.\n\n");

    printf("This is the bottleneck. The fix is to "
           "stop\n");
    printf("compressing everything into one vector "
           "and\n");
    printf("let the decoder look back "
           "at ALL encoder\n");
    printf("hidden states. That is attention, "
           "Chapter 19.\n");

    return 0;
}
