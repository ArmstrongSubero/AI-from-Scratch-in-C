/* 097_Decoder.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}

#define N_HID 4
#define VOCAB_SIZE 6  /* simplified output vocabulary */

static void rnn_step(const float wx[N_HID], 
                      const float wh[N_HID][N_HID], 
                     const float bh[N_HID], float x, 
                     const float hp[N_HID], 
                         float hn[N_HID])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = bh[i] + wx[i] * x;
        for (j = 0; j < N_HID; j++)
            z += wh[i][j] * hp[j];
        hn[i] = my_tanh(z);
    }
}

static void softmax(float *x, int n)
{
    float mx = -FLT_MAX, s = 0;
    int i;
    for (i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

int main(void)
{
    /* This context vector came from encoding
       "hello" in the previous step */
    float context[N_HID] = { 0.45f, -0.23f, 
        0.61f, 0.18f };

    /* Decoder weights */
    float wx[N_HID] = { 0.4f, -0.2f, 0.3f, 0.1f };
    float wh[N_HID][N_HID] = {
        { 0.3f, 0.1f, 0, 0 }, 
        { 0, 0.3f, 0.1f, 0 }, 
        { 0, 0, 0.3f, 0.1f }, 
        { 0.1f, 0, 0, 0.3f }, 
    };
    float bh[N_HID] = { 0, 0, 0, 0 };

    /* Output projection: hidden -> vocabulary */
    float W_out[VOCAB_SIZE][N_HID] = {
        /* token 0, <START> */
        { 0.5f, -0.1f, 0.3f, 0.2f }, 
        /* token 1: 'b' */
        { -0.3f, 0.4f, 0.1f, -0.2f }, 
        /* token 2: 'o' */
        { 0.2f, 0.3f, -0.4f, 0.1f }, 
        /* token 3: 'n' */
        { -0.1f, -0.2f, 0.5f, 0.3f }, 
        /* token 4: 'j' */
        { 0.3f, 0.1f, 0.1f, -0.4f }, 
        /* token 5: <END> */
        { 0.1f, 0.2f, 0.1f, 0.5f }, 
    };
    const char *vocab[] = { "<START>", "b", "o",
                            "n", "j", "<END>" };

    float h[N_HID], h_new[N_HID];
    float input_token;
    int t, i, j;

    /* Start the decoder from the context vector */
    for (i = 0; i < N_HID; i++) h[i] = context[i];

    printf("Decoder: generating output sequence\n\n");
    printf("  t  input_tok  hidden               "
           "probs                    pred\n");

    /* First input is always <START>, token 0 */
    input_token = 0.0f;

    for (t = 0; t < 6; t++) {
        /* RNN step */
        rnn_step(wx, wh, bh, input_token, h, h_new);

        /* Project to vocabulary */
        float logits[VOCAB_SIZE];
        for (i = 0; i < VOCAB_SIZE; i++) {
            logits[i] = 0;
            for (j = 0; j < N_HID; j++)
                logits[i] += W_out[i][j] * h_new[j];
        }
        softmax(logits, VOCAB_SIZE);

        /* Find prediction */
        int pred = 0;
        for (i = 1; i < VOCAB_SIZE; i++)
            if (logits[i] > logits[pred]) pred = i;

        printf("  %d  %-8s  "
               "[%+.2f,%+.2f,%+.2f,%+.2f]  ", t,
               t == 0 ? "<START>" : vocab[pred],
               h_new[0], h_new[1], h_new[2], h_new[3]);
        for (i = 0; i < VOCAB_SIZE; i++)
            printf("%.2f ", logits[i]);
        printf(" -> %s\n", vocab[pred]);

        /* Feed prediction as next input
           (autoregressive) */
        input_token = (float)pred / VOCAB_SIZE;
        for (i = 0; i < N_HID; i++) h[i] = h_new[i];

        /* Stop if <END> predicted */
        if (pred == 5) break;
    }

    printf("\nThe decoder generates tokens one at "
           "a time.\n");
    printf("Each token feeds back as input to the "
           "next step.\n");
    printf("Generation stops when "
           "<END> is predicted.\n");

    return 0;
}
