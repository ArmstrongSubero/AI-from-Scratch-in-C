/* 099_Seq2seq.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

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

#define N_HID 6
#define VOCAB_IN 5
#define VOCAB_OUT 5
#define MAX_LEN 10

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
    /* Input vocab 0=<PAD> 1='a' 2='b' 3='c' 4='d' */
    /* Output vocab 0=<START> 1='A' 2='B' 3='C'
       4 = <END> */
    const char *in_vocab[]  = { "<PAD>", "a", "b",
                                "c", "d" };
    const char *out_vocab[] = { "<START>", "A", "B",
                                "C", "<END>" };

    /* Task: reverse the input sequence */
    /* Input:  a b c  (tokens: 1 2 3) */
    /* Output: C B A  (tokens: 3 2 1 then <END>=4) */

    int input_tokens[] = { 1, 2, 3 };
    int target_tokens[] = { 3, 2, 1, 4 };
    /* C B A <END> */
    int input_len = 3;
    int target_len = 4;

    /* Encoder and decoder weights (random) */
    float enc_wx[N_HID], enc_bh[N_HID];
    float enc_wh[N_HID][N_HID];
    float dec_wx[N_HID], dec_bh[N_HID];
    float dec_wh[N_HID][N_HID];
    float W_out[VOCAB_OUT][N_HID];
    int i, j;

    srand(42);
    for (i = 0; i < N_HID; i++) {
        enc_wx[i] = randf()*0.4f-0.2f;
        enc_bh[i] = 0;
        dec_wx[i] = randf()*0.4f-0.2f;
        dec_bh[i] = 0;
        for (j = 0; j < N_HID; j++) {
            enc_wh[i][j] = randf()*0.4f-0.2f;
            dec_wh[i][j] = randf()*0.4f-0.2f;
        }
    }
    for (i = 0; i < VOCAB_OUT; i++)
        for (j = 0; j < N_HID; j++)
            W_out[i][j] = randf()*0.4f-0.2f;

    /* === ENCODE === */
    float h[N_HID] = {0}, h_new[N_HID];
    printf("=== ENCODER ===\n");
    for (int t = 0; t < input_len; t++) {
        float x = (float)input_tokens[t] / VOCAB_IN;
        rnn_step(enc_wx, enc_wh, enc_bh, x, h, h_new);
        printf("  t=%d  input='%s'  h=[", t,
               in_vocab[input_tokens[t]]);
        for (i = 0; i < N_HID; i++)
            printf("%+.2f%s", h_new[i],
                   i<N_HID-1?",":"");
        printf("]\n");
        for (i = 0; i < N_HID; i++) h[i] = h_new[i];
    }

    printf("\nContext vector: [");
    for (i = 0; i < N_HID; i++)
        printf("%+.3f%s", h[i], i<N_HID-1?", ":"");
    printf("]\n");

    /* === DECODE (with teacher forcing) === */
    printf("\n=== DECODER (teacher forcing) ===\n");
    float dec_input = 0.0f;  /* <START> = token 0 */

    for (int t = 0; t < target_len; t++) {
        rnn_step(dec_wx, dec_wh, dec_bh, dec_input, 
                 h, h_new);

        /* Project to output vocabulary */
        float logits[VOCAB_OUT];
        for (i = 0; i < VOCAB_OUT; i++) {
            logits[i] = 0;
            for (j = 0; j < N_HID; j++)
                logits[i] += W_out[i][j] * h_new[j];
        }
        softmax(logits, VOCAB_OUT);

        int pred = 0;
        for (i = 1; i < VOCAB_OUT; i++)
            if (logits[i] > logits[pred]) pred = i;

        printf("  t=%d  input='%s'  pred='%s'  "
               "target='%s'  %s\n",
               t, 
               t == 0 ? "<START>"
                      : out_vocab[target_tokens[t-1]], 
               out_vocab[pred], 
               out_vocab[target_tokens[t]], 
               pred == target_tokens[t]
                   ? "OK" : "WRONG");

        /* Teacher forcing: use target as next input */
        dec_input = (float)target_tokens[t] / VOCAB_OUT;
        for (i = 0; i < N_HID; i++) h[i] = h_new[i];
    }

    printf("\nWeights are random so predictions are "
           "wrong.\n");
    printf("With training, cross-entropy loss and "
           "backprop through\n");
    printf("both encoder and decoder, this learns "
           "to reverse.\n");

    return 0;
}
