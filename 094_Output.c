/* 094_Output.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}

#define N_HID 3
#define SEQ_LEN 6
#define N_COMBINED (2 * N_HID)
#define N_CLASSES 5

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
    /* Simulated word embeddings for a 6-word
       sentence */
    float seq[SEQ_LEN] = 
        { 0.2f, -0.5f, 0.8f, -0.1f, 0.6f, -0.3f };
    const char *words[] = { "The", "old", "man",
                            "sat", "on", "chair" };
    const char *labels[] = { "DET", "ADJ", "NOUN",
                             "VERB", "PREP", "NOUN" };

    /* Forward and backward RNN weights */
    float fw_x[N_HID] = {0.5f, -0.3f, 0.2f};
    float fw_h[N_HID][N_HID] = {{0.3f, 0.1f, 0}, 
                               {0, 0.3f, 0.1f}, 
                               {0.1f, 0, 0.3f}};
    float fw_b[N_HID] = {0, 0, 0};
    float bw_x[N_HID] = {-0.2f, 0.4f, 0.1f};
    float bw_h[N_HID][N_HID] = {{0.3f, 0, -0.1f}, 
                               {0.1f, 0.3f, 0}, 
                               {0, 0.1f, 0.3f}};
    float bw_b[N_HID] = {0, 0, 0};

    /* Output weights: N_CLASSES x N_COMBINED */
    /* One row per tag, DET ADJ NOUN VERB PREP */
    float W_out[N_CLASSES][N_COMBINED] = {
        { 1.0f, 0.2f, -0.3f, 0.5f, -0.1f, 0.4f }, 
        { -0.5f, 0.8f, 0.1f, -0.2f, 0.6f, -0.3f }, 
        { 0.2f, -0.4f, 0.7f, 0.1f, -0.3f, 0.5f }, 
        { -0.3f, 0.5f, -0.2f, 0.7f, 0.2f, -0.4f }, 
        { 0.4f, 0.1f, -0.6f, -0.3f, 0.5f, 0.2f }, 
    };
    float b_out[N_CLASSES] = { 0, 0, 0, 0, 0 };

    float h_fwd[SEQ_LEN][N_HID], h_bwd[SEQ_LEN][N_HID];
    int t, i, j;

    /* Forward pass */
    for (t = 0; t < SEQ_LEN; t++)
        rnn_step(fw_x, fw_h, fw_b, seq[t], 
                 t == 0 ? (float[])
                 {
                     0, 0, 0
                 }
                        : h_fwd[t-1], 
                 h_fwd[t]);

    /* Backward pass */
    for (t = SEQ_LEN - 1; t >= 0; t--)
        rnn_step(bw_x, bw_h, bw_b, seq[t], 
                 t == SEQ_LEN-1
                     ? (float[]){0, 0, 0} : h_bwd[t+1], 
                 h_bwd[t]);

    /* Classify each position */
    printf("Bidirectional RNN for sequence "
           "labeling:\n\n");
    printf("  word    gold   "
           "DET   ADJ   NOUN  VERB  PREP   pred\n");
    printf("  ------  -----  ----  ----  ----  "
           "----  ----   ----\n");

    for (t = 0; t < SEQ_LEN; t++) {
        /* Concatenate */
        float combined[N_COMBINED];
        for (i = 0; i < N_HID; i++) {
            combined[i] = h_fwd[t][i];
            combined[N_HID + i] = h_bwd[t][i];
        }

        /* Linear + softmax */
        float logits[N_CLASSES];
        for (i = 0; i < N_CLASSES; i++) {
            logits[i] = b_out[i];
            for (j = 0; j < N_COMBINED; j++)
                logits[i] += W_out[i][j] * combined[j];
        }
        softmax(logits, N_CLASSES);

        /* Find predicted class */
        int pred = 0;
        for (i = 1; i < N_CLASSES; i++)
            if (logits[i] > logits[pred]) pred = i;

        const char *class_names[] = { "DET", "ADJ",
                                      "NOUN", "VERB",
                                      "PREP" };
        printf("  %-6s  %-5s  %.2f  %.2f  %.2f  "
               "%.2f  %.2f   %s\n",
               words[t], labels[t], 
               logits[0], logits[1], logits[2], 
               logits[3], logits[4], 
               class_names[pred]);
    }

    printf("\nWeights are random, so predictions "
           "are wrong.\n");
    printf("With training, the bidirectional "
           "context would let\n");
    printf("the model correctly tag each word.\n");

    return 0;
}
