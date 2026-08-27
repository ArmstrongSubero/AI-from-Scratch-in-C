/* 138_Generate.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}
static float gelu(float x)
{
    float c = 0.7978846f;
    float u = x + 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * u));
}
static float dot(const float *a, const float *b, 
                 int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += a[i] * b[i];
    return s;
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
static void rmsnorm(float *x, int n)
{
    int i;
    double ss = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float ri = 1.0f / sqrtf((float)(ss/n) + eps);
    for (i = 0; i < n; i++) x[i] *= ri;
}

#define VOCAB 8
#define MAX_SEQ 20
#define DM 4

/* Greedy, return the highest scoring token */
static int sample_greedy(const float *logits, int n)
{
    int best = 0, i;
    for (i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* Temperature sampling: add randomness */
static int sample_temperature(float *logits, int n, 
                              float temp)
{
    int i;
    for (i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);

    /* Random sample from the distribution */
    float r = randf();
    float cumsum = 0;
    for (i = 0; i < n; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

int main(void)
{
    /* A tiny stand-in model, embedding plus
       output projection and nothing between */
    float embed[VOCAB][DM];
    int i, j;
    srand(42);
    for (i = 0; i < VOCAB; i++)
        for (j = 0; j < DM; j++)
            embed[i][j] = (randf()*2-1) * 0.5f;

    const char *names[] = { "<PAD>", "the", "cat",
                            "sat", "on", "a",
                            "mat", "<END>" };

    /* Prompt: "the cat" */
    int sequence[MAX_SEQ] = { 1, 2 };  /* the, cat */
    int cur_len = 2;
    int max_gen = 6;

    printf("Autoregressive generation:\n\n");
    printf("  Prompt: ");
    for (i = 0; i < cur_len; i++)
        printf("%s ", names[sequence[i]]);
    printf("\n\n");

    printf("  Generating (greedy):\n");
    for (int step = 0; step < max_gen; step++) {
        /* Use last token's embedding as the "model
           output"
           a real model would run every block here */
        int last = sequence[cur_len - 1];
        float logits[VOCAB];
        for (j = 0; j < VOCAB; j++)
            logits[j] = dot(embed[last], embed[j], DM);

        int next = sample_greedy(logits, VOCAB);
        sequence[cur_len++] = next;

        printf("    Step %d: predict '%s'", step,
               names[next]);
        if (next == 7) {
            printf(" (stop)\n");
            break;
            }
        printf("\n");
    }

    printf("\n  Full sequence: ");
    for (i = 0; i < cur_len; i++)
        printf("%s ", names[sequence[i]]);
    printf("\n");

    printf("\n  With temperature 0.8, same prompt\n");
    for (int trial = 0; trial < 3; trial++) {
        int seq2[MAX_SEQ] = { 1, 2 };
        int len2 = 2;
        printf("    Trial %d: the cat ", trial);
        for (int step = 0; step < 4; step++) {
            int last = seq2[len2-1];
            float logits[VOCAB];
            for (j = 0; j < VOCAB; j++)
                logits[j] = dot(embed[last], 
                    embed[j], DM);
            int next = sample_temperature(logits, 
                                          VOCAB, 0.8f);
            seq2[len2++] = next;
            printf("%s ", names[next]);
            if (next == 7) break;
        }
        printf("\n");
    }

    printf("\n  Temperature controls randomness:\n");
    printf("    temp=0 always takes the top token\n");
    printf("    temp=1 samples the raw distribution\n");
    printf("    temp>1 flattens it, more random\n");
    printf("    temp<1 sharpens it, "
           "more predictable\n");

    return 0;
}
