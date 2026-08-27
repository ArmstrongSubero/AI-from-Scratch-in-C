/* 155_Pretrain.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define VOCAB 28    /* a-z + space + period */
#define DM 16
#define N_HID 32
#define SEQ_LEN 8

static int char_to_id(char c)
{
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c == ' ') return 26;
    if (c == '.') return 27;
    return 26;  /* unknown -> space */
}

static char id_to_char(int id)
{
    if (id < 26) return 'a' + id;
    if (id == 26) return ' ';
    return '.';
}

/* Simple one-layer network: embed -> hidden ->
   output */
typedef struct {
    float embed[VOCAB][DM];
    float W1[N_HID][DM];
    float b1[N_HID];
    float W2[VOCAB][N_HID];
    float b2[VOCAB];
}
Model;

static void model_init(Model *m)
{
    int i, j;
    for (i = 0; i < VOCAB; i++)
        for (j = 0; j < DM; j++)
            m->embed[i][j] = (randf()*2-1) * 0.1f;
    for (i = 0; i < N_HID; i++) {
        m->b1[i] = 0;
        for (j = 0; j < DM; j++)
            m->W1[i][j] = (randf()*2-1) * 0.2f;
    }
    for (i = 0; i < VOCAB; i++) {
        m->b2[i] = 0;
        for (j = 0; j < N_HID; j++)
            m->W2[i][j] = (randf()*2-1) * 0.2f;
    }
}

static float relu(float x)
{
    return x > 0 ? x : 0;
}

/* Forward: given previous chars, predict next char */
static void forward(const Model *m, int input_id, 
                    float hidden[N_HID], 
                     float logits[VOCAB])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        hidden[i] = m->b1[i];
        for (j = 0; j < DM; j++)
            hidden[i] += 
                m->W1[i][j] * m->embed[input_id][j];
        hidden[i] = relu(hidden[i]);
    }
    for (i = 0; i < VOCAB; i++) {
        logits[i] = m->b2[i];
        for (j = 0; j < N_HID; j++)
            logits[i] += m->W2[i][j] * hidden[j];
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

/* Train one step: predict target from input,
   backprop */
/* Average loss over a corpus. A single argmax is
   too crude to tell whether pretraining survived, 
   so measure the whole distribution instead. */
static float corpus_loss(const Model *m, const char *d)
{
    int len = (int)strlen(d), i;
    float total = 0;
    for (i = 0; i < len - 1; i++) {
        float h[N_HID], lo[VOCAB];
        forward(m, char_to_id(d[i]), h, lo);
        softmax(lo, VOCAB);
        total += -logf(lo[char_to_id(d[i+1])] + 1e-9f);
    }
    return total / (len - 1);
}

static float train_step(Model *m, int input_id, 
                        int target_id, float lr)
{
    float hidden[N_HID], logits[VOCAB], probs[VOCAB];
    int i, j;

    forward(m, input_id, hidden, logits);
    memcpy(probs, logits, sizeof(logits));
    softmax(probs, VOCAB);

    float loss = -logf(probs[target_id] + 1e-8f);

    /* d_loss/d_logits is probs - one_hot(target) */
    float d_logits[VOCAB];
    for (i = 0; i < VOCAB; i++)
        d_logits[i] = 
            probs[i] - (i == target_id ? 1.0f : 0.0f);

    /* Hidden gradients, taken from W2 as it stands
       before any update touches it. Updating W2 first
       and differentiating through the new values is a
       bug that produces a wrong gradient silently. */
    float d_hidden[N_HID];
    for (j = 0; j < N_HID; j++) {
        d_hidden[j] = 0;
        for (i = 0; i < VOCAB; i++)
            d_hidden[j] += d_logits[i] * m->W2[i][j];
        if (hidden[j] <= 0) d_hidden[j] = 0;  /* ReLU */
    }

    /* W2, b2 gradients */
    for (i = 0; i < VOCAB; i++) {
        m->b2[i] -= lr * d_logits[i];
        for (j = 0; j < N_HID; j++)
            m->W2[i][j] -= lr * d_logits[i] * hidden[j];
    }

    /* W1, b1, embedding gradients */
    float d_embed[DM] = {0};
    for (i = 0; i < N_HID; i++) {
        m->b1[i] -= lr * d_hidden[i];
        for (j = 0; j < DM; j++) {
            d_embed[j] += d_hidden[i] * m->W1[i][j];
            m->W1[i][j] -= 
                lr * d_hidden[i] * m
                    ->embed[input_id][j];
        }
    }
    for (j = 0; j < DM; j++)
        m->embed[input_id][j] -= lr * d_embed[j];

    return loss;
}

int main(void)
{
    Model m;
    srand(42);
    model_init(&m);

    /* Pretraining corpus: simple repeating patterns */
    const char *corpus = 
        "the cat sat. the dog ran. "
        "the cat ran. the dog sat. "
        "a big cat. a big dog. "
        "the big cat sat. a dog ran. ";
    int corpus_len = strlen(corpus);

    printf("=== PRETRAINING ===\n\n");
    printf("  Corpus: \"%.*s...\"\n", 40, corpus);
    printf("  Length: %d characters\n\n", corpus_len);

    /* Train */
    int epoch;
    for (epoch = 0; epoch < 400; epoch++) {
        float total_loss = 0;
        int count = 0;
        int i;

        for (i = 0; i < corpus_len - 1; i++) {
            int input = char_to_id(corpus[i]);
            int target = char_to_id(corpus[i + 1]);
            total_loss += 
                train_step(&m, input, target, 0.01f);
            count++;
        }

        if ((epoch + 1) % 100 == 0)
            printf("  Epoch %3d: avg loss = %.3f\n",
                   epoch + 1, total_loss / count);
    }

    /* Test: generate from pretrained model */
    printf("\n  Generate from the pretrained model, "
           "seed 't'\n    ");
    int cur = char_to_id('t');
    printf("t");
    for (int i = 0; i < 30; i++) {
        float hidden[N_HID], logits[VOCAB];
        forward(&m, cur, hidden, logits);
        softmax(logits, VOCAB);
        /* Greedy */
        int best = 0;
        for (int j = 1; j < VOCAB; j++)
            if (logits[j] > logits[best]) best = j;
        printf("%c", id_to_char(best));
        cur = best;
    }
    printf("\n");

    /* Save weights for fine-tuning */
    printf("\n  Pretrained weights saved. The model "
           "has learned\n");
    printf("  basic character patterns from the "
           "corpus.\n");

    /* === FINE-TUNING === */
    float loss_before = corpus_loss(&m, corpus);

    printf("\n=== FINE-TUNING ===\n\n");

    /* New task, after 'q' always output 'u' */
    const char
        *ft_data = "qu qu qu qu qu qu qu qu qu qu ";
    int ft_len = strlen(ft_data);

    printf("  Fine-tuning data: \"%s\"\n", ft_data);
    printf("  Task, learn that 'q' is always "
           "followed by 'u'\n\n");

    /* What does the model predict after 'q'? */
    {
        float hidden[N_HID], logits[VOCAB];
        forward(&m, char_to_id('q'), hidden, logits);
        softmax(logits, VOCAB);
        int best = 0;
        for (int j = 1; j < VOCAB; j++)
            if (logits[j] > logits[best]) best = j;
        printf("  Before fine-tuning: 'q' -> '%c' "
               "at prob %.3f\n",
               id_to_char(best), logits[best]);
        printf("  P('u' | 'q') = %.3f\n",
               logits[char_to_id('u')]);
    }

    /* Fine-tune with smaller learning rate */
    for (epoch = 0; epoch < 50; epoch++) {
        float total_loss = 0;
        int count = 0;
        for (int i = 0; i < ft_len - 1; i++) {
            int input = char_to_id(ft_data[i]);
            int target = char_to_id(ft_data[i + 1]);
            total_loss += 
                train_step(&m, input, target, 0.002f);
            count++;
        }
    }

    /* After fine-tuning */
    {
        float hidden[N_HID], logits[VOCAB];
        forward(&m, char_to_id('q'), hidden, logits);
        softmax(logits, VOCAB);
        int best = 0;
        for (int j = 1; j < VOCAB; j++)
            if (logits[j] > logits[best]) best = j;
        printf("\n  After fine-tuning:  'q' -> '%c' "
               "at prob %.3f\n",
               id_to_char(best), logits[best]);
        printf("  P('u' | 'q') = %.3f\n",
               logits[char_to_id('u')]);
    }

    /* Did the pretraining survive? Report the loss
       on the original corpus rather than guessing
       from one prediction. */
    printf("\n  Loss on the pretraining corpus\n");
    printf("    before fine-tuning: %.3f\n",
           loss_before);
    printf("    after  fine-tuning: %.3f\n",
           corpus_loss(&m, corpus));

    printf("\n  The new pattern was learned from 30\n");
    printf("  characters of data. Whether the old\n");
    printf("  corpus survived is the "
           "loss above, and\n");
    printf("  a rise there is the cost "
           "of the change.\n");

    return 0;
}
