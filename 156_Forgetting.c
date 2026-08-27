/* 156_Forgetting.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}
static float relu(float x)
{
    return x > 0 ? x : 0;
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

#define V 28
#define DM 16
#define NH 32

typedef struct {
    float embed[V][DM];
    float W1[NH][DM], b1[NH];
    float W2[V][NH], b2[V];
}
Model;

static int c2i(char c)
{
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c == ' ') return 26;
    return 27;
}
static char i2c(int i)
{
    if (i < 26) return 'a' + i;
    if (i == 26) return ' ';
    return '.';
}

static void fwd(const Model *m, int id, 
                float h[NH], float lo[V])
{
    int i, j;
    for (i = 0; i < NH; i++) {
        h[i] = m->b1[i];
        for (j = 0; j < DM; j++)
            h[i] += m->W1[i][j] * m->embed[id][j];
        h[i] = relu(h[i]);
    }
    for (i = 0; i < V; i++) {
        lo[i] = m->b2[i];
        for (j = 0; j < NH; j++)
            lo[i] += m->W2[i][j] * h[j];
    }
}

static void train(Model *m, const char *data, 
                  int epochs, float lr)
{
    int len = strlen(data), ep, i, j, k;
    for (ep = 0; ep < epochs; ep++)
    for (i = 0; i < len - 1; i++) {
        float h[NH], lo[V], pr[V];
        int inp = c2i(data[i]), tgt = c2i(data[i+1]);
        fwd(m, inp, h, lo);
        memcpy(pr, lo, sizeof(lo));
        softmax(pr, V);

        float dl[V];
        for (j = 0; j < V; j++)
            dl[j] = pr[j] - (j == tgt ? 1.0f : 0.0f);

        /* dh must come from W2 as it stands now, so it
           is computed before W2 is touched. Updating
           first and differentiating afterward is a bug
           that quietly corrupts every later
           gradient. */
        float dh[NH];
        for (j = 0; j < NH; j++) {
            dh[j] = 0;
            for (k = 0; k < V; k++)
                dh[j] += dl[k] * m->W2[k][j];
            if (h[j] <= 0) dh[j] = 0;
        }

        for (j = 0; j < V; j++) {
            m->b2[j] -= lr * dl[j];
            for (k = 0; k < NH; k++)
                m->W2[j][k] -= lr * dl[j] * h[k];
        }

        float de[DM] = {0};
        for (j = 0; j < NH; j++) {
            m->b1[j] -= lr * dh[j];
            for (k = 0; k < DM; k++) {
                de[k] += dh[j] * m->W1[j][k];
                m->W1[j][k] -= 
                    lr * dh[j] * m->embed[inp][k];
            }
        }
        for (j = 0; j < DM; j++)
            m->embed[inp][j] -= lr * de[j];
    }
}

static char predict(const Model *m, int inp)
{
    float h[NH], lo[V];
    fwd(m, inp, h, lo);
    softmax(lo, V);
    int best = 0;
    for (int i = 1; i < V; i++)
        if (lo[i] > lo[best]) best = i;
    return i2c(best);
}

/* Average cross-entropy of the model on a corpus.
   This is the number that measures forgetting, since
   a single argmax says almost nothing. */
static float corpus_loss(const Model *m, const char *d)
{
    int len = strlen(d), i, j;
    float total = 0;
    for (i = 0; i < len - 1; i++) {
        float h[NH], lo[V];
        int inp = c2i(d[i]), tgt = c2i(d[i+1]);
        fwd(m, inp, h, lo);
        softmax(lo, V);
        total += -logf(lo[tgt] + 1e-9f);
    }
    return total / (len - 1);
}

/* Fraction of positions the model gets right */
static float corpus_acc(const Model *m, const char *d)
{
    int len = strlen(d), i, j, hit = 0;
    for (i = 0; i < len - 1; i++) {
        float h[NH], lo[V];
        int inp = c2i(d[i]), 
            tgt = c2i(d[i+1]), best = 0;
        fwd(m, inp, h, lo);
        for (j = 1; j < V; j++)
            if (lo[j] > lo[best]) best = j;
        if (best == tgt) hit++;
    }
    return 100.0f * hit / (len - 1);
}

static void init(Model *m)
{
    int i, j;
    srand(42);
    for (i = 0; i < V; i++)
        for (j = 0; j < DM; j++)
            m->embed[i][j] = (randf()*2-1)*0.1f;
    for (i = 0; i < NH; i++) {
        m->b1[i] = 0;
        for (j = 0; j < DM; j++)
            m->W1[i][j] = (randf()*2-1)*0.2f;
    }
    for (i = 0; i < V; i++) {
        m->b2[i] = 0;
        for (j = 0; j < NH; j++)
            m->W2[i][j] = (randf()*2-1)*0.2f;
    }
}

int main(void)
{
    Model base, m;

    const char *pretrain_data = 
        "the cat sat. the dog ran. the cat ran. "
        "the dog sat. a big cat. a big dog. "
        "the big cat sat. a dog ran. ";
    const char *ft_data = 
        "xyz xyz xyz xyz xyz xyz xyz xyz xyz xyz ";

    printf("Catastrophic forgetting, measured\n\n");

    init(&base);
    printf("  before pretraining   old loss %.3f  "
           "old acc %5.1f%%\n",
           corpus_loss(&base, pretrain_data), 
           corpus_acc(&base, pretrain_data));

    train(&base, pretrain_data, 300, 0.05f);
    printf("  after  pretraining   old loss %.3f  "
           "old acc %5.1f%%\n\n",
           corpus_loss(&base, pretrain_data), 
           corpus_acc(&base, pretrain_data));

    printf("  Now fine-tune the same "
           "pretrained model\n");
    printf("  on 'xyz xyz ...' at three learning "
           "rates.\n\n");

    printf("     lr   epochs   old loss  old acc   "
           "new acc\n");
    printf("  -----   ------   --------  -------   "
           "-------\n");

    float rates[] = { 0.0005f, 0.005f, 0.05f };
    int eps[] = { 50, 100, 200 };
    int k;
    for (k = 0; k < 3; k++) {
        /* fresh copy each time */
        m = base;
        train(&m, ft_data, eps[k], rates[k]);
        printf("  %.4f   %6d   %8.3f  %6.1f%%   "
               "%6.1f%%\n",
               rates[k], eps[k], 
               corpus_loss(&m, pretrain_data), 
               corpus_acc(&m, pretrain_data), 
               corpus_acc(&m, ft_data));
    }

    printf("\n  Read the two accuracy "
           "columns against\n");
    printf("  each other. The gentlest "
           "run keeps most\n");
    printf("  of what pretraining taught and learns\n");
    printf("  little. The harshest learns the new "
           "task\n");
    printf("  completely and throws "
           "the old one away.\n");
    printf("  That trade is the whole "
           "of the problem,\n");
    printf("  and the learning rate is the dial.\n");

    return 0;
}
