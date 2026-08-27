/* 137_Gpt.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

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
static void rmsnorm(const float *x, float *out, 
                    int n)
{
    int i;
    double ss = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float ri = 1.0f / sqrtf((float)(ss/n) + eps);
    for (i = 0; i < n; i++) out[i] = x[i] * ri;
}

#define VOCAB 8
#define SL 6
#define DM 8
#define NH 2
#define DH (DM/NH)
#define DF 16
#define N_BLOCKS 2

typedef struct {
    float WQ[NH][DH][DM], WK[NH][DH][DM];
    float WV[NH][DH][DM], WO[DM][DM];
    float W1[DF][DM], W2[DM][DF];
}
Block;

static void block_fwd(const Block *b, 
    float X[][DM], int sl)
{
    float norm[SL][DM];
    int h, i, j, k, d;

    /* Causal self-attention */
    for (i = 0; i < sl; i++) rmsnorm(X[i], norm[i], DM);
    float ho[NH][SL][DH], scale = 1.0f/sqrtf((float)DH);
    for (h = 0;h<NH;h++) for (i = 0;i<sl;i++) {
        float q[DH];
        for (k = 0; k < DH; k++) {
            q[k] = 0;
            for (d = 0; d < DM; d++)
                q[k] += b->WQ[h][k][d] * norm[i][d];
        }
        float sc[SL];
        for (j = 0; j < sl; j++) {
            float key[DH];
            for (k = 0; k < DH; k++) {
                key[k] = 0;
                for (d = 0; d < DM; d++)
                    key[k] += b
                        ->WK[h][k][d] * norm[j][d];
            }
            sc[j] = dot(q, key, DH) * scale;
            /* the causal mask, one line */
            if (j > i) sc[j] = -1e9f;
        }
        softmax(sc, sl);
        for (k = 0; k < DH; k++) {
            ho[h][i][k] = 0;
            for (j = 0; j < sl; j++) {
                float v = 0;
                for (d = 0; d < DM; d++)
                    v += b->WV[h][k][d] * norm[j][d];
                ho[h][i][k] += sc[j] * v;
            }
        }
    }
    for (i = 0; i < sl; i++) {
        float cat[DM];
        for (h = 0; h < NH; h++)
            for (k = 0; k < DH; k++)
                cat[h*DH + k] = ho[h][i][k];
        float sub[DM];
        for (j = 0; j < DM; j++) {
            sub[j] = 0;
            for (k = 0; k < DM; k++)
                sub[j] += b->WO[j][k] * cat[k];
        }
        for(j = 0;j<DM;j++)X[i][j] += sub[j];
        }

    /* FFN */
    for(i = 0;i<sl;i++) rmsnorm(X[i], norm[i], DM);
    for(i = 0;i<sl;i++) {
        float hid[DF];
        int f;
        for (f = 0; f < DF; f++) {
            float z = 0;
            for (j = 0; j < DM; j++)
                z += b->W1[f][j] * norm[i][j];
            hid[f] = gelu(z);
        }
        for (j = 0; j < DM; j++) {
            float z = 0;
            for (int f = 0; f < DF; f++)
                z += b->W2[j][f] * hid[f];
            X[i][j] += z;
        }
    }
}

int main(void)
{
    /* Token embeddings */
    float embed[VOCAB][DM];
    /* Positional embeddings (learned) */
    float pos_embed[SL][DM];
    /* Transformer blocks */
    Block blocks[N_BLOCKS];
    /* Output head (weight-tied with embedding) */

    int i, j;
    srand(42);
    for (i = 0; i < VOCAB; i++)
        for (j = 0; j < DM; j++)
            embed[i][j] = (randf()*2-1) * 0.3f;
    for (i = 0; i < SL; i++)
        for (j = 0; j < DM; j++)
            pos_embed[i][j] = (randf()*2-1) * 0.1f;
    for (int b = 0; b < N_BLOCKS; b++) {
        float *w = (float*)&blocks[b];
        int n = (int)(sizeof(Block)/sizeof(float));
        for (i = 0; i < n; i++)
            w[i] = (randf()*2-1) * 0.1f;
    }

    /* Input sequence */
    int tokens[] = { 3, 1, 4, 1, 5, 2 };
    int seq_len = 6;

    printf("GPT-style Decoder-Only Transformer\n");
    printf("  vocab=%d  d_model=%d  heads=%d  "
           "layers=%d\n\n", VOCAB, DM, NH, N_BLOCKS);

    /* 1. Token embedding + position embedding */
    float X[SL][DM];
    for (i = 0; i < seq_len; i++)
        for (j = 0; j < DM; j++)
            X[i][j] = embed[tokens[i]][j]
                + pos_embed[i][j];

    printf("Input tokens: [");
    for (i = 0; i < seq_len; i++)
        printf("%d%s", tokens[i],
               i<seq_len-1?", ":"");
    printf("]\n\n");

    /* 2. Transformer blocks */
    for (int b = 0; b < N_BLOCKS; b++) {
        block_fwd(&blocks[b], X, seq_len);
        printf("  After block %d: "
               "pos0 norm=%.3f\n", b+1,
               sqrtf(dot(X[0], X[0], DM)));
    }

    /* 3. Final RMSNorm */
    for (i = 0; i < seq_len; i++)
        rmsnorm(X[i], X[i], DM);

    /* 4. Output head, weight-tied with the
          embedding table */
    printf("\nNext-token predictions:\n\n");
    printf("  Position  Context          Predicted  "
           "Actual\n");
    printf("  --------  ---------------  ---------  "
           "------\n");

    for (i = 0; i < seq_len - 1; i++) {
        /* Logits = X[i] . embed[v]^T (weight tying) */
        float logits[VOCAB];
        for (j = 0; j < VOCAB; j++)
            logits[j] = dot(X[i], embed[j], DM);
        softmax(logits, VOCAB);

        int pred = 0;
        for (j = 1; j < VOCAB; j++)
            if (logits[j] > logits[pred]) pred = j;

        printf("  %4d      tokens[0..%d]    %5d      "
               "%5d\n",
               i, i, pred, tokens[i+1]);
    }

    printf("\nWeight tying, the output projection\n");
    printf("reuses the embedding matrix transposed.\n");
    printf("It saves parameters, and works because\n");
    printf("similar tokens should have similar\n");
    printf("embeddings AND similar output logits.\n");

    /* Parameter count */
    int emb_params = VOCAB * DM + SL * DM;
    int block_params = 
        N_BLOCKS * (4*DM*DM + 2*DM*DF + 2*DM);
    /* the output head is tied, so it is free */
    int total = emb_params + block_params;
    printf("\nParameters: %d embedding + %d blocks "
           "= %d\n",
           emb_params, block_params, total);
    printf("  The tied output head adds 0.\n");

    return 0;
}
