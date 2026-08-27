/* 134_Full_Transformer.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float randf(void)
{
    return(float)rand()/RAND_MAX;
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

#define DM 8
#define DK 4
#define DF 16
#define VOCAB 10
#define MAX_LEN 6

/* Simple single-head attention */
static void attn(const float *Q, int ql, 
                 const float *K, int kl, 
                 const float *V, 
                 const float *WQ, const float *WK, 
                 const float *WV, const float *WO, 
                  int causal, float *out)
{
    float scale = 1.0f / sqrtf((float)DK);
    int i, j, k, d;
    for (i = 0; i < ql; i++) {
        float q[DK];
        for (k = 0; k < DK; k++) {
            q[k] = 0;
            for (d = 0; d < DM; d++)
                q[k] += WQ[k*DM + d] * Q[i*DM + d];
        }
        float scores[MAX_LEN];
        for (j = 0; j < kl; j++) {
            float key[DK];
            for (k = 0; k < DK; k++) {
                key[k] = 0;
                for (d = 0; d < DM; d++)
                    key[k] += WK[k*DM + d]
                        * K[j*DM + d];
            }
            scores[j] = dot(q, key, DK) * scale;
            /* the mask, and the whole of it */
            if (causal && j > i) scores[j] = -1e9f;
        }
        softmax(scores, kl);
        float head[DK] = {0};
        for (j = 0; j < kl; j++)
            for (k = 0; k < DK; k++) {
                float v = 0;
                for (d = 0; d < DM; d++)
                    v += WV[k*DM + d] * V[j*DM + d];
                head[k] += scores[j] * v;
            }
        for (d = 0; d < DM; d++) {
            out[i*DM + d] = 0;
            for (k = 0; k < DK; k++)
                out[i*DM + d] += WO[d*DK + k] * head[k];
        }
    }
}

int main(void)
{
    int enc_len = 4, dec_len = 3;

    /* Token IDs */
    /* source sentence */
    int enc_tokens[] = { 1, 3, 5, 7 };
    /* target, shifted right */
    int dec_tokens[] = { 0, 2, 4 };

    /* Embedding table */
    float embed[VOCAB][DM];
    srand(42);
    int i, j;
    for (i = 0; i < VOCAB; i++)
        for (j = 0; j < DM; j++)
            embed[i][j] = (randf()*2-1)*0.3f;

    /* Sinusoidal positional encoding */
    float pe[MAX_LEN][DM];
    for (i = 0; i < MAX_LEN; i++)
        for (j = 0; j < DM; j++) {
            float e = (float)(j/2*2) / DM;
            float angle = i / powf(10000.0f, e);
            pe[i][j] = (j%2==0)
                ? sinf(angle)
                : cosf(angle);
        }

    /* === EMBED + POSITION === */
    float enc_x[MAX_LEN*DM], dec_x[MAX_LEN*DM];
    for (i = 0; i < enc_len; i++)
        for (j = 0; j < DM; j++)
            enc_x[i*DM + j] = 
                embed[enc_tokens[i]][j] + pe[i][j];
    for (i = 0; i < dec_len; i++)
        for (j = 0; j < DM; j++)
            dec_x[i*DM + j] = 
                embed[dec_tokens[i]][j] + pe[i][j];

    /* Weights (all random for demonstration) */
    float enc_WQ[DK*DM], enc_WK[DK*DM];
    float enc_WV[DK*DM], enc_WO[DM*DK];
    float enc_W1[DF*DM], enc_W2[DM*DF];
    float dec_WQ1[DK*DM], dec_WK1[DK*DM];
    float dec_WV1[DK*DM], dec_WO1[DM*DK];
    float dec_WQ2[DK*DM], dec_WK2[DK*DM];
    float dec_WV2[DK*DM], dec_WO2[DM*DK];
    float dec_W1[DF*DM], dec_W2[DM*DF];
    float out_W[VOCAB*DM];

    float *all[] = {
        enc_WQ, enc_WK, enc_WV, enc_WO, 
        enc_W1, enc_W2, 
        dec_WQ1, dec_WK1, dec_WV1, dec_WO1, 
        dec_WQ2, dec_WK2, dec_WV2, dec_WO2, 
                    dec_W1, dec_W2, out_W};
    int sz[] = {DK*DM, DK*DM, DK*DM, DM*DK, 
        DF*DM, DM*DF, 
        DK*DM, DK*DM, DK*DM, DM*DK, 
        DK*DM, DK*DM, DK*DM, DM*DK, 
                DF*DM, DM*DF, VOCAB*DM};
    for (i = 0; i < 17; i++)
        for (j = 0; j < sz[i]; j++)
            all[i][j] = (randf()*2-1)*0.1f;

    printf("Full Encoder-Decoder Transformer\n");
    printf("  Encoder %d tokens, decoder %d tokens, "
           "d_model=%d\n\n", enc_len, dec_len, DM);

    /* === ENCODER (1 block) === */
    printf("1. Encoder self-attention + FFN\n");
    float norm[MAX_LEN*DM], sub[MAX_LEN*DM];
    /* Self-attention */
    for (i = 0; i < enc_len; i++)
        rmsnorm(&enc_x[i*DM], &norm[i*DM], DM);
    attn(norm, enc_len, norm, enc_len, norm, 
         enc_WQ, enc_WK, enc_WV, enc_WO, 0, sub);
    for(i = 0;i<enc_len*DM;i++) enc_x[i] += sub[i];
    /* FFN */
    for (i = 0; i < enc_len; i++)
        rmsnorm(&enc_x[i*DM], &norm[i*DM], DM);
    for(i = 0;i<enc_len;i++) {
        float hid[DF];
        int f;
        for (f = 0; f < DF; f++) {
            float z = 0;
            for (j = 0; j < DM; j++)
                z += enc_W1[f*DM + j] * norm[i*DM + j];
            hid[f] = gelu(z);
        }
        for (j = 0; j < DM; j++) {
            float z = 0;
            for (int f = 0; f < DF; f++)
                z += enc_W2[j*DF + f] * hid[f];
            enc_x[i*DM + j] += z;
        }
    }

    /* === DECODER (1 block) === */
    printf("2. Decoder causal self-attention\n");
    for (i = 0; i < dec_len; i++)
        rmsnorm(&dec_x[i*DM], &norm[i*DM], DM);
    /* causal flag on, this is self-attention */
    attn(norm, dec_len, norm, dec_len, norm, 
         dec_WQ1, dec_WK1, dec_WV1, dec_WO1, 1, sub);
    for(i = 0;i<dec_len*DM;i++) dec_x[i] += sub[i];

    printf("3. Decoder cross-attention to encoder\n");
    for (i = 0; i < dec_len; i++)
        rmsnorm(&dec_x[i*DM], &norm[i*DM], DM);
    /* queries from dec, keys and values from enc */
    attn(norm, dec_len, enc_x, enc_len, enc_x, 
         dec_WQ2, dec_WK2, dec_WV2, dec_WO2, 0, sub);
    for(i = 0;i<dec_len*DM;i++) dec_x[i] += sub[i];

    printf("4. Decoder FFN\n");
    for (i = 0; i < dec_len; i++)
        rmsnorm(&dec_x[i*DM], &norm[i*DM], DM);
    for(i = 0;i<dec_len;i++) {
        float hid[DF];
        int f;
        for (f = 0; f < DF; f++) {
            float z = 0;
            for (j = 0; j < DM; j++)
                z += dec_W1[f*DM + j] * norm[i*DM + j];
            hid[f] = gelu(z);
        }
        for (j = 0; j < DM; j++) {
            float z = 0;
            for (int f = 0; f < DF; f++)
                z += dec_W2[j*DF + f] * hid[f];
            dec_x[i*DM + j] += z;
        }
    }

    /* === OUTPUT HEAD === */
    printf("5. Output projection -> vocabulary "
           "logits\n\n");
    for (i = 0; i < dec_len; i++) {
        float logits[VOCAB];
        for (j = 0; j < VOCAB; j++) {
            logits[j] = 0;
            for (int k = 0; k < DM; k++)
                logits[j] += out_W[j*DM+k]
                    * dec_x[i*DM+k];
        }
        softmax(logits, VOCAB);

        int pred = 0;
        for (j = 1; j < VOCAB; j++)
            if (logits[j] > logits[pred]) pred = j;

        printf("  Dec pos %d: token %d at prob %.3f\n",
               i, pred, logits[pred]);
    }

    printf("\nComplete pipeline:\n");
    printf("  source -> embed+PE -> encoder\n");
    printf("  target -> embed+PE -> decoder\n");
    printf("  -> output head -> softmax -> tokens\n");

    return 0;
}
