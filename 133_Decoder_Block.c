/* 133_Decoder_Block.c */
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

#define EL 4   /* encoder length */
#define DL 3   /* decoder length */
#define DM 6   /* model dimension */
#define DK 3   /* head dimension */
#define DF 12  /* FFN intermediate */

/* Simplified single-head attention */
static void attention(const float Q[][DM], int q_len, 
                       const float K[][DM], int k_len, 
                       const float V[][DM], 
                       const float WQ[DK][DM], 
                       const float WK[DK][DM], 
                       const float WV[DK][DM], 
                       const float WO[DM][DK], 
                       int causal, float out[][DM])
{
    float scale = 1.0f / sqrtf((float)DK);
    int i, j, k, d;

    for (i = 0; i < q_len; i++) {
        float q[DK];
        for (k = 0; k < DK; k++) {
            q[k] = 0;
            for (d = 0; d < DM; d++)
                q[k] += WQ[k][d] * Q[i][d];
        }

        float scores[8];  /* max seq len */
        for (j = 0; j < k_len; j++) {
            float key[DK];
            for (k = 0; k < DK; k++) {
                key[k] = 0;
                for (d = 0; d < DM; d++)
                    key[k] += WK[k][d] * K[j][d];
            }
            scores[j] = dot(q, key, DK) * scale;
            if (causal && j > i) scores[j] = -1e9f;
        }
        softmax(scores, k_len);

        /* Weighted sum of values, then project */
        float head[DK] = {0};
        for (j = 0; j < k_len; j++) {
            for (k = 0; k < DK; k++) {
                float val = 0;
                for (d = 0; d < DM; d++)
                    val += WV[k][d] * V[j][d];
                head[k] += scores[j] * val;
            }
        }
        for (d = 0; d < DM; d++) {
            out[i][d] = 0;
            for (k = 0; k < DK; k++)
                out[i][d] += WO[d][k] * head[k];
        }
    }
}

int main(void)
{
    /* Encoder output, as if from the stack */
    float enc[EL][DM] = {
        {0.5f, 0.1f, -0.3f, 0.8f, 0.2f, -0.1f}, 
        {0.3f, 0.7f, 0.1f, -0.2f, 0.5f, 0.4f}, 
        {-0.1f, 0.4f, 0.6f, 0.3f, -0.4f, 0.2f}, 
        {0.8f, 0.2f, -0.1f, 0.4f, 0.3f, 0.6f}, 
    };
    /* Decoder input */
    float dec[DL][DM] = {
        {0.4f, 0.6f, -0.2f, 0.3f, 0.1f, 0.5f}, 
        {0.1f, 0.3f, 0.5f, -0.1f, 0.4f, -0.3f}, 
        {0.6f, -0.1f, 0.2f, 0.7f, -0.2f, 0.3f}, 
    };
    const char *dec_words[] = { "J'", "aime", "les" };

    /* Weights for three attention layers + FFN */
    float WQ1[DK][DM], WK1[DK][DM];
    float WV1[DK][DM], WO1[DM][DK];
    float WQ2[DK][DM], WK2[DK][DM];
    float WV2[DK][DM], WO2[DM][DK];
    float W1[DF][DM], W2[DM][DF];
    int i, j;

    srand(42);
    float *all_w[] = {
        (float*)WQ1, (float*)WK1, (float*)WV1, 
            (float*)WO1, 
        (float*)WQ2, (float*)WK2, (float*)WV2, 
            (float*)WO2, 
        (float*)W1, (float*)W2
    };
    int sizes[] = { DK*DM, DK*DM, DK*DM, DM*DK, 
                    DK*DM, DK*DM, DK*DM, DM*DK, 
                    DF*DM, DM*DF };
    for (i = 0; i < 10; i++)
        for (j = 0; j < sizes[i]; j++)
            all_w[i][j] = (randf()*2-1)*0.15f;

    printf("Decoder Block: 3 sublayers\n\n");

    /* === Sublayer 1: Causal self-attention === */
    float norm_dec[DL][DM], sub[DL][DM];
    for (i = 0; i < DL; i++)
        rmsnorm(dec[i], norm_dec[i], DM);
    attention(norm_dec, DL, norm_dec, DL, norm_dec, 
              WQ1, WK1, WV1, WO1, 1 /* causal */, sub);
    for (i = 0; i < DL; i++)
        for (j = 0; j < DM; j++) dec[i][j] += sub[i][j];
    printf("  1. Causal self-attention, the decoder\n");
    printf("     attending to itself\n");

    /* === Sublayer 2: Cross-attention === */
    for (i = 0; i < DL; i++)
        rmsnorm(dec[i], norm_dec[i], DM);
    attention(norm_dec, DL, enc, EL, enc, 
              WQ2, WK2, WV2, WO2, 0 /* no mask */, sub);
    for (i = 0; i < DL; i++)
        for (j = 0; j < DM; j++) dec[i][j] += sub[i][j];
    printf("  2. Cross-attention, the decoder\n");
    printf("     attending to the encoder\n");

    /* === Sublayer 3: FFN === */
    for (i = 0; i < DL; i++)
        rmsnorm(dec[i], norm_dec[i], DM);
    for (i = 0; i < DL; i++) {
        float hid[DF], fo[DM];
        int f;
        for (f = 0; f < DF; f++) {
            float z = 0;
            for (j = 0; j < DM; j++)
                z += W1[f][j] * norm_dec[i][j];
            hid[f] = gelu(z);
        }
        for (j = 0; j < DM; j++) {
            float z = 0;
            for (f = 0; f < DF; f++)
                z += W2[j][f] * hid[f];
            dec[i][j] += z;
        }
    }
    printf("  3. Feed-forward network\n\n");

    printf("Output:\n");
    for (i = 0; i < DL; i++) {
        printf("  %-5s [", dec_words[i]);
        for (j = 0; j < DM; j++)
            printf("%+.3f%s", dec[i][j],
                   j<DM-1?",":"");
        printf("]\n");
    }

    printf("\nA decoder block has 3 sublayers where\n");
    printf("an encoder block has 2.\n");
    printf("Cross-attention is the bridge across.\n");

    return 0;
}
