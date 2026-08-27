/* 140_Kv_Cache.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}
static float dot(const float *a, const float *b, int n)
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

#define DM 6
#define DK 4
#define MAX_SEQ 20

typedef struct {
    float K[MAX_SEQ][DK];   /* cached keys */
    float V[MAX_SEQ][DK];   /* cached values */
    /* number of cached entries */
    int len;
}
KVCache;

typedef struct {
    float WQ[DK][DM];
    float WK[DK][DM];
    float WV[DK][DM];
    float WO[DM][DK];
}
AttnWeights;

/* Process ONE new token using the cache */
static void attention_with_cache(
    const AttnWeights *w, 
    /* the new token's representation */
    const float x_new[DM], 
    KVCache *cache, 
    float out[DM])
{
    int i, j, k;
    float scale = 1.0f / sqrtf((float)DK);

    /* Project new token to Q, K, V */
    float q[DK], k_new[DK], v_new[DK];
    for (i = 0; i < DK; i++) {
        q[i] = 0;
        k_new[i] = 0;
        v_new[i] = 0;
        for (j = 0; j < DM; j++) {
            q[i] += w->WQ[i][j] * x_new[j];
            k_new[i] += w->WK[i][j] * x_new[j];
            v_new[i] += w->WV[i][j] * x_new[j];
        }
    }

    /* Append new K, V to cache */
    int pos = cache->len;
    for (i = 0; i < DK; i++) {
        cache->K[pos][i] = k_new[i];
        cache->V[pos][i] = v_new[i];
    }
    cache->len++;

    /* Compute attention: new Q against ALL cached K */
    float scores[MAX_SEQ];
    for (j = 0; j < cache->len; j++)
        scores[j] = dot(q, cache->K[j], DK) * scale;

    /* The mask is automatic here, since the
       cache holds only the past and the present */
    softmax(scores, cache->len);

    /* Weighted sum of cached V */
    float head[DK] = {0};
    for (j = 0; j < cache->len; j++)
        for (k = 0; k < DK; k++)
            head[k] += scores[j] * cache->V[j][k];

    /* Output projection */
    for (i = 0; i < DM; i++) {
        out[i] = 0;
        for (k = 0; k < DK; k++)
            out[i] += w->WO[i][k] * head[k];
    }
}

int main(void)
{
    AttnWeights w;
    KVCache cache = { .len = 0 };
    int i, j;

    srand(42);
    for (i = 0; i < DK; i++)
        for (j = 0; j < DM; j++) {
            w.WQ[i][j] = (randf()*2-1)*0.2f;
            w.WK[i][j] = (randf()*2-1)*0.2f;
            w.WV[i][j] = (randf()*2-1)*0.2f;
        }
    for (i = 0; i < DM; i++)
        for (j = 0; j < DK; j++)
            w.WO[i][j] = (randf()*2-1)*0.2f;

    /* Simulate processing 6 tokens one at a time */
    float tokens[6][DM] = {
        { 0.5f, -0.2f, 0.8f, 0.1f, -0.3f, 0.6f }, 
        { 0.3f, 0.7f, 0.1f, -0.2f, 0.5f, 0.4f }, 
        { -0.1f, 0.4f, 0.6f, 0.3f, -0.4f, 0.2f }, 
        { 0.2f, -0.3f, 0.5f, 0.6f, 0.1f, -0.5f }, 
        { 0.8f, 0.2f, -0.1f, 0.4f, 0.3f, 0.6f }, 
        { -0.4f, 0.1f, 0.3f, -0.2f, 0.7f, 0.5f }, 
    };
    const char *words[] = { "The", "cat", "sat",
                            "on", "the", "mat" };

    printf("Attention with a KV cache, one token "
           "at a time\n\n");
    printf("  Token    Cache size   K/V projections "
           "computed\n");
    printf("  ------   ----------   ----------------"
           "-------\n");

    for (int t = 0; t < 6; t++) {
        float out[DM];
        int prev_len = cache.len;
        attention_with_cache(&w, tokens[t], 
            &cache, out);

        printf("  %-6s   %3d -> %3d   1 (new token "
               "only)\n",
               words[t], prev_len, cache.len);
    }

    printf("\n  Without cache: 1+2+3+4+5+6 = 21 "
           "projections\n");
    printf("  With cache:    6 projections, one "
           "per token\n");
    printf("  Cache memory:  %d entries x %d dims "
           "x 2 = %d floats\n",
           cache.len, DK, cache.len * DK * 2);

    return 0;
}
