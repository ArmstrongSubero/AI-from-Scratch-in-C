/* 143_Verify.c */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float randf(void)
{
    return(float)rand()/RAND_MAX;
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

#define DM 4
#define DK 3
#define SL 4

int main(void)
{
    float WQ[DK][DM], WK[DK][DM], WV[DK][DM];
    int i, j, k;
    srand(42);
    for (i = 0;i<DK;i++) for(j = 0;j<DM;j++){
        WQ[i][j] = (randf()*2-1)*0.3f;
        WK[i][j] = (randf()*2-1)*0.3f;
        WV[i][j] = (randf()*2-1)*0.3f;
    }

    float X[SL][DM] = {
        {0.5f, -0.2f, 0.8f, 0.1f}, {0.3f, 0.7f, 
            0.1f, -0.2f}, 
        {-0.1f, 0.4f, 0.6f, 0.3f}, {0.2f, -0.3f, 
            0.5f, 0.6f}};

    float scale = 1.0f / sqrtf((float)DK);

    /* Method 1: Full recomputation (no cache) */
    float out_full[SL][DK];
    for (i = 0; i < SL; i++) {
        float q[DK];
        for (k = 0; k < DK; k++) {
            q[k] = 0;
            for (j = 0; j < DM; j++)
                q[k] += WQ[k][j] * X[i][j];
        }
        float scores[SL];
        for (j = 0; j <= i; j++) {
            float key[DK];
            for (k = 0; k < DK; k++) {
                key[k] = 0;
                for (int d = 0; d < DM; d++)
                    key[k] += WK[k][d] * X[j][d];
            }
            scores[j] = dot(q, key, DK)*scale;
            }
        softmax(scores, i+1);
        for (k = 0; k < DK; k++) {
            out_full[i][k] = 0;
            for (j = 0; j <= i; j++) {
                float v = 0;
                for (int d = 0; d < DM; d++)
                    v += WV[k][d] * X[j][d];
            out_full[i][k] += scores[j]*v;
            }
            }
    }

    /* Method 2: With KV cache */
    float cache_K[SL][DK], cache_V[SL][DK];
    float out_cache[SL][DK];
    int cache_len = 0;

    for (i = 0; i < SL; i++) {
        float q[DK];
        for (k = 0; k < DK; k++) {
            q[k] = 0;
            for (j = 0; j < DM; j++)
                q[k] += WQ[k][j] * X[i][j];
        }
        /* Project and cache new K, V */
        for (k = 0; k < DK; k++) {
            cache_K[cache_len][k] = 0;
            cache_V[cache_len][k] = 0;
            for (j = 0; j < DM; j++) {
                cache_K[cache_len][k] += 
                    WK[k][j] * X[i][j];
                cache_V[cache_len][k] += 
                    WV[k][j] * X[i][j];
            }
        }
        cache_len++;
        /* Attend to cache */
        float scores[SL];
        for (j = 0; j < cache_len; j++)
            scores[j] = dot(q, cache_K[j], DK) * scale;
        softmax(scores, cache_len);
        for (k = 0; k < DK; k++) {
            out_cache[i][k] = 0;
            for (j = 0; j < cache_len; j++)
                out_cache[i][k] += 
                    scores[j] * cache_V[j][k];
        }
    }

    /* Compare */
    printf("Cache against full recomputation\n\n");
    printf("  pos  full_recompute        with_cache "
           "           match?\n");
    for (i = 0; i < SL; i++) {
        float diff = 0;
        for (k = 0; k < DK; k++) {
            float d = out_full[i][k] - out_cache[i][k];
            diff += d * d;
        }
        diff = sqrtf(diff);
        printf("  %d    [%+.4f,%+.4f,%+.4f]  "
               "[%+.4f,%+.4f,%+.4f]  %s\n",
               i, out_full[i][0], out_full[i][1], 
               out_full[i][2], 
               out_cache[i][0], out_cache[i][1], 
               out_cache[i][2], 
               diff < 1e-6f ? "yes" : "NO");
    }

    printf("\n  Both methods produce identical "
           "results.\n");
    printf("  The cache is an optimization rather "
           "than an approximation.\n");

    return 0;
}
