/* 128_Block.c */
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

#define SEQ_LEN 4
#define D_MODEL 8
#define N_HEADS 2
#define D_HEAD (D_MODEL / N_HEADS)
#define D_FF (D_MODEL * 4)

/* --- RMSNorm --- */
static void rmsnorm(const float *x, float *out, int n)
{
    int i;
    double ss = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float ri = 1.0f / sqrtf((float)(ss / n) + eps);
    for (i = 0; i < n; i++) out[i] = x[i] * ri;
}

/* --- Multi-Head Self-Attention (simplified) --- */
static float dot(const float *a, const float *b, int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void softmax(float *x, int n)
{ float mx = -FLT_MAX, s = 0; int i;
  for (i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
  for (i = 0; i < n; i++) x[i] /= s;
  }

typedef struct {
    float W_Q[N_HEADS][D_HEAD][D_MODEL];
    float W_K[N_HEADS][D_HEAD][D_MODEL];
    float W_V[N_HEADS][D_HEAD][D_MODEL];
    float W_O[D_MODEL][D_MODEL];
}
MHA;

static void mha_forward(const MHA *m, 
            const float X[SEQ_LEN][D_MODEL], 
                          float out[SEQ_LEN][D_MODEL])
{
    float head_out[N_HEADS][SEQ_LEN][D_HEAD];
    float scale = 1.0f / sqrtf((float)D_HEAD);
    int h, i, j, k, d;

    for (h = 0; h < N_HEADS; h++) {
        for (i = 0; i < SEQ_LEN; i++) {
            /* Project query */
            float q[D_HEAD];
            for (k = 0; k < D_HEAD; k++) {
                q[k] = 0;
                for (d = 0; d < D_MODEL; d++)
                    q[k] += m->W_Q[h][k][d] * X[i][d];
            }
            /* Score every key, then blend values */
            float scores[SEQ_LEN];
            for (j = 0; j < SEQ_LEN; j++) {
                float key[D_HEAD];
                for (k = 0; k < D_HEAD; k++) {
                    key[k] = 0;
                    for (d = 0; d < D_MODEL; d++)
                        key[k] += m
                            ->W_K[h][k][d] * X[j][d];
                }
                scores[j] = dot(q, key, D_HEAD) * scale;
            }
            softmax(scores, SEQ_LEN);
            for (k = 0; k < D_HEAD; k++) {
                head_out[h][i][k] = 0;
                for (j = 0; j < SEQ_LEN; j++) {
                    float val = 0;
                    for (d = 0; d < D_MODEL; d++)
                        val += m
                            ->W_V[h][k][d] * X[j][d];
                    head_out[h][i][k]
                            += scores[j] * val;
                }
            }
        }
    }

    /* Concatenate heads and project with W_O */
    for (i = 0; i < SEQ_LEN; i++) {
        float concat[D_MODEL];
        for (h = 0; h < N_HEADS; h++)
            for (k = 0; k < D_HEAD; k++)
                concat[h * D_HEAD + k] =
                    head_out[h][i][k];
        for (j = 0; j < D_MODEL; j++) {
            out[i][j] = 0;
            for (k = 0; k < D_MODEL; k++)
                out[i][j] += m->W_O[j][k] * concat[k];
        }
    }
}

/* --- Feed-Forward Network --- */
typedef struct {
    float W1[D_FF][D_MODEL];
    float W2[D_MODEL][D_FF];
}
FFN;

static void ffn_forward(const FFN *f, 
                        const float in[D_MODEL], 
                          float out[D_MODEL])
{
    float hidden[D_FF];
    int i, j;
    for (i = 0; i < D_FF; i++) {
        float z = 0;
        for (j = 0; j < D_MODEL; j++)
            z += f->W1[i][j] * in[j];
        hidden[i] = gelu(z);
    }
    for (i = 0; i < D_MODEL; i++) {
        float z = 0;
        for (j = 0; j < D_FF; j++)
            z += f->W2[i][j] * hidden[j];
        out[i] = z;
    }
}

/* --- Transformer Block --- */
typedef struct {
    MHA attn;
    FFN ffn;
}
TransformerBlock;

static void block_forward(const TransformerBlock *blk, 
                            float X[SEQ_LEN][D_MODEL])
{
    float normed[SEQ_LEN][D_MODEL];
    float sublayer[SEQ_LEN][D_MODEL];
    int i, j;

    /* === Sublayer 1: Attention === */
    /* Pre-norm */
    for (i = 0; i < SEQ_LEN; i++)
        rmsnorm(X[i], normed[i], D_MODEL);

    /* Multi-head self-attention */
    mha_forward(&blk->attn, normed, sublayer);

    /* Residual add */
    for (i = 0; i < SEQ_LEN; i++)
        for (j = 0; j < D_MODEL; j++)
            X[i][j] += sublayer[i][j];

    /* === Sublayer 2: FFN === */
    /* Pre-norm */
    for (i = 0; i < SEQ_LEN; i++)
        rmsnorm(X[i], normed[i], D_MODEL);

    /* Feed-forward (per position) */
    for (i = 0; i < SEQ_LEN; i++) {
        float ffn_out[D_MODEL];
        ffn_forward(&blk->ffn, normed[i], ffn_out);
        /* Residual add */
        for (j = 0; j < D_MODEL; j++)
            X[i][j] += ffn_out[j];
    }
}

static void init_random(float *w, int n, float scale)
{
    int i;
    for (i = 0; i < n; i++)
        w[i] = (randf() * 2 - 1) * scale;
}

int main(void)
{
    TransformerBlock blk;
    float X[SEQ_LEN][D_MODEL] = {
        { 1.0f, 0.2f, -0.3f, 0.5f, 
          -0.1f, 0.8f, 0.3f, -0.2f }, 
        { 0.3f, 0.9f, 0.1f, -0.2f, 
          0.6f, 0.4f, -0.1f, 0.5f }, 
        { -0.1f, 0.4f, 0.8f, 0.2f, 
          -0.5f, 0.3f, 0.7f, 0.1f }, 
        { 0.6f, -0.2f, 0.5f, 0.7f, 
          0.3f, -0.4f, 0.2f, 0.8f }, 
    };
    const char *words[] = { "The", "cat",
        "sat", "down" };
    int i, j;

    srand(42);
    init_random((float*)&blk.attn, 
                sizeof(MHA)/sizeof(float), 0.1f);
    init_random((float*)&blk.ffn, 
                sizeof(FFN)/sizeof(float), 0.1f);

    printf("Transformer block, d_model=%d "
           "heads=%d d_ff=%d\n\n",
           D_MODEL, N_HEADS, D_FF);

    printf("Input:\n");
    for (i = 0; i < SEQ_LEN; i++) {
        printf("  %-5s [", words[i]);
        for (j = 0; j < D_MODEL; j++)
            printf("%+.2f%s", X[i][j],
                j<D_MODEL-1?",":"");
        printf("]\n");
    }

    block_forward(&blk, X);

    printf("\nOutput (after 1 transformer block):\n");
    for (i = 0; i < SEQ_LEN; i++) {
        printf("  %-5s [", words[i]);
        for (j = 0; j < D_MODEL; j++)
            printf("%+.2f%s", X[i][j],
                j<D_MODEL-1?",":"");
        printf("]\n");
    }

    printf("\nOutput shape equals input shape.\n");
    printf("This is critical: blocks "
           "can be stacked.\n");

    /* Parameter count */
    /* W_Q, W_K, W_V and W_O */
    int attn_params = 4 * D_MODEL * D_MODEL;
    /* W1, W2 */
    int ffn_params = 2 * D_MODEL * D_FF;
    /* two RMSNorm gamma vectors */
    int norm_params = 2 * D_MODEL;
    int total = attn_params + ffn_params + norm_params;

    printf("\nParameter count per block:\n");
    printf("  Attention: 4 * %d^2 = %d\n",
           D_MODEL, attn_params);
    printf("  FFN:       2 * %d * %d = %d\n",
           D_MODEL, D_FF, ffn_params);
    printf("  Norms:     2 * %d = %d\n",
           D_MODEL, norm_params);
    printf("  Total:     %d\n", total);

    return 0;
}
