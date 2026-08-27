/* 142_Mla.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define DM 64       /* model dimension */
#define N_HEADS 8
#define D_HEAD 8    /* per-head dimension */
/* Compression dimension, far below the 64 that
   N_HEADS * D_HEAD comes to */
#define D_C 16
#define D_ROPE 8    /* decoupled RoPE key dimension */

/* Down-project h to compressed latent c_KV */
static void compress_kv(const float h[DM], 
                         const float W_DKV[D_C][DM], 
                         float c_kv[D_C])
{
    int i, j;
    for (i = 0; i < D_C; i++) {
        c_kv[i] = 0;
        for (j = 0; j < DM; j++)
            c_kv[i] += W_DKV[i][j] * h[j];
    }
}

/* Up-project c_KV to full K or V */
static void decompress(const float c_kv[D_C], 
                        const float
                            W_U[N_HEADS*D_HEAD][D_C], 
                        float full[N_HEADS * D_HEAD])
{
    int i, j;
    for (i = 0; i < N_HEADS * D_HEAD; i++) {
        full[i] = 0;
        for (j = 0; j < D_C; j++)
            full[i] += W_U[i][j] * c_kv[j];
    }
}

int main(void)
{
    float W_DKV[D_C][DM];
    float W_UK[N_HEADS * D_HEAD][D_C];
    float W_UV[N_HEADS * D_HEAD][D_C];
    int i, j;

    srand(42);
    for (i = 0; i < D_C; i++)
        for (j = 0; j < DM; j++)
            W_DKV[i][j] = (randf()*2-1)*0.1f;
    for (i = 0; i < N_HEADS * D_HEAD; i++)
        for (j = 0; j < D_C; j++) {
            W_UK[i][j] = (randf()*2-1)*0.1f;
            W_UV[i][j] = (randf()*2-1)*0.1f;
        }

    /* Input hidden state */
    float h[DM];
    for (i = 0; i < DM; i++) h[i] = (randf()*2-1)*0.5f;

    /* Standard MHA: cache full K and V */
    int standard_cache = N_HEADS * D_HEAD * 2;
    /* K + V */

    /* Compress, cache the latent, and
       decompress only when it is needed */
    float c_kv[D_C];
    compress_kv(h, W_DKV, c_kv);

    int mla_cache = D_C + D_ROPE;  /* c_KV + k_R */

    /* Verify: decompress produces valid K and V */
    float K_full[N_HEADS * D_HEAD];
    float V_full[N_HEADS * D_HEAD];
    decompress(c_kv, W_UK, K_full);
    decompress(c_kv, W_UV, V_full);

    printf("Multi-Head Latent Attention (MLA):\n\n");
    printf("  Standard MHA per token: %d floats, "
           "K and V\n", standard_cache);
    printf("  MLA per token:          %d floats, "
           "c_KV and k_R\n", mla_cache);
    printf("  Compression ratio:            %.1fx\n\n",
           (float)standard_cache / mla_cache);

    printf("  Dimensions:\n");
    printf("    d_model = %d\n", DM);
    printf("    n_heads * d_head = %d, one full "
           "K or V\n", N_HEADS * D_HEAD);
    printf("    d_c = %d (compressed latent)\n", D_C);
    printf("    d_rope = %d, the decoupled key\n\n",
           D_ROPE);

    printf("  Pipeline:\n");
    printf("    Encode:  h [%d] -> c_KV [%d], and "
           "cache that\n", DM, D_C);
    printf("    Decode:  c_KV [%d] -> K [%d]\n",
           D_C, N_HEADS * D_HEAD);
    printf("             c_KV [%d] -> V [%d]\n\n",
           D_C, N_HEADS * D_HEAD);

    /* Real DeepSeek-V3 numbers */
    printf("  DeepSeek-V3 actual dimensions:\n");
    printf("    n_heads=128 d_head=128 d_c=512 "
           "d_rope=64\n");
    int ds_standard = 128 * 128 * 2;  /* full K + V */
    int ds_mla = 512 + 64;            /* c_KV + k_R */
    printf("    Standard cache: %d floats/token\n",
           ds_standard);
    printf("    MLA cache:      "
           "%d floats/token\n", ds_mla);
    printf("    Compression:    %.1fx\n",
           (float)ds_standard / ds_mla);

    return 0;
}
