/* 177_Deepseek.c */
#include <stdio.h>

#define LAYERS 61
/* first layers keep a dense FFN */
#define DENSE 3
#define D_MODEL 7168
#define D_FFN     18432  /* dense layer intermediate */
#define D_EXPERT  2048   /* MoE expert intermediate */
#define N_ROUTED 256
#define N_SHARED 1
#define TOP_K 8
#define VOCAB 129280

/* SwiGLU FFN carries gate, up and down projections */
static long long ffn_params(int d_model, int d_hidden)
{
    return 3LL * d_model * d_hidden;
}

/* MLA projections for one layer */
static long long mla_params(void)
{
    long long q_a = 7168LL * 1536;
    long long q_b = 1536LL * 128 * (128 + 64);
    long long kv_a = 7168LL * (512 + 64);
    long long kv_b = 512LL * 128 * (128 + 128);
    long long o = 128LL * 128 * 7168;
    return q_a + q_b + kv_a + kv_b + o;
}

int main(void)
{
    int moe_layers = LAYERS - DENSE;

    long long per_expert =
        ffn_params(D_MODEL, D_EXPERT);
    long long per_dense = ffn_params(D_MODEL, D_FFN);
    long long per_moe = per_expert
                         * (N_ROUTED + N_SHARED);

    long long ffn_all = DENSE * per_dense
                      + (long long)moe_layers * per_moe;
    long long ffn_act = DENSE * per_dense
            + (long long)moe_layers * per_expert
                        * (TOP_K + N_SHARED);

    long long attn = mla_params() * LAYERS;
    long long embed = 2LL * VOCAB * D_MODEL;

    long long all = ffn_all + attn + embed;
    long long act = ffn_act + attn
                  + (long long)VOCAB * D_MODEL;

    printf("DeepSeek-V3 parameter budget:\n\n");
    printf("  Per routed expert:  %lld (%.1fM)\n",
           per_expert, per_expert / 1e6);
    printf("  Per MoE layer:      %.2fB (1 + %d "
           "experts)\n",
           per_moe / 1e9, N_ROUTED);
    printf("  Per dense FFN:      %.2fB\n",
           per_dense / 1e9);
    printf("  MLA per layer:      %.1fM\n",
           mla_params() / 1e6);

    printf("\n  Component      Total       Active\n");
    printf("  ---------      -----       ------\n");
    printf("  FFN            %6.1fB     %6.1fB\n",
           ffn_all / 1e9, ffn_act / 1e9);
    printf("  Attention      %6.1fB     %6.1fB\n",
           attn / 1e9, attn / 1e9);
    printf("  Embeddings     %6.1fB     %6.1fB\n",
           embed / 1e9, VOCAB * (double)D_MODEL / 1e9);
    printf("  ---------      -----       ------\n");
    printf("  Model          %6.1fB     %6.1fB\n",
           all / 1e9, act / 1e9);

    printf("\n  Published:     671.0B      37.0B\n");
    printf("  Error:         %5.1f%%       %5.1f%%\n",
           (all / 1e9 - 671.0) / 671.0 * 100, 
           (act / 1e9 - 37.0) / 37.0 * 100);
    printf("  Ratio: %.1fx capacity per unit of "
           "compute\n",
           (double)all / act);

    printf("\n  The FFN holds %.0f%% of the weights "
           "and\n",
           100.0 * ffn_all / all);
    printf("  contributes %.0f%% of the active ones.\n",
           100.0 * ffn_act / act);

    return 0;
}
