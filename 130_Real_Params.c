/* 130_Real_Params.c */
#include <stdio.h>

/* Print a count in a readable unit */
static void human(long long v)
{
    if (v >= 1000000000LL)
        printf("%6.1fB", v / 1e9);
    else if (v >= 1000000LL)
        printf("%6.1fM", v / 1e6);
    else
        printf("%6lldK", v / 1000);
}

int main(void)
{
    struct {
        const char *name;
        int d_model, n_heads, d_ff, n_layers;
        long long published;
    } models[] = {
        { "GPT-2 Small",  768,  12,  3072, 12,
          124000000LL },
        { "GPT-2 Medium", 1024, 16,  4096, 24,
          355000000LL },
        { "GPT-2 Large",  1280, 20,  5120, 36,
          774000000LL },
        { "LLaMA-7B",     4096, 32, 11008, 32,
          6700000000LL },
        { "DeepSeek-V3",  7168, 128, 18432, 61,
          671000000000LL },
    };
    int n = 5, i;

    printf("Transformer block parameters in real "
           "models\n\n");
    printf("  model          d_model  d_ff   L  ");
    printf("per block  all blocks  published\n");
    printf("  -------------  -------  -----  --  ");
    printf("-------  ---------  ---------\n");

    for (i = 0; i < n; i++) {
        int dm = models[i].d_model;
        int df = models[i].d_ff;
        long long attn  = 4LL * dm * dm;
        long long ffn   = 2LL * dm * df;
        long long norms = 2LL * dm;
        long long per   = attn + ffn + norms;
        long long all   = per * models[i].n_layers;

        printf("  %-13s  %7d  %5d  %2d  ",
               models[i].name, dm, df,
               models[i].n_layers);
        human(per);   printf("  ");
        human(all);   printf("  ");
        human(models[i].published);
        printf("\n");
    }

    printf("\n  The last two columns disagree, and the\n");
    printf("  gap is what the block accounting leaves\n");
    printf("  out. Token embeddings, the output head\n");
    printf("  and the final norm all sit outside the\n");
    printf("  blocks. For GPT-2 Small the embedding\n");
    printf("  table alone is 50257 x 768, about 39M.\n\n");

    printf("  DeepSeek-V3 is the extreme case. Its FFN\n");
    printf("  is a mixture of experts, so the real\n");
    printf("  count is far above a dense estimate, and\n");
    printf("  only 37B of the 671B are active per\n");
    printf("  token. Chapter 36 builds that.\n\n");

    printf("  Within a block, attention costs 4*d^2\n");
    printf("  and the FFN costs 2*d*d_ff, so the FFN\n");
    printf("  dominates whenever d_ff exceeds 2*d.\n");

    return 0;
}
