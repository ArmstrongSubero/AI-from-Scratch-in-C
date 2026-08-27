/* 141_Memory.c */
#include <stdio.h>

int main(void)
{
    printf("KV cache memory for real models:\n\n");

    struct {
        const char *name;
        int n_layers, n_heads, d_head, max_seq;
    }
    models[] = {
        { "GPT-2 Small",   12,  12,  64,   1024 },
        { "LLaMA-7B",      32,  32,  128,  4096 },
        { "LLaMA-70B",     80,  64,  128,  4096 },
        { "DeepSeek-V3",   61,  128, 128,  128000 },
    };
    int n = 4, i;

    printf("  %-14s  layers  heads  d_head  max_seq "
           " cache/token   full cache\n", "Model");
    printf("  %-14s  ------  -----  ------  ------- "
           " -----------   ----------\n", "");

    for (i = 0; i < n; i++) {
        int L = models[i].n_layers;
        int H = models[i].n_heads;
        int D = models[i].d_head;
        int S = models[i].max_seq;

        /* Per token, 2 for K and V, times
           layers times heads times d_head
           times sizeof(float) */
        long long per_token = 2LL * L * H * D * 4;
        long long full = per_token * S;

        printf("  %-14s  %4d    %3d    %4d    %6d  "
               " %7lld B     ",
               models[i].name, L, H, D, S, per_token);

        if (full < 1024LL*1024*1024)
            printf("%.0f MB\n", full / (1024.0 * 1024));
        else
            printf("%.1f GB\n",
                   full / (1024.0 * 1024 * 1024));
    }

    printf("\n  DeepSeek-V3 at 128K tokens would "
           "need this much\n");
    printf("  without MLA. That is why MLA exists.\n");

    return 0;
}
