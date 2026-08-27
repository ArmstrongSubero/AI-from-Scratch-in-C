/* 168_Savings.c */
#include <stdio.h>

int main(void)
{
    printf("LoRA parameter savings for real "
           "models:\n\n");

    struct {
        const char *name;
        int d_model;
        int n_layers;
        int n_matrices;
        /* W_Q, W_K, W_V, W_O per layer */
    }
    models[] = {
        { "LLaMA-7B",    4096,  32, 4 },
        { "LLaMA-70B",   8192,  80, 4 },
        { "DeepSeek-V3", 7168,  61, 4 },
    };
    int n = 3, i;
    int ranks[] = { 4, 8, 16, 32 };
    int n_ranks = 4;

    for (i = 0; i < n; i++) {
        int d = models[i].d_model;
        int L = models[i].n_layers;
        int M = models[i].n_matrices;
        long long full = (long long)d * d * L * M;

        printf("  %s, d=%d, %d layers, %d attention "
               "matrices\n",
               models[i].name, d, L, M);
        printf("    Full fine-tune: %lld params "
               "(%.1fM)\n", full, full/1e6);

        for (int r = 0; r < n_ranks; r++) {
            int rank = ranks[r];
            long long lora = 
            (long long)(d * rank + rank * d)
                * L * M;
            printf("    LoRA rank=%2d: %lld params "
                   "(%.1fM, %.0fx less)\n",
                   rank, lora, lora/1e6, 
                   (float)full / lora);
        }
        printf("\n");
    }

    printf("  At rank 16, LLaMA-7B trains 16.8M "
           "rather\n");
    printf("  than 2.1B, which is the 128x in the "
           "table\n");
    printf("  above and fits on one consumer "
           "card.\n\n");

    printf("  The frozen weights still have to be "
           "held\n");
    printf("  in memory, and that is the real floor "
           "on\n");
    printf("  what LoRA can do. What it removes is "
           "the\n");
    printf("  gradient and the optimizer state, "
           "which\n");
    printf("  for Adam is two more copies of every\n");
    printf("  parameter being trained.\n");

    return 0;
}
