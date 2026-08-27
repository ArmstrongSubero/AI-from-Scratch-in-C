/* 158_Chinchilla.c */
#include <stdio.h>

int main(void)
{
    printf("Chinchilla optimal, about 20 tokens "
           "per parameter\n\n");

    struct {
        const char *name;
        double params;
        double tokens;
        double ratio;
    }
    models[] = {
        { "GPT-3",        175e9,   300e9,
            300e9/175e9 }, 
        { "Chinchilla",   70e9,    1400e9,
            1400e9/70e9 }, 
        { "LLaMA-7B",     7e9,     1000e9,
            1000e9/7e9 }, 
        { "LLaMA-70B",    70e9,    2000e9,
            2000e9/70e9 }, 
        { "DeepSeek-V3",  671e9,   14800e9,
            14800e9/671e9 }, 
    };
    int n = 5, i;

    printf("  %-12s %7s %9s %8s  %s\n", "Model",
           "Params", "Tokens", "Tok/Par", "Status");
    printf("  %-12s %7s %9s %8s  %s\n", "",
           "------", "------", "-------", "------");

    for (i = 0; i < n; i++) {
        printf("  %-12s %6.0fB %8.0fB %8.1f  %s\n",
               models[i].name, 
               models[i].params / 1e9, 
               models[i].tokens / 1e9, 
               models[i].ratio, 
               models[i].ratio < 10 ? "UNDERTRAINED" :
               models[i].ratio > 50 ? "DATA-RICH"
                                    : "BALANCED");
    }

    printf("\n  GPT-3 was undertrained at only 1.7 "
           "tokens\n  per parameter.\n");
    printf("  Chinchilla showed a 4x smaller model\n");
    printf("  on 4x more data beats it.\n\n");

    printf("  Modern models go far past that ratio.\n");
    printf("  LLaMA-7B uses 143 tokens per param,\n");
    printf("  seven times Chinchilla optimal.\n");
    printf("  DeepSeek-V3 uses 22, which is close "
           "to\n");
    printf("  optimal, but it is MoE with 37B "
           "active,\n");
    printf("  so the effective ratio is about 400\n");
    printf("  tokens per active parameter.\n");

    return 0;
}
