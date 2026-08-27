/* 159_Compute.c */
#include <stdio.h>

int main(void)
{
    printf("Training compute, C = 6 * N * D\n\n");

    struct {
        const char *name;
        /* total, what gets stored */
        double params;
        double active;      /* what runs per token */
        double tokens;
        double gpu_hours;
        double cost_per_hour;
    }
    models[] = {
        { "GPT-2 Small", 124e6, 124e6, 100e9,
          120, 2.0 }, 
        { "LLaMA-7B",    7e9,   7e9,   1000e9,
          82000, 2.0 }, 
        { "LLaMA-70B",   70e9,  70e9,  2000e9,
          1720000, 2.0 }, 
        /* MoE, so only 37B of the 671B run per token
           and only those cost anything to train */
        { "DeepSeek-V3", 671e9, 37e9,  14800e9,
          2788000, 2.0 }, 
    };
    int n = 4, i;

    printf("  %-12s %6s %6s %7s %9s %8s %6s\n",
           "Model", "Total", "Active", "Tokens",
           "FLOPs", "GPU-hrs", "Cost");
    printf("  %-12s %6s %6s %7s %9s %8s %6s\n", "",
           "-----", "------", "------", "-----",
           "-------", "----");

    for (i = 0; i < n; i++) {
        double flops = 
            6.0 * models[i].active * models[i].tokens;
        double pflops = flops / 1e15;

        printf("  %-12s %5.1fB %5.1fB %6.0fB %9.1e ",
               models[i].name, 
               models[i].params / 1e9, 
               models[i].active / 1e9, 
               models[i].tokens / 1e9, 
               pflops);

        double cost = models[i].gpu_hours
                      * models[i].cost_per_hour;
        if (models[i].gpu_hours >= 1000)
            printf("%7.0fK $%5.1fM",
                   models[i].gpu_hours
                       / 1000, cost / 1e6);
        else
            printf("%8.0f $%5.1fK",
                   models[i].gpu_hours, cost / 1e3);
        printf("\n");
    }

    printf("\n  The FLOPs column uses active "
           "params,\n");
    printf("  not total. A dense model has one "
           "number\n");
    printf("  for both, and an MoE does not, since "
           "a\n");
    printf("  token only touches the experts it is\n");
    printf("  routed to. Using 671B here would give\n");
    printf("  6.0e10 PF, which against 2788K "
           "GPU-hours\n");
    printf("  works out at 5.9 PFLOP/s per GPU, "
           "well\n");
    printf("  past what the hardware can reach.\n\n");

    printf("  DeepSeek-V3: $5.6M total training "
           "cost.\n");
    printf("  GPT-4 is estimated at $100M or more.\n");
    printf("  The gap comes from architecture, "
           "since\n");
    printf("  MoE activates 37B of 671B, and from\n");
    printf("  engineering, FP8 and DualPipe.\n");

    return 0;
}
