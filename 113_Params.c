/* 113_Params.c */
#include <stdio.h>

int main(void)
{
    int d_model = 512;
    int heads[] = { 1, 2, 4, 8, 16 };
    int n = 5;
    int i;

    printf("Multi-head attention parameters, "
           "d_model=%d\n\n", d_model);
    printf("  heads  d_head  W_Q+W_K+W_V      W_O    "
           "      total\n");
    printf("  -----  ------  ---------------  -------"
           "----  --------\n");

    for (i = 0; i < n; i++) {
        int h = heads[i];
        int d_head = d_model / h;

        /* Per head W_Q is [d_head x d_model],
           and the same for W_K and W_V */
        /* Across h heads that comes to
           h * 3 * d_head * d_model, 
           which is 3 * d_model * d_model */
        int qkv_params = h * 3 * d_head * d_model;

        /* W_O: [d_model x d_model] (always the same) */
        int wo_params = d_model * d_model;

        int total = qkv_params + wo_params;

        printf("  %3d    %4d    %10d       %8d     "
               "%8d\n",
               h, d_head, qkv_params, wo_params, total);
    }

    printf("\nThe total is ALWAYS 4 * d_model^2 "
           "= %d.\n", 4 * d_model * d_model);
    printf("The head count does not change it.\n\n");

    printf("Each head has smaller projections, "
           "since d_head = d_model/h.\n");
    printf("More heads with smaller projections "
           "costs the same total\n");
    printf("as fewer heads with "
           "larger projections.\n\n");

    printf("What changes is the diversity of "
           "attention patterns.\n");
    printf("8 heads can learn 8 relationship "
           "types, 1 head learns 1.\n");

    return 0;
}
