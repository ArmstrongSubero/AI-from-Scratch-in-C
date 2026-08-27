/* 146_Return.c */
#include <stdio.h>
#include <math.h>

int main(void)
{
    /* A sequence of rewards */
    /* One optimal run of the Step 1 environment,
       three moves at -1 then the goal move at +10 */
    float rewards[] = { -1, -1, -1, 10 };
    int n = (int)(sizeof(rewards) / sizeof(rewards[0]));
    int i, j;

    printf("Discounted return computation:\n\n");
    printf("  Rewards: [");
    for (i = 0; i < n; i++)
        printf("%.0f%s", rewards[i],
               i<n-1?", ":"");
    printf("]\n\n");

    float gammas[] = { 1.0f, 0.99f, 0.9f, 0.5f };
    int n_gamma = 4;

    printf("  gamma   G_0 (return from step 0)\n");
    printf("  ------  -----------------------\n");

    for (j = 0; j < n_gamma; j++) {
        float gamma = gammas[j];
        float G = 0;
        for (i = n - 1; i >= 0; i--)
            G = rewards[i] + gamma * G;
        printf("  %.2f    %+.2f\n", gamma, G);
    }

    printf("\n  gamma=1.0 does not discount at all\n");
    printf("  gamma=0.9 makes the +10 at step 3 "
           "worth 10*0.9^3 = %.2f\n",
           10 * powf(0.9f, 3));
    printf("  gamma=0.5 makes it worth "
           "10*0.5^3 = %.2f\n",
           10 * powf(0.5f, 3));
    printf("\n  Lower gamma = more short-sighted "
           "agent.\n");
    printf("  Higher gamma = plans further ahead.\n");

    return 0;
}
