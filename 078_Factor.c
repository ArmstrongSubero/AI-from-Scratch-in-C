/* 078_Factor.c */
#include <stdio.h>
#include <math.h>

int main(void)
{
    /* Simulate the per-step shrinkage factor */
    /* tanh derivative at various hidden state values */
    float h_values[] = { 0.0f, 0.3f, 0.5f, 0.7f, 
                         0.9f, 0.95f, 0.99f };
    int n = 7;
    int i;

    printf("Tanh derivative (1 - h^2) at various "
           "hidden values:\n\n");
    printf("  h        tanh'    after 10   "
           "after 20   after 50\n");
    printf("  -------  ------   --------   "
           "--------   --------\n");

    for (i = 0; i < n; i++) {
        float h = h_values[i];
        float d = 1.0f - h * h;
        double d10 = pow(d, 10);
        double d20 = pow(d, 20);
        double d50 = pow(d, 50);
        printf("  %5.2f    %6.4f"
               "   %.2e   %.2e   %.2e\n",
               h, d, d10, d20, d50);
    }

    printf("\nWhen h is near 0, the derivative is 1 "
           "and gradients survive.\n");
    printf("When h is 0.9 (common after training), "
           "the derivative is 0.19.\n");
    printf("After 20 steps of 0.19: %.2e "
           "(effectively zero).\n",
           pow(0.19, 20));
    printf("\nThis is why basic RNNs cannot learn "
           "long-range dependencies.\n");
    printf("The gradient vanishes exponentially "
           "with sequence length.\n");

    return 0;
}
