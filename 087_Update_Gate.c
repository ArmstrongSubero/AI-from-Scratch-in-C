/* 087_Update_Gate.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

int main(void)
{
    /* Demonstrate the interpolation
       h = z * h_old + (1-z) * s */
    float h_old = 0.8f;    /* existing state */
    float s     = -0.3f;   /* new candidate */

    float z_values[] = { 0.0f, 0.2f, 0.5f, 0.8f, 1.0f };
    int n = 5;
    int i;

    printf("Update gate interpolation: "
           "h = z * h_old + (1-z) * s\n");
    printf("  h_old = %.1f, candidate s = %.1f\n\n",
           h_old, s);
    printf("  z      h_new    interpretation\n");
    printf("  -----  ------   ------------------------"
           "----------------\n");

    for (i = 0; i < n; i++) {
        float z = z_values[i];
        float h_new = z * h_old + (1.0f - z) * s;
        printf("  %.1f    %+.3f   ", z, h_new);

        if (z > 0.9f) 
            printf("keep old state (no update)");
        else if (z < 0.1f)
            printf("replace with candidate "
                   "(full update)");
        else if (z > 0.6f)
            printf("mostly keep, small update");
        else if (z < 0.4f)
            printf("mostly replace, small retention");
        else printf("equal blend");
        printf("\n");
    }

    printf("\nCompare to LSTM. The forget gate and "
           "input gate are\n");
    printf("independent. f can be 0.9 and i can be "
           "0.9 at once,\n");
    printf("meaning the LSTM can both keep the old "
           "AND add the new.\n");
    printf("The GRU couples them, so keeping more "
           "old means adding less new.\n");

    return 0;
}
