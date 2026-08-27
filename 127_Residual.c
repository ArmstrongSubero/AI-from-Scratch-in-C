/* 127_Residual.c */
#include <stdio.h>

#define DIM 4

static void residual_add(const float in[DIM], 
                         const float sublayer_out[DIM], 
                          float out[DIM])
{
    int i;
    for (i = 0; i < DIM; i++)
        out[i] = in[i] + sublayer_out[i];
}

int main(void)
{
    float x[DIM] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float sub[DIM] = { 0.1f, -0.2f, 0.3f, -0.1f };
    float out[DIM];
    int i;

    residual_add(x, sub, out);

    printf("Residual, out = x + sublayer(x)\n\n");
    printf("  Input:     [");
    for (i = 0; i < DIM; i++)
        printf("%.1f%s", x[i], i<DIM-1?", ":"");
    printf("]\n  Sublayer:  [");
    for (i = 0; i < DIM; i++)
        printf("%+.1f%s", sub[i], i<DIM-1?", ":"");
    printf("]\n  Output:    [");
    for (i = 0; i < DIM; i++)
        printf("%.1f%s", out[i], i<DIM-1?", ":"");
    printf("]\n\n");

    printf("  In backprop, d_out/d_x is\n");
    printf("  1 + d_sublayer/d_x\n");
    printf("  Even when d_sublayer/d_x is small,\n");
    printf("  the gradient through the identity\n");
    printf("  path is always 1.\n");
    printf("  Deep transformers train "
           "because of it.\n");

    return 0;
}
