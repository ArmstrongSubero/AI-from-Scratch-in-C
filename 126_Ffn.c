/* 126_Ffn.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* GELU approximation */
static float gelu(float x)
{
    float c = 0.7978846f;
    float u = x + 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * u));
}

#define D_MODEL 6
#define D_FF 12  /* typically 4 * d_model */

typedef struct {
    float W1[D_FF][D_MODEL];   /* up-projection */
    float b1[D_FF];
    float W2[D_MODEL][D_FF];   /* down-projection */
    float b2[D_MODEL];
}
FFN;

static void ffn_init(FFN *f)
{
    int i, j;
    float scale = 0.1f;
    for (i = 0; i < D_FF; i++) {
        f->b1[i] = 0;
        for (j = 0; j < D_MODEL; j++)
            f->W1[i][j] = (randf() * 2 - 1) * scale;
    }
    for (i = 0; i < D_MODEL; i++) {
        f->b2[i] = 0;
        for (j = 0; j < D_FF; j++)
            f->W2[i][j] = (randf() * 2 - 1) * scale;
    }
}

/* FFN forward: up-project, GELU, down-project */
static void ffn_forward(const FFN *f, 
                        const float in[D_MODEL], 
                         float out[D_MODEL])
{
    float hidden[D_FF];
    int i, j;

    /* Up-project: d_model -> d_ff */
    for (i = 0; i < D_FF; i++) {
        float z = f->b1[i];
        for (j = 0; j < D_MODEL; j++)
            z += f->W1[i][j] * in[j];
        hidden[i] = gelu(z);
    }

    /* Down-project: d_ff -> d_model */
    for (i = 0; i < D_MODEL; i++) {
        float z = f->b2[i];
        for (j = 0; j < D_FF; j++)
            z += f->W2[i][j] * hidden[j];
        out[i] = z;
    }
}

int main(void)
{
    FFN f;
    float in[D_MODEL] = { 0.5f, -0.2f, 0.8f, 
                          0.1f, -0.3f, 0.6f };
    float out[D_MODEL];
    int i;

    srand(42);
    ffn_init(&f);
    ffn_forward(&f, in, out);

    printf("Feed-Forward Network, d_model=%d "
           "d_ff=%d\n\n", D_MODEL, D_FF);
    printf("  Input:  [");
    for (i = 0; i < D_MODEL; i++)
        printf("%+.3f%s", in[i],
               i<D_MODEL-1?", ":"");
    printf("]\n  Output: [");
    for (i = 0; i < D_MODEL; i++)
        printf("%+.3f%s", out[i],
               i<D_MODEL-1?", ":"");
    printf("]\n\n");

    int params = D_FF * D_MODEL + D_FF
                 + D_MODEL * D_FF + D_MODEL;
    printf("  Parameters: %d\n", params);
    printf("  Shape: %d -> %d (GELU) -> %d\n",
           D_MODEL, D_FF, D_MODEL);
    printf("  Applied independently "
           "to each position.\n");

    return 0;
}
