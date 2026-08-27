/* 165_Lora_Forward.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define D_IN 6
#define D_OUT 6
#define RANK 2

typedef struct {
    float W[D_OUT][D_IN];
    /* frozen pretrained weights */
    /* trainable: down-project */
    float A[RANK][D_IN];
    float B[D_OUT][RANK];    /* trainable: up-project */
    float alpha;              /* scaling factor */
}
LoRALayer;

static void lora_init(LoRALayer *l)
{
    int i, j;
    /* Pretrained weights (frozen, random for demo) */
    for (i = 0; i < D_OUT; i++)
        for (j = 0; j < D_IN; j++)
            l->W[i][j] = (randf()*2-1) * 0.3f;

    /* A gets small random values, B gets zeros */
    for (i = 0; i < RANK; i++)
        for (j = 0; j < D_IN; j++)
            l->A[i][j] = (randf()*2-1) * 0.01f;

    /* B is zero at the start, so the LoRA path
       contributes nothing and the model behaves
       exactly like the pretrained one. */
    for (i = 0; i < D_OUT; i++)
        for (j = 0; j < RANK; j++)
            l->B[i][j] = 0.0f;

    l->alpha = 1.0f;
}

static void lora_forward(const LoRALayer *l, 
                         const float x[D_IN], 
                           float out[D_OUT])
{
    int i, j;
    float scale = l->alpha / RANK;

    /* Frozen path: W * x */
    for (i = 0; i < D_OUT; i++) {
        out[i] = 0;
        for (j = 0; j < D_IN; j++)
            out[i] += l->W[i][j] * x[j];
    }

    /* LoRA path: (alpha/r) * B * A * x */
    float mid[RANK];
    for (i = 0; i < RANK; i++) {
        mid[i] = 0;
        for (j = 0; j < D_IN; j++)
            mid[i] += l->A[i][j] * x[j];
    }
    for (i = 0; i < D_OUT; i++)
        for (j = 0; j < RANK; j++)
            out[i] += scale * l->B[i][j] * mid[j];
}

int main(void)
{
    LoRALayer l;
    srand(42);
    lora_init(&l);

    float x[D_IN] = { 0.5f, -0.2f, 0.8f, 
                      0.1f, -0.3f, 0.6f };
    float out_before[D_OUT], out_after[D_OUT];
    int i;

    /* Before training, with B still zero */
    lora_forward(&l, x, out_before);

    /* Simulate training: set B to nonzero values */
    for (i = 0; i < D_OUT; i++) {
        l.B[i][0] = (randf()*2-1) * 0.1f;
        l.B[i][1] = (randf()*2-1) * 0.1f;
    }

    lora_forward(&l, x, out_after);

    printf("LoRA forward pass:\n\n");
    printf("  output = W*x + (alpha/r) * B*A*x\n\n");

    printf("  Before training, B=0 and LoRA idle\n    "
           "[");
    for (i = 0; i < D_OUT; i++)
        printf("%+.4f%s", out_before[i],
               i<D_OUT-1?", ":"");
    printf("]\n\n");

    printf("  After training, B nonzero\n    [");
    for (i = 0; i < D_OUT; i++)
        printf("%+.4f%s", out_after[i],
               i<D_OUT-1?", ":"");
    printf("]\n\n");

    printf("  Difference (the LoRA adaptation):\n    "
           "[");
    for (i = 0; i < D_OUT; i++)
        printf("%+.4f%s",
               out_after[i]-out_before[i], 
               i<D_OUT-1?", ":"");
    printf("]\n\n");

    printf("  B starts at zero, so the model is\n");
    printf("  identical to the pretrained one at "
           "the\n");
    printf("  start. Only A and B ever move.\n\n");

    int frozen = D_OUT * D_IN;
    int trainable = RANK * D_IN + D_OUT * RANK;
    printf("  Frozen params:    %d\n", frozen);
    printf("  Trainable params: %d\n", trainable);
    printf("  Ratio:            %.1f%%\n\n",
           100.0f * trainable / frozen);
    printf("  Two thirds is a terrible ratio and it\n");
    printf("  is honest. At d=6 with r=2 the "
           "adapter\n");
    printf("  is nearly the size of what it adapts.\n");
    printf("  LoRA only pays when d is large, since\n");
    printf("  W grows as d squared while A and B "
           "grow\n");
    printf("  as d. Listing 5 has the real numbers.\n");

    return 0;
}
