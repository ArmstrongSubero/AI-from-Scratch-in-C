/* 060_Embeddings_Backward_Pass.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_CAT 5
#define EMBED_DIM 3

typedef struct {
    float table[N_CAT][EMBED_DIM];
    float grad[N_CAT][EMBED_DIM];
}
Embedding;

static void embed_zero_grad(Embedding *e)
{
    memset(e->grad, 0, sizeof(e->grad));
}

static const float
    *embed_forward(const Embedding *e, int cat)
{
    return e->table[cat];
}

static void embed_backward(Embedding *e, int cat, 
                            const float *upstream_grad)
{
    /* Only the selected row gets gradient */
    int i;
    for (i = 0; i < EMBED_DIM; i++)
        e->grad[cat][i] += upstream_grad[i];
}

static void embed_update(Embedding *e, float lr)
{
    int i, j;
    for (i = 0; i < N_CAT; i++)
        for (j = 0; j < EMBED_DIM; j++)
            e->table[i][j] -= lr * e->grad[i][j];
}

int main(void)
{
    Embedding e;
    int i, j;

    /* Initialize */
    srand(42);
    for (i = 0; i < N_CAT; i++)
        for (j = 0; j < EMBED_DIM; j++)
            e.table[i][j] = ((float)rand() / RAND_MAX)
                * 0.2f - 0.1f;

    /* Simulate: look up category 2, get some
       upstream gradient */
    int cat = 2;
    embed_zero_grad(&e);

    printf("Before update (category %d):\n", cat);
    printf("  Embedding: [%.4f, %.4f, %.4f]\n",
           e.table[cat][0], e.table[cat][1], 
               e.table[cat][2]);

    /* Pretend the upstream gradient is [0.1, -0.2,
       0.05] */
    float upstream[] = { 0.1f, -0.2f, 0.05f };
    embed_backward(&e, cat, upstream);

    printf("\n  Upstream gradient: "
           "[%.2f, %.2f, %.2f]\n",
           upstream[0], upstream[1], upstream[2]);
    printf("  Gradient buffer:\n");
    for (i = 0; i < N_CAT; i++) {
        printf("    Cat %d: [%.2f, %.2f, %.2f]", i,
               e.grad[i][0], e.grad[i][1], 
                   e.grad[i][2]);
        if (i == cat)
            printf("  <-- only this row has gradient");
        printf("\n");
    }

    embed_update(&e, 0.1f);

    printf("\nAfter update:\n");
    printf("  Category %d: [%.4f, %.4f, %.4f]\n",
           cat, e.table[cat][0], e.table[cat][1], 
               e.table[cat][2]);

    /* Show that other categories did not change */
    printf("  Category 0: [%.4f, %.4f, %.4f] "
           "(unchanged)\n",
           e.table[0][0], e.table[0][1], e.table[0][2]);

    return 0;
}
