/* 058_Embedding_Layer.c */
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *table;       /* [n_categories][embed_dim] */
    int n_categories;
    int embed_dim;
}
Embedding;

static Embedding embed_create(int n_categories, 
    int embed_dim)
{
    Embedding e;
    int i;

    e.n_categories = n_categories;
    e.embed_dim = embed_dim;
    e.table = (float *)malloc(
        n_categories * embed_dim * sizeof(float));

    /* Initialize with small random values */
    for (i = 0; i < n_categories * embed_dim; i++)
        e.table[i] = ((float)rand() / RAND_MAX)
            * 0.2f - 0.1f;

    return e;
}

static const float *embed_lookup(const Embedding *e, 
    int category)
{
    /* Just return a pointer to the right row */
    return &e->table[category * e->embed_dim];
}

static void embed_free(Embedding *e)
{
    free(e->table);
}

int main(void)
{
    Embedding e;
    int i, c;

    srand(42);
    e = embed_create(5, 3);
    /* 5 categories, 3-dimensional embeddings */

    printf("Embedding table (5 categories x 3 "
           "dimensions):\n\n");
    for (c = 0; c < 5; c++) {
        const float *vec = embed_lookup(&e, c);
        printf("  Category %d: [%+.3f, %+.3f, %+.3f]\n",
               c, vec[0], vec[1], vec[2]);
    }

    printf("\nLookup is O(1). No matrix multiply "
           "needed.\n");
    printf("The values are learned during training, "
           "just\n");
    printf("like any other weight in the network.\n");

    embed_free(&e);
    return 0;
}
