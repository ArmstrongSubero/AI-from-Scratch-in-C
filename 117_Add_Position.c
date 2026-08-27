/* 117_Add_Position.c */
#include <stdio.h>
#include <math.h>

#define D_MODEL 6
#define SEQ_LEN 4

static void sinusoidal_pe(int pos, 
    int d_model, float *pe)
{
    int i;
    for (i = 0; i < d_model; i++) {
        float e = (float)(i / 2 * 2) / d_model;
        float angle = pos / powf(10000.0f, e);
        pe[i] = (i % 2 == 0)
            ? sinf(angle)
            : cosf(angle);
    }
}

int main(void)
{
    /* Token embeddings, as if from the table */
    float embeddings[SEQ_LEN][D_MODEL] = {
        /* the */
        { 0.5f, -0.2f, 0.8f, 0.1f, -0.3f, 0.6f }, 
        /* cat */
        { 0.3f, 0.7f, -0.1f, 0.4f, 0.5f, -0.2f }, 
        /* sat */
        { -0.4f, 0.3f, 0.6f, -0.5f, 0.2f, 0.1f }, 
        /* down */
        { 0.2f, -0.1f, 0.4f, 0.7f, -0.6f, 0.3f }, 
    };
    const char *words[] = { "the", "cat",
        "sat", "down" };
    int pos, i;

    printf("Token embeddings + "
           "positional encoding:\n\n");

    printf("  Token embeddings (no position):\n");
    for (pos = 0; pos < SEQ_LEN; pos++) {
        printf("    %-5s [", words[pos]);
        for (i = 0; i < D_MODEL; i++)
            printf("%+.2f%s", embeddings[pos][i],
                   i < D_MODEL-1 ? "," : "");
        printf("]\n");
    }

    printf("\n  Positional encodings:\n");
    for (pos = 0; pos < SEQ_LEN; pos++) {
        float pe[D_MODEL];
        sinusoidal_pe(pos, D_MODEL, pe);
        printf("    pos %d  [", pos);
        for (i = 0; i < D_MODEL; i++)
            printf("%+.2f%s", pe[i],
                   i < D_MODEL-1 ? "," : "");
        printf("]\n");
    }

    printf("\n  After addition (embedding + "
           "position):\n");
    for (pos = 0; pos < SEQ_LEN; pos++) {
        float pe[D_MODEL];
        sinusoidal_pe(pos, D_MODEL, pe);
        printf("    %-5s [", words[pos]);
        for (i = 0; i < D_MODEL; i++)
            printf("%+.2f%s",
                embeddings[pos][i] + pe[i], 
                   i < D_MODEL-1 ? "," : "");
        printf("]\n");
    }

    printf("\n  The same word at two positions now "
           "has\n");
    printf("  different vectors. 'cat' at 1 differs\n");
    printf("  from 'cat' at position 5.\n");

    return 0;
}
