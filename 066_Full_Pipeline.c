/* 066_Full_Pipeline.c */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define EMBED_DIM 4
#define MAX_SEQ 20

int main(void)
{
    /* Simulated vocabulary (in a real system, this
       comes from BPE training) */
    const char *vocab[] = {
        "the", "cat", "sat", "on", "mat", "a",
            "big", "red",
        "dog", "ran", "fast", "<UNK>"
    };
    int vocab_size = 12;

    /* Simulated embeddings (in a real system, these
       are learned) */
    float embeddings[12][EMBED_DIM];
    srand(42);
    int i, j;
    for (i = 0; i < vocab_size; i++)
        for (j = 0; j < EMBED_DIM; j++)
            embeddings[i][j] =
                ((float)rand() / RAND_MAX) * 2 - 1;

    /* Input text */
    const char *text = "the big red cat sat on a mat";

    /* Step 1: Tokenize (simple word split for this
       demo) */
    int token_ids[MAX_SEQ];
    int seq_len = 0;
    char buf[32];
    int start = 0, len = strlen(text);

    printf("=== Full Pipeline: Text to Vectors "
           "===\n\n");
    printf("Step 1: Tokenize\n");
    printf("  Input: \"%s\"\n  Tokens: ", text);

    for (i = 0; i <= len; i++) {
        if (i == len || text[i] == ' ') {
            int wlen = i - start;
            if (wlen > 0 && wlen < 32) {
                strncpy(buf, text + start, wlen);
                buf[wlen] = '\0';

                /* Look up in vocab */
                int id = vocab_size - 1;  /* <UNK> */
                for (j = 0; j < vocab_size; j++) {
                    if (strcmp(vocab[j], buf) == 0) {
                        id = j;
                        break;
                    }
                }
                token_ids[seq_len++] = id;
                printf("\"%s\"(%d) ", buf, id);
            }
            start = i + 1;
        }
    }

    /* Step 2: Token IDs */
    printf("\n\nStep 2: Token IDs\n  [");
    for (i = 0; i < seq_len; i++)
        printf("%d%s", token_ids[i],
            i < seq_len - 1 ? ", " : "");
    printf("]\n");

    /* Step 3: Embedding lookup */
    printf("\nStep 3: Embedding Lookup "
           "(%d-dimensional)\n", EMBED_DIM);
    for (i = 0; i < seq_len; i++) {
        int id = token_ids[i];
        printf("  \"%s\" (id=%d) -> [", vocab[id], id);
        for (j = 0; j < EMBED_DIM; j++)
            printf("%+.2f%s", embeddings[id][j],
                j < EMBED_DIM - 1 ? ", " : "");
        printf("]\n");
    }

    printf("\nStep 4: Feed sequence of %d vectors into "
           "neural network\n", seq_len);
    printf("  Each vector is "
           "%d-dimensional\n", EMBED_DIM);
    printf("  Input tensor shape: [%d, %d]\n",
        seq_len, EMBED_DIM);
    printf("\n  This is where RNNs, LSTMs, or "
           "Transformers take over.\n");

    return 0;
}
