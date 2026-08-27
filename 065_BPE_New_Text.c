/* 065_BPE_New_Text.c */
#include <stdio.h>
#include <string.h>

#define MAX_TOKENS 50
#define MAX_TOKEN_LEN 16
#define MAX_RULES 20

typedef struct {
    char a[MAX_TOKEN_LEN];
    char b[MAX_TOKEN_LEN];
    char merged[MAX_TOKEN_LEN];
}
MergeRule;

static void apply_merges(
        char tokens[MAX_TOKENS][MAX_TOKEN_LEN],
                          int *n_tokens, 
                          const MergeRule
                              *rules, int n_rules)
{
    int r, t;
    for (r = 0; r < n_rules; r++) {
        for (t = 0; t < *n_tokens - 1; t++) {
            if (strcmp(tokens[t], rules[r].a) == 0 &&
                strcmp(tokens[t+1], rules[r].b) == 0) {
                strcpy(tokens[t], rules[r].merged);
                int j;
                for (j = t + 1; j < *n_tokens - 1; j++)
                    strcpy(tokens[j], tokens[j+1]);
                (*n_tokens)--;
                t--;  /* recheck at same position */
            }
        }
    }
}

/* Simple token-to-ID mapping */
static int token_to_id(const char *token, 
                        char vocab[100][MAX_TOKEN_LEN], 
                            int *vocab_size)
{
    int i;
    for (i = 0; i < *vocab_size; i++)
        if (strcmp(vocab[i], token) == 0)
            return i;
    strcpy(vocab[*vocab_size], token);
    return (*vocab_size)++;
}

int main(void)
{
    /* Pretend these merge rules were learned from
       training */
    MergeRule rules[] = {
        { "e", "s", "es" },
        { "es", "t", "est" },
        { "l", "o", "lo" },
        { "lo", "w", "low" },
        { "n", "e", "ne" },
        { "ne", "w", "new" },
    };
    int n_rules = 6;

    /* Tokenize some text */
    const char *test_words[] = { "newest", "lower",
        "low", "newest", "unknown" };
    int n_test = 5;

    char vocab[100][MAX_TOKEN_LEN];
    int vocab_size = 0;
    int w, i;

    printf("Merge rules (learned from training):\n");
    for (i = 0; i < n_rules; i++)
        printf("  %s + %s -> %s\n", rules[i].a,
            rules[i].b, rules[i].merged);

    printf("\nTokenizing new text:\n\n");

    for (w = 0; w < n_test; w++) {
        char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
        int n_tokens = 0;

        /* Split into characters + end marker */
        for (i = 0; test_words[w][i]; i++) {
            char c[2] = { test_words[w][i], '\0' };
            strcpy(tokens[n_tokens++], c);
        }
        strcpy(tokens[n_tokens++], "_");

        /* Apply merge rules */
        apply_merges(tokens, &n_tokens, rules, n_rules);

        /* Print result */
        printf("  \"%s\" -> ", test_words[w]);
        for (i = 0; i < n_tokens; i++)
            printf("[%s]", tokens[i]);

        /* Assign IDs */
        printf("  IDs: [");
        for (i = 0; i < n_tokens; i++) {
            int id = token_to_id(tokens[i], vocab, 
                &vocab_size);
            printf("%d%s", id,
                i < n_tokens - 1 ? ", " : "");
        }
        printf("]\n");
    }

    printf("\nVocabulary (%d tokens):\n", vocab_size);
    for (i = 0; i < vocab_size; i++)
        printf("  %d: \"%s\"\n", i, vocab[i]);

    printf("\n\"unknown\" was split into characters "
           "because no merge\n");
    printf("rules matched. BPE can handle any "
           "input.\n");

    return 0;
}
