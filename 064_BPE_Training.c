/* 064_BPE_Training.c */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_TOKENS 200
#define MAX_TOKEN_LEN 16
#define MAX_WORDS 20
#define MAX_WORD_TOKENS 20

/* A word split into tokens */
typedef struct {
    char tokens[MAX_WORD_TOKENS][MAX_TOKEN_LEN];
    int n_tokens;
    int freq;   /* how many times this word appears */
}
Word;

/* Find the most frequent adjacent pair across all
   words */
static int find_best_pair(Word *words, int n_words, 
                           char best_a[MAX_TOKEN_LEN], 
                           char best_b[MAX_TOKEN_LEN])
{
    int best_count = 0;
    int w, t;

    for (w = 0; w < n_words; w++) {
        for (t = 0; t < words[w].n_tokens - 1; t++) {
            /* Count this pair across all words */
            int count = 0;
            int w2, t2;
            for (w2 = 0; w2 < n_words; w2++) {
                for (t2 = 0;
                 t2 < words[w2].n_tokens - 1;
                 t2++) {
                    if (strcmp(words[w2].tokens[t2],
                         words[w].tokens[t]) == 0 &&
                        strcmp(words[w2].tokens[t2+1], 
                            words[w].tokens[t+1]) == 0)
                        count += words[w2].freq;
                }
            }
            if (count > best_count) {
                best_count = count;
                strcpy(best_a, words[w].tokens[t]);
                strcpy(best_b, words[w].tokens[t+1]);
            }
        }
    }
    return best_count;
}

/* Merge all occurrences of (a, b) into "ab" */
static void merge_pair(Word *words, int n_words, 
                        const char *a, const char *b)
{
    char merged[MAX_TOKEN_LEN];
    snprintf(merged, MAX_TOKEN_LEN, "%s%s", a, b);

    int w, t;
    for (w = 0; w < n_words; w++) {
        for (t = 0; t < words[w].n_tokens - 1; t++) {
            if (strcmp(words[w].tokens[t], a) == 0 &&
                strcmp(words[w].tokens[t+1], b) == 0) {
                /* Replace token[t] with merged,
                   remove token[t+1] */
                strcpy(words[w].tokens[t], merged);
                int j;
                for (j = t + 1;
                     j < words[w].n_tokens - 1; j++)
                    strcpy(words[w].tokens[j], 
                        words[w].tokens[j+1]);
                words[w].n_tokens--;
                /* Don't advance t: check if the new
                   merged token
                   can merge with the next one too */
            }
        }
    }
}

static void print_words(const Word *words, int n_words)
{
    int w, t;
    for (w = 0; w < n_words; w++) {
        printf("  (%dx) ", words[w].freq);
        for (t = 0; t < words[w].n_tokens; t++)
            printf("[%s]", words[w].tokens[t]);
        printf("\n");
    }
}

int main(void)
{
    /* Corpus: word frequencies */
    const char *raw[] = { "low",  "lower",  "newest",
        "widest", "new" };
    int freqs[] = { 5, 2, 6, 3, 2 };
    int n_words = 5;

    Word words[MAX_WORDS];
    int w, i;

    /* Initialize: split each word into characters +
       end marker */
    for (w = 0; w < n_words; w++) {
        words[w].freq = freqs[w];
        words[w].n_tokens = 0;
        for (i = 0; raw[w][i]; i++) {
            char c[2] = { raw[w][i], '\0' };
            strcpy(words[w].tokens[words[w].n_tokens++],
                   c);
        }
        strcpy(words[w].tokens[words[w].n_tokens++],
               "_");
    }

    printf("BPE Training\n\n");
    printf("Initial state (characters + end "
           "marker):\n");
    print_words(words, n_words);

    /* Perform merge steps */
    int step;
    for (step = 0; step < 10; step++) {
        char a[MAX_TOKEN_LEN], b[MAX_TOKEN_LEN];
        int count = find_best_pair(words, 
            n_words, a, b);
        if (count < 2) break;
        /* stop when no pair appears 2+ times */

        printf("\nMerge %d: (%s, %s) count=%d -> "
               "[%s%s]\n",
               step + 1, a, b, count, a, b);
        merge_pair(words, n_words, a, b);
        print_words(words, n_words);
    }

    return 0;
}
