/* 062_Word_Tokenization.c */
#include <stdio.h>
#include <string.h>

#define MAX_WORDS 100
#define MAX_WORD_LEN 32

typedef struct {
    char words[MAX_WORDS][MAX_WORD_LEN];
    int count;
}
Vocab;

static int vocab_add(Vocab *v, const char *word)
{
    int i;
    /* Check if word already exists */
    for (i = 0; i < v->count; i++)
        if (strcmp(v->words[i], word) == 0)
            return i;
    /* Add new word */
    strncpy(v->words[v->count], word, MAX_WORD_LEN - 1);
    v->words[v->count][MAX_WORD_LEN - 1] = '\0';
    return v->count++;
}

static int vocab_lookup(const Vocab *v, 
    const char *word)
{
    int i;
    for (i = 0; i < v->count; i++)
        if (strcmp(v->words[i], word) == 0)
            return i;
    return -1;  /* unknown word */
}

int main(void)
{
    const char
        *text = "the cat sat on the mat "
                "the dog sat on the rug";
    Vocab v = { .count = 0 };
    char buf[MAX_WORD_LEN];
    int tokens[MAX_WORDS];
    int n_tokens = 0;
    int i = 0, len = strlen(text);

    /* Tokenize by splitting on spaces */
    int start = 0;
    for (i = 0; i <= len; i++) {
        if (i == len || text[i] == ' ') {
            int wlen = i - start;
            if (wlen > 0 && wlen < MAX_WORD_LEN) {
                strncpy(buf, text + start, wlen);
                buf[wlen] = '\0';
                tokens[n_tokens++] = vocab_add(&v, buf);
            }
            start = i + 1;
        }
    }

    printf("Text: \"%s\"\n\n", text);
    printf("Vocabulary (%d words):\n", v.count);
    for (i = 0; i < v.count; i++)
        printf("  %d: \"%s\"\n", i, v.words[i]);

    printf("\nToken IDs: [");
    for (i = 0; i < n_tokens; i++)
        printf("%d%s", tokens[i],
            i < n_tokens - 1 ? ", " : "");
    printf("]\n");

    printf("\nSequence length: %d (one per "
           "word)\n", n_tokens);

    return 0;
}
