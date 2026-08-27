/* 061_Tokenizier.c */
#include <stdio.h>
#include <string.h>

int main(void)
{
    const char *text = "hello world";
    int i, len = strlen(text);

    printf("Text: \"%s\"\n\n", text);
    printf("Character tokens:\n");
    for (i = 0; i < len; i++)
        printf("  '%c' -> %d\n", text[i],
            (unsigned char)text[i]);

    printf("\nToken IDs: [");
    for (i = 0; i < len; i++)
        printf("%d%s", (unsigned char)text[i],
            i < len - 1 ? ", " : "");
    printf("]\n");

    printf("\nVocabulary size: 256 (one per byte)\n");
    printf("Sequence length: %d (one per "
           "character)\n", len);

    return 0;
}
