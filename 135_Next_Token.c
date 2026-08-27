/* 135_Next_Token.c */
#include <stdio.h>

int main(void)
{
    /* A simple token sequence */
    int tokens[] = { 4, 2, 7, 1, 5, 3, 6, 0 };
    int len = 8;
    int i;

    printf("Next-token prediction setup:\n\n");
    printf("  Tokens:  ");
    for (i = 0; i < len; i++) printf("%d ", tokens[i]);
    printf("\n\n");

    printf("  Position  Input  Target\n");
    printf("  --------  -----  ------\n");
    for (i = 0; i < len - 1; i++) {
        printf("  %4d      %3d    %3d\n", i,
               tokens[i], tokens[i + 1]);
    }

    printf("\n  At each position the model sees\n");
    printf("  tokens[0..i] and predicts "
           "tokens[i+1].\n");
    printf("  The causal mask is what stops it\n");
    printf("  from seeing future tokens.\n\n");

    printf("  In training, ALL positions "
           "are trained\n");
    printf("  at once.\n");
    printf("  The loss sums cross-entropy "
           "over them.\n");
    printf("  Seq2seq trains only on the output\n");
    printf("  side, so this gets far more from\n");
    printf("  the same text.\n");

    return 0;
}
