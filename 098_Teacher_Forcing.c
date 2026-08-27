/* 098_Teacher_Forcing.c */
#include <stdio.h>

#define V 6   /* vocabulary size */

/* 0=<S>  1='b'  2='o'  3='n'  4='j'  5=<E> */
static const char *tok[V] = {
    "<S>", "b", "o", "n", "j", "<E>"
};

/* A stand-in for a partly trained decoder. Row p
   holds the score the model gives each next token
   after p. Every row is right except row 1, where
   the model wrongly prefers 'j' over 'o'. */
static const float score[V][V] = {
    /*      <S>   b     o     n     j    <E> */
    /* <S> */ { 0.0f, 0.7f, 0.1f, 0.1f, 0.0f, 0.1f },
    /* b   */ { 0.0f, 0.0f, 0.3f, 0.1f, 0.5f, 0.1f },
    /* o   */ { 0.0f, 0.1f, 0.0f, 0.6f, 0.2f, 0.1f },
    /* n   */ { 0.0f, 0.0f, 0.1f, 0.0f, 0.8f, 0.1f },
    /* j   */ { 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.9f },
    /* <E> */ { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f },
};

static int predict(int prev)
{
    int best = 0, i;
    for (i = 1; i < V; i++)
        if (score[prev][i] > score[prev][best])
            best = i;
    return best;
}

int main(void)
{
    /* The target output is b o n j <E> */
    int target[] = { 1, 2, 3, 4, 5 };
    int len = 5;
    int t, prev, pred, hits;

    printf("Target sequence: ");
    for (t = 0; t < len; t++)
        printf("%s ", tok[target[t]]);
    printf("\n\n");

    printf("Autoregressive, feeding back "
           "predictions\n\n");
    printf("  step  input  predict  target  \n");
    printf("  ----  -----  -------  ------  \n");
    prev = 0;
    hits = 0;
    for (t = 0; t < len; t++) {
        pred = predict(prev);
        if (pred == target[t]) hits++;
        printf("  %3d   %-5s  %-7s  %-6s  %s\n",
               t, tok[prev], tok[pred], tok[target[t]], 
               pred == target[t] ? "ok" : "wrong");
        /* feed our own output back */
        prev = pred;
        if (pred == 5) {
            t++;
            break;
            }
    }
    printf("\n  correct: %d of %d\n", hits, len);
    printf("  One wrong step at t=1 sent it to <E>\n");
    printf("  early, so the rest was never tried.\n\n");

    printf("Teacher forcing, feeding back the "
           "target\n\n");
    printf("  step  input  predict  target  \n");
    printf("  ----  -----  -------  ------  \n");
    prev = 0;
    hits = 0;
    for (t = 0; t < len; t++) {
        pred = predict(prev);
        if (pred == target[t]) hits++;
        printf("  %3d   %-5s  %-7s  %-6s  %s\n",
               t, tok[prev], tok[pred], tok[target[t]], 
               pred == target[t] ? "ok" : "wrong");
        /* feed the correct token */
        prev = target[t];
    }
    printf("\n  correct: %d of %d\n", hits, len);
    printf("  The same flawed model scores far "
           "better,\n");
    printf("  because one bad step no longer spoils\n");
    printf("  the input to every step after it.\n\n");

    printf("The model is identical in both runs.\n");
    printf("Only the input at each step differs. At\n");
    printf("training time we can feed "
           "the target, at\n");
    printf("inference we cannot. That gap is called\n");
    printf("exposure bias.\n");

    return 0;
}
