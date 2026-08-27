/* 063_BPE_Trace.c */
#include <stdio.h>
#include <string.h>

int main(void)
{
    printf("BPE algorithm trace:\n\n");
    printf("Training corpus: \"low lower newest "
           "widest\"\n\n");

    printf("Step 0 - Split into characters (with "
           "end-of-word marker '_'):\n");
    printf("  low:    l o w _\n");
    printf("  lower:  l o w e r _\n");
    printf("  newest: n e w e s t _\n");
    printf("  widest: w i d e s t _\n\n");

    printf("Step 1 - Count adjacent pairs:\n");
    printf("  (l,o): 2   (o,w): 2   (w,_): 1   (w,e): "
           "1\n");
    printf("  (e,r): 1   (r,_): 1   (n,e): 1   (e,w): "
           "1\n");
    printf("  (e,s): 2   (s,t): 2   (t,_): 2   (w,i): "
           "1\n");
    printf("  (i,d): 1   (d,e): 1\n\n");

    printf("Most frequent pairs: (l,o)=2, (o,w)=2, "
           "(e,s)=2, (s,t)=2, (t,_)=2\n");
    printf("Pick one (e.g., (e,s)) and merge into new "
           "token 'es'.\n\n");

    printf("Step 2 - After merging (e,s) -> 'es':\n");
    printf("  low:    l o w _\n");
    printf("  lower:  l o w e r _\n");
    printf("  newest: n e w es t _\n");
    printf("  widest: w i d es t _\n\n");

    printf("Step 3 - Recount pairs, merge most "
           "frequent...\n");
    printf("  (es,t): 2 -> merge into 'est'\n\n");

    printf("Step 4 - After merging (es,t) -> 'est':\n");
    printf("  low:    l o w _\n");
    printf("  lower:  l o w e r _\n");
    printf("  newest: n e w est _\n");
    printf("  widest: w i d est _\n\n");

    printf("Continue until desired vocabulary size is "
           "reached.\n");
    printf("Common subwords like 'est' become single "
           "tokens.\n");
    printf("Rare words are still decomposed into "
           "pieces.\n");

    return 0;
}
