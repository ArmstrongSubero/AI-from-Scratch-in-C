/* 139_Problem.c */
#include <stdio.h>

int main(void)
{
    printf("K/V projection cost, cache against "
           "no cache\n\n");
    printf("  Seq len   Without cache     With cache "
           "      Savings\n");
    printf("  -------   ----------------  ----------"
           "-----  -------\n");

    int lengths[] = { 10, 100, 1000, 4096 };
    int n = 4, i;

    for (i = 0; i < n; i++) {
        int L = lengths[i];

        /* Without cache: recompute all K/V at every
           step */
        long long without = 0;
        int t;
        for (t = 1; t <= L; t++)
            without += t;  /* t projections at step t */

        /* With cache: only compute 1 new K/V per
           step */
        long long with_cache = L;
        /* L projections total */
        /* Plus L lookups over the cache */

        printf("  %5d     %12lld       %12lld    "
               "%7.1fx\n",
               L, without, with_cache, 
               (float)without / with_cache);
    }

    printf("\n  K/V projections fall from O(n^2) "
           "to O(n).\n");
    printf("  At length 4096 that is 2048.5 times "
           "fewer.\n");
    printf("  Scoring is still O(n) per token, but\n");
    printf("  the K/V are just lookups.\n");

    return 0;
}
