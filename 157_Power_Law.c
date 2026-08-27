/* 157_Power_Law.c */
#include <stdio.h>
#include <math.h>

int main(void)
{
    printf("Scaling laws: loss vs model size\n\n");
    printf("  %10s  %8s  %s\n", "Params",
           "Rel loss", "Improvement");
    printf("  %10s  %8s  %s\n", "----------",
           "--------", "-----------");

    /* L ~ N^(-0.076), normalized so 1B params = 1.0
       loss */
    float alpha = 0.076f;
    double sizes[] = { 1e8, 3e8, 1e9, 3e9, 
                       1e10, 3e10, 1e11, 7e11 };
    int n = 8;
    double base_loss = pow(1e9, -alpha);
    int i;

    for (i = 0; i < n; i++) {
        double loss = pow(sizes[i], -alpha) / base_loss;
        printf("  %9.1fB  %8.4f  ",
               sizes[i] / 1e9, loss);
        if (i > 0) {
            double prev = 
                pow(sizes[i-1], -alpha) / base_loss;
            printf("%.1f%% better",
                   (1.0 - loss/prev) * 100);
        }
        printf("\n");
    }

    printf("\n  Each 10x in parameters gives about "
           "%.0f%% lower loss.\n",
           (1.0 - pow(0.1, alpha)) * 100);
    printf("  The percentage is the same at every\n");
    printf("  scale. What shrinks is the absolute\n");
    printf("  drop, since 16%% of a smaller number\n");
    printf("  is a smaller number.\n");
    printf("  It never stops either. A bigger model\n");
    printf("  is always better, given enough data "
           "to\n");
    printf("  train it on.\n");

    return 0;
}
