/* 160_Emergence.c */
#include <stdio.h>
#include <math.h>

int main(void)
{
    /* L ~ C^(-0.05), so a factor of ten in compute
       multiplies the loss by 10^(-0.05) every time, 
       whatever the compute already was. */
    double compute = 1;
    int i;

    printf("Loss against compute, L ~ C^-0.05\n\n");
    printf("  compute       rel loss     drop   "
           "absolute\n");
    printf("  step          at start   per 10x   "
           "    drop\n");
    printf("  -----------   --------   -------   "
           "--------\n");

    for (i = 0; i < 6; i++) {
        double loss = pow(compute, -0.05);
        double next = pow(compute * 10, -0.05);
        double pct = (1.0 - next / loss) * 100;
        printf("  %.0e -> %.0e  %8.4f  %6.1f%%  "
               "%7.4f\n",
               compute, compute * 10, loss, pct, 
               loss - next);
        compute *= 10;
    }

    printf("\n  The percentage never moves. Every\n");
    printf("  factor of ten buys the same 10.9%%, "
           "which\n");
    printf("  is what a power law means.\n\n");

    printf("  The absolute column tells the other "
           "half\n");
    printf("  of the story. Read the first and last\n");
    printf("  rows of it against each other. The "
           "same\n");
    printf("  10.9%% is worth far less by the end,\n");
    printf("  because a constant fraction of a "
           "smaller\n");
    printf("  number is less. That is where the "
           "sense\n");
    printf("  of diminishing returns comes from, "
           "and\n");
    printf("  it is about the absolute drop rather\n");
    printf("  than the rate.\n");

    return 0;
}
