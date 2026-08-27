/* 056_One_Hot_Encoding.c */
#include <stdio.h>
#include <string.h>

static void one_hot(int category, 
    int n_categories, float *out)
{
    int i;
    for (i = 0; i < n_categories; i++)
        out[i] = 0.0f;
    out[category] = 1.0f;
}

int main(void)
{
    int n_categories = 5;
    float vec[5];
    int i, c;

    printf("One-hot encoding for 5 categories:\n\n");
    for (c = 0; c < n_categories; c++) {
        one_hot(c, n_categories, vec);
        printf("  Category %d: [", c);
        for (i = 0; i < n_categories; i++)
            printf("%.0f%s", vec[i],
                i < n_categories - 1 ? ", " : "");
        printf("]\n");
    }

    printf("\nNo ordering implied. Each category is "
           "equidistant\n");
    printf("from every other category (distance = "
           "sqrt(2)).\n");

    return 0;
}
