/* 053_Flatten.c */
#include <stdio.h>

int main(void)
{
    /* Simulated output of conv layers: 4 channels, 3
       values each */
    float feature_maps[4][3] = {
        { 2.1f, 0.0f, 1.5f }, 
        { 0.0f, 3.2f, 0.0f }, 
        { 1.1f, 1.1f, 0.0f }, 
        { 0.0f, 0.0f, 2.8f }, 
    };
    int n_channels = 4, spatial_size = 3;

    /* Flatten: just read all values in order */
    int flat_size = n_channels * spatial_size;
    float flat[12];
    int c, s, idx = 0;

    for (c = 0; c < n_channels; c++)
        for (s = 0; s < spatial_size; s++)
            flat[idx++] = feature_maps[c][s];

    printf("Feature maps (4 channels x 3 spatial):\n");
    for (c = 0; c < n_channels; c++) {
        printf("  Ch%d: ", c);
        for (s = 0; s < spatial_size; s++)
            printf("%.1f ", feature_maps[c][s]);
        printf("\n");
    }

    printf("\nFlattened (%d values):\n  ", flat_size);
    for (idx = 0; idx < flat_size; idx++)
        printf("%.1f ", flat[idx]);
    printf("\n");

    printf("\nThis vector feeds into a dense layer for "
           "classification.\n");

    return 0;
}
