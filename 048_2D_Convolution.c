/* 048_2D_Convolution.c */
#include <stdio.h>

#define IMG_H 5
#define IMG_W 5
#define KER_H 3
#define KER_W 3
#define OUT_H (IMG_H - KER_H + 1)
#define OUT_W (IMG_W - KER_W + 1)

int main(void)
{
    /* A small "image" with a vertical edge */
    float image[IMG_H][IMG_W] = {
        { 0, 0, 1, 1, 1 }, 
        { 0, 0, 1, 1, 1 }, 
        { 0, 0, 1, 1, 1 }, 
        { 0, 0, 1, 1, 1 }, 
        { 0, 0, 1, 1, 1 }, 
    };

    /* Vertical edge detector */
    float kernel[KER_H][KER_W] = {
        { -1, 0, 1 }, 
        { -1, 0, 1 }, 
        { -1, 0, 1 }, 
    };

    float output[OUT_H][OUT_W];
    int oi, oj, ki, kj;

    /* 2D convolution */
    for (oi = 0; oi < OUT_H; oi++) {
        for (oj = 0; oj < OUT_W; oj++) {
            float sum = 0.0f;
            for (ki = 0; ki < KER_H; ki++)
                for (kj = 0; kj < KER_W; kj++)
                    sum += image[oi + ki][oj + kj]
                        * kernel[ki][kj];
            output[oi][oj] = sum;
        }
    }

    printf("Image:\n");
    for (oi = 0; oi < IMG_H; oi++) {
        printf("  ");
        for (oj = 0; oj < IMG_W; oj++)
            printf("%5.1f ", image[oi][oj]);
        printf("\n");
    }

    printf("\nKernel (vertical edge detector):\n");
    for (ki = 0; ki < KER_H; ki++) {
        printf("  ");
        for (kj = 0; kj < KER_W; kj++)
            printf("%5.1f ", kernel[ki][kj]);
        printf("\n");
    }

    printf("\nOutput:\n");
    for (oi = 0; oi < OUT_H; oi++) {
        printf("  ");
        for (oj = 0; oj < OUT_W; oj++)
            printf("%5.1f ", output[oi][oj]);
        printf("\n");
    }

    printf("\nThe edge appears as a column of 3s where "
           "the\n");
    printf("transition from 0 to 1 occurs in the "
           "image.\n");

    return 0;
}
