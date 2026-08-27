/* 054_Complete_CNN.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

static float relu(float z)
{
    return z > 0 ? z : 0;
}

/* --- Convolution layer --- */
static void conv1d(const float *in, 
    int in_ch, int in_w, 
                   const float *ker, const float *bias, 
                   int out_ch, int ker_w, 
                   float *out, int *out_w)
{
    *out_w = in_w - ker_w + 1;
    int oc, ic, ow, kw;
    for (oc = 0; oc < out_ch; oc++) {
        for (ow = 0; ow < *out_w; ow++) {
            float sum = bias[oc];
            for (ic = 0; ic < in_ch; ic++)
                for (kw = 0; kw < ker_w; kw++)
                    sum += in[ic * in_w + ow + kw]
                      * ker[(oc * in_ch + ic)
                              * ker_w + kw];
            out[oc * (*out_w) + ow] = relu(sum);
        }
    }
}

/* --- Max pooling --- */
static void maxpool(const float *in, 
    int channels, int in_w, 
                    float *out, int *out_w)
{
    *out_w = in_w / 2;
    int c, i;
    for (c = 0; c < channels; c++) {
        for (i = 0; i < *out_w; i++) {
            float a = in[c * in_w + i * 2];
            float b = in[c * in_w + i * 2 + 1];
            out[c * (*out_w) + i] = a > b ? a : b;
        }
    }
}

/* --- Dense layer --- */
static void dense(const float *in, int in_size, 
                  const float *w, const float *bias, 
                  int out_size, float *out)
{
    int i, j;
    for (i = 0; i < out_size; i++) {
        float sum = bias[i];
        for (j = 0; j < in_size; j++)
            sum += w[i * in_size + j] * in[j];
        out[i] = sum;
    }
}

/* --- Softmax --- */
static void softmax(float *x, int n)
{
    float max_val = -FLT_MAX, sum = 0;
    int i;
    for (i = 0; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (i = 0; i < n; i++)
        x[i] /= sum;
}

int main(void)
{
    /* Three signal classes:
       0 = spike at start
       1 = spike in middle
       2 = spike at end */
    float signals[3][16] = {
        { 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0 }, 
        { 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 
            0, 0, 0, 0 }, 
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 1, 3 }, 
    };

    /* Hand-designed weights to demonstrate the
       architecture */

    /* Conv1: 1 input channel, 4 output channels,
       kernel size 3 */
    float conv1_w[4 * 1 * 3] = {
        2, -1, 0,    /* detects rising edge */
        0, -1, 2,    /* detects falling edge */
        -1, 2, -1,   /* detects spike */
        1, 1, 1,     /* smoother */
    };
    float conv1_b[4] = { 0, 0, 0, 0 };

    /* Conv2: 4 input channels, 2 output channels,
       kernel size 3 */
    float conv2_w[2 * 4 * 3];
    float conv2_b[2] = { 0, 0 };

    /* Initialize conv2 with small values */
    srand(42);
    int i;
    for (i = 0; i < 2 * 4 * 3; i++)
        conv2_w[i] = ((float)rand() / RAND_MAX)
            * 0.4f - 0.2f;

    /* Dense: flat_size -> 3 classes */
    /* We will compute flat_size after running the
       conv layers */

    float buf1[256], buf2[256], buf3[256];
    int w1, w2, w3;
    int s;

    /* Run each signal through the CNN */
    for (s = 0; s < 3; s++) {
        printf("Signal %d (spike at %s):\n", s,
              s == 0 ? "start"
                     : s == 1 ? "middle" : "end");

        /* Conv1: 1ch x 16 -> 4ch x 14 */
        conv1d(signals[s], 1, 16, conv1_w, conv1_b, 4, 
            3, buf1, &w1);
        printf("  After conv1 (%d ch x %d): ", 4, w1);
        /* Just show channel 2 (spike detector) */
        printf("ch2=[");
        for (i = 0; i < w1; i++)
            printf("%.0f%s", buf1[2*w1+i],
                i<w1-1?",":" ");
        printf("]\n");

        /* Pool1: 4ch x 14 -> 4ch x 7 */
        maxpool(buf1, 4, w1, buf2, &w2);
        printf("  After pool1 (%d ch x %d): ", 4, w2);
        printf("ch2=[");
        for (i = 0; i < w2; i++)
            printf("%.0f%s", buf2[2*w2+i],
                i<w2-1?",":" ");
        printf("]\n");

        /* Conv2: 4ch x 7 -> 2ch x 5 */
        conv1d(buf2, 4, w2, conv2_w, conv2_b, 2, 
            3, buf3, &w3);
        printf("  After conv2 (%d ch x %d)\n", 2, w3);

        /* Pool2: 2ch x 5 -> 2ch x 2 (drop last) */
        int w4 = w3 / 2;
        float buf4[256];
        maxpool(buf3, 2, 
            w3 - (w3 % 2 ? 1 : 0), buf4, &w4);
        printf("  After pool2 (%d ch x %d)\n", 2, w4);

        /* Flatten */
        int flat_size = 2 * w4;
        printf("  Flattened: %d values -> ["
               , flat_size);
        for (i = 0; i < flat_size; i++)
            printf("%.2f%s", buf4[i],
                i < flat_size-1 ? ", " : "");
        printf("]\n\n");
    }

    printf("In a trained CNN, the dense layer after "
           "flattening\n");
    printf("would classify based on these features.\n");
    printf("The conv layers have already extracted the "
           "relevant\n");
    printf("patterns. The dense layer just reads the "
           "summary.\n");

    return 0;
}
