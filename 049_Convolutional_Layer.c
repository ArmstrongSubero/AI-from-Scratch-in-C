/* 049_Convolutional_Layer.c */
#include <stdio.h>
#include <math.h>

/* 1D convolution layer:
   input: [in_channels][in_width]
   kernel: [out_channels][in_channels][ker_width]
   bias: [out_channels]
   output: [out_channels][out_width]
*/

static void conv1d_layer(
    const float *input, int in_ch, int in_w, 
    const float *kernel, int out_ch, int ker_w, 
    const float *bias, 
    int stride, int pad, 
    float *output)
{
    int out_w = (in_w + 2 * pad - ker_w) / stride + 1;
    int oc, ic, ow, kw;

    for (oc = 0; oc < out_ch; oc++) {
        for (ow = 0; ow < out_w; ow++) {
            float sum = bias[oc];
            int start = ow * stride - pad;

            for (ic = 0; ic < in_ch; ic++) {
                for (kw = 0; kw < ker_w; kw++) {
                    int idx = start + kw;
                    if (idx >= 0 && idx < in_w)
                        sum += input[ic * in_w + idx]
                      * kernel[(oc * in_ch + ic)
                                 * ker_w + kw];
                }
            }
            output[oc * out_w + ow] = sum;
        }
    }
}

static float relu(float z)
{
    return z > 0.0f ? z : 0.0f;
}

int main(void)
{
    /* 2-channel input signal, 8 samples wide */
    float input[2 * 8] = {
        /* Channel 0: a spike */
        0, 0, 1, 2, 1, 0, 0, 0, 
        /* Channel 1: a ramp */
        0, 1, 2, 3, 4, 5, 6, 7, 
    };

    /* 3 kernels, each 2-channel, width 3 */
    /* Kernel layout: [out_ch][in_ch][ker_w] =
       [3][2][3] */
    float kernel[3 * 2 * 3] = {
        /* Kernel 0: spike detector on ch0, ignore
           ch1 */
        -1, 2, -1, 0, 0, 0, 
        /* Kernel 1: rising edge on ch1, ignore ch0 */
         0, 0, 0, -1, 0, 1, 
        /* Kernel 2: both channels combined */
         1, 0, -1, 0, 1, 0, 
    };
    float bias[3] = { 0, 0, 0 };

    int out_w = 8 - 3 + 1;  /* no padding, stride 1 */
    float output[3 * 6];
    int oc, ow;

    conv1d_layer(input, 2, 8, kernel, 3, 3, bias, 
        1, 0, output);

    printf("Input (2 channels, 8 wide):\n");
    printf("  Ch0: ");
    for (ow = 0; ow < 8; ow++)
        printf("%5.1f ", input[ow]);
    printf("\n");
    printf("  Ch1: ");
    for (ow = 0; ow < 8; ow++) printf("%5.1f ",
        input[8+ow]);
    printf("\n\n");

    printf("Output (3 kernels, %d wide):\n", out_w);
    const char *names[] = { "spike", "edge", "combo" };
    for (oc = 0; oc < 3; oc++) {
        printf("  %-5s: ", names[oc]);
        for (ow = 0; ow < out_w; ow++)
            printf("%5.1f ", output[oc * out_w + ow]);
        printf("\n");
    }

    /* Apply ReLU */
    printf("\nAfter ReLU:\n");
    for (oc = 0; oc < 3; oc++) {
        printf("  %-5s: ", names[oc]);
        for (ow = 0; ow < out_w; ow++)
            printf("%5.1f ",
                relu(output[oc * out_w + ow]));
        printf("\n");
    }

    return 0;
}
