/* 125_Backward.c */
#include <stdio.h>
#include <math.h>

#define DIM 5

static void layernorm_forward(const float *x,
                              float *out, 
                               int n, 
                                   float *std_inv_out)
{
    int i;
    double sum = 0;
    float eps = 1e-5f;
    for (i = 0; i < n; i++) sum += x[i];
    float mean = (float)(sum / n);
    double var = 0;
    for (i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float si = 1.0f / sqrtf((float)(var / n) + eps);
    for (i = 0; i < n; i++) out[i] = (x[i] - mean) * si;
    *std_inv_out = si;
}

/* KANN-style backward: efficient closed form */
static void layernorm_backward(const float *upstream, 
                                const float *normed, 
                                float std_inv, 
                                float *grad_in, int n)
{
    int i;
    double sum_g = 0, sum_gx = 0;

    /* Compute two dot products */
    for (i = 0; i < n; i++) {
        sum_g += upstream[i];
        sum_gx += normed[i] * upstream[i];
    }
    float mean_g = (float)(sum_g / n);
    float mean_gx = (float)(sum_gx / n);

    /* grad_in = std_inv * (upstream
         - mean(upstream)
         - normed * mean(normed * upstream)) */
    for (i = 0; i < n; i++)
        grad_in[i] = std_inv * (upstream[i] - mean_g
                     - normed[i] * mean_gx);
}

int main(void)
{
    float x[DIM] = { 1.0f, 3.0f, 2.0f, 5.0f, 4.0f };
    float normed[DIM], grad_in[DIM];
    float std_inv;
    float upstream[DIM] = 
        { 0.1f, -0.2f, 0.3f, -0.1f, 0.2f };
    int i;

    layernorm_forward(x, normed, DIM, &std_inv);

    printf("Input:      [");
    for (i = 0; i < DIM; i++)
        printf("%.1f%s", x[i], i<DIM-1?", ":"");
    printf("]\n");
    printf("Normalized: [");
    for (i = 0; i < DIM; i++)
        printf("%+.4f%s", normed[i], i<DIM-1?", ":"");
    printf("]\n");
    printf("Upstream:   [");
    for (i = 0; i < DIM; i++)
        printf("%+.1f%s", upstream[i],
               i<DIM-1?", ":"");
    printf("]\n\n");

    layernorm_backward(upstream, normed, std_inv, 
                       grad_in, DIM);

    printf("Gradient:   [");
    for (i = 0; i < DIM; i++)
        printf("%+.4f%s", grad_in[i],
               i<DIM-1?", ":"");
    printf("]\n");

    /* Verify with numerical gradient */
    printf("\nNumerical verification:\n");
    float h = 0.0001f;
    for (i = 0; i < DIM; i++) {
        float x_plus[DIM], x_minus[DIM];
        float out_p[DIM], out_m[DIM], si;
        int j;
        for (j = 0; j < DIM; j++)
            x_plus[j] = x_minus[j] = x[j];
        x_plus[i] += h;
        x_minus[i] -= h;
        layernorm_forward(x_plus, out_p, DIM, &si);
        layernorm_forward(x_minus, out_m, DIM, &si);

        float num_grad = 0;
        for (j = 0; j < DIM; j++)
            num_grad += upstream[j]
                        * (out_p[j] - out_m[j]) / (2*h);

        printf("  dim %d: analytical=%+.4f  "
               "numerical=%+.4f  match=%s\n",
               i, grad_in[i], num_grad, 
               fabsf(grad_in[i] - num_grad) < 0.001f
                   ? "yes" : "NO");
    }

    return 0;
}
