/* 124_Pre_Post.c */
#include <stdio.h>
#include <math.h>

#define DIM 4

static void rmsnorm(const float *x, float *out, int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    s = sqrtf(s / n + 1e-6f);
    for (i = 0; i < n; i++) out[i] = x[i] / s;
}

/* A stand-in sublayer. Any fixed function will do,
   what matters is where the norm sits around it. */
static void sublayer(const float *x, float *out, int n)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = 0.5f * x[i] + 0.1f;
}

/* Run depth layers and return the output */
static void run(const float *x0, int depth, int pre, 
                float *out)
{
    float h[DIM], t[DIM], n[DIM];
    int d, i;
    for (i = 0; i < DIM; i++) h[i] = x0[i];
    for (d = 0; d < depth; d++) {
        if (pre) {
            /* x + Sublayer(Norm(x)) */
            rmsnorm(h, n, DIM);
            sublayer(n, t, DIM);
            for (i = 0; i < DIM; i++)
                h[i] = h[i] + t[i];
        }
        else {
            /* Norm(x + Sublayer(x)) */
            sublayer(h, t, DIM);
            for (i = 0; i < DIM; i++)
                t[i] = h[i] + t[i];
            rmsnorm(t, h, DIM);
        }
    }
    for (i = 0; i < DIM; i++) out[i] = h[i];
}

static float norm_of(const float *v, int n)
{
    float s = 0;
    int i;
    for (i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

/* How much does the output move when input 0 moves?
   This is the gradient the first layer would
   receive. */
static float sensitivity(const float *x0, int depth, 
                         int pre)
{
    float a[DIM], b[DIM], xp[DIM], d[DIM];
    float eps = 1e-3f;
    int i;
    for (i = 0; i < DIM; i++) xp[i] = x0[i];
    xp[0] += eps;
    run(x0, depth, pre, a);
    run(xp, depth, pre, b);
    for (i = 0; i < DIM; i++)
        d[i] = (b[i] - a[i]) / eps;
    return norm_of(d, DIM);
}

int main(void)
{
    float x0[DIM] = { 1.0f, 2.0f, 3.0f, 4.0f };
    int depths[] = { 1, 2, 4, 8, 16, 32 };
    float o[DIM];
    int k;

    printf("Post-norm  Norm(x + Sublayer(x))\n");
    printf("Pre-norm   x + Sublayer(Norm(x))\n\n");

    printf("           post-norm         pre-norm\n");
    printf("  depth    |out|   dout/dx   |out|   "
           "dout/dx\n");
    printf("  -----    -----   -------   -----   "
           "-------\n");
    for (k = 0; k < 6; k++) {
        int dpt = depths[k];
        float npost, npre, spost, spre;
        run(x0, dpt, 0, o);
        npost = norm_of(o, DIM);
        run(x0, dpt, 1, o);
        npre = norm_of(o, DIM);
        spost = sensitivity(x0, dpt, 0);
        spre = sensitivity(x0, dpt, 1);
        printf("  %5d  %7.3f  %8.5f %7.3f  %8.3f\n",
               dpt, npost, spost, npre, spre);
    }

    printf("\nThe dout/dx columns are what matter. "
           "They\n");
    printf("say how much the output responds to a\n");
    printf("nudge at the input, the gradient the\n");
    printf("first layer would receive.\n\n");

    printf("Post-norm loses it with depth, since\n");
    printf("every layer divides by the accumulated\n");
    printf("norm and those divisions compound.\n");
    printf("Pre-norm holds, because the residual\n");
    printf("path is a plain addition whose\n");
    printf("derivative is 1 at any depth.\n\n");

    printf("Post-norm: original transformer, 2017.\n");
    printf("Pre-norm:  GPT, LLaMA, DeepSeek-V3.\n");

    return 0;
}
