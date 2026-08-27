/* 031_Reusable_Optimizer.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float *m;       /* first moment (mean of gradients) */
    float *v;
    /* second moment (mean of squared gradients) */
    float beta1;
    float beta2;
    float eps;
    float lr;
    float b1t;      /* beta1^t for bias correction */
    float b2t;      /* beta2^t for bias correction */
    int n;          /* number of parameters */
} Adam;

static Adam adam_create(int n, float lr)
{
    Adam opt;
    opt.m = (float *)calloc(n, sizeof(float));
    opt.v = (float *)calloc(n, sizeof(float));
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.lr = lr;
    opt.b1t = 1.0f;
    opt.b2t = 1.0f;
    opt.n = n;
    return opt;
}

static void adam_update(Adam *opt, float *params,
    const float *grads)
{
    int i;
    opt->b1t *= opt->beta1;
    opt->b2t *= opt->beta2;

    for (i = 0; i < opt->n; i++) {
        opt->m[i] = opt->beta1 * opt->m[i]
            + (1.0f - opt->beta1) * grads[i];
        opt->v[i] = opt->beta2 * opt->v[i]
            + (1.0f - opt->beta2) * grads[i] * grads[i];

        float m_hat = opt->m[i] / (1.0f - opt->b1t);
        float v_hat = opt->v[i] / (1.0f - opt->b2t);

        params[i] -= opt->lr * m_hat
            / (sqrtf(v_hat) + opt->eps);
    }
}

static void adam_free(Adam *opt)
{
    free(opt->m);
    free(opt->v);
    opt->m = NULL;
    opt->v = NULL;
}

/* --- Test on our 2D function --- */

int main(void)
{
    float params[2] = { 5.0f, 3.0f };
    /* x and y as a parameter array */
    float grads[2];
    Adam opt = adam_create(2, 0.1f);
    int i;

    printf("Adam optimizer module test\n");
    printf("  step   x        y        f(x,y)\n");
    for (i = 0; i < 50; i++) {
        float fval = params[0]*params[0] + 10.0f*params[1]
            *params[1];
        if (i % 10 == 0)
            printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", i,
                params[0], params[1], fval);

        grads[0] = 2.0f * params[0];
        grads[1] = 20.0f * params[1];
        adam_update(&opt, params, grads);
    }
    float fval = params[0]*params[0] + 10.0f*params[1]
        *params[1];
    printf("  %3d   %+7.4f  %+7.4f  %8.4f\n", 50, params[0],
        params[1], fval);

    adam_free(&opt);
    return 0;
}
