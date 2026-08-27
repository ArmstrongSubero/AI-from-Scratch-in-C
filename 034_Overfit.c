/* 034_Overfit.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* 2-input, 20-hidden, 1-output (intentionally
   oversized) */
#define N_HID 20
#define N_PARAMS (2*N_HID + N_HID + N_HID + 1)  /* wh
    + bh + wo + bo */

static float forward(const float *p, const float x[2], 
    float h[N_HID])
{
    int i, j;
    /* Hidden layer */
    for (i = 0; i < N_HID; i++) {
        float z = p[2 * N_HID + i];  /* bias */
        for (j = 0; j < 2; j++)
            z += p[i * 2 + j] * x[j];
        h[i] = sigmoid(z);
    }
    /* Output */
    {
        int base_wo = 3 * N_HID;
        float z = p[base_wo + N_HID];  /* output bias */
        for (i = 0; i < N_HID; i++)
            z += p[base_wo + i] * h[i];
        return sigmoid(z);
    }
}

static void backward(const float *p, const float x[2], 
    const float h[N_HID], 
   float y, float t, float *g)
{
    int i, j;
    int base_bh = 2 * N_HID;
    int base_wo = 3 * N_HID;
    float delta_out = -2.0f * (t - y) * y * (1.0f - y);

    for (i = 0; i < N_PARAMS; i++) g[i] = 0.0f;

    /* Output gradients */
    for (i = 0; i < N_HID; i++)
        g[base_wo + i] = delta_out * h[i];
    g[base_wo + N_HID] = delta_out;

    /* Hidden gradients */
    for (i = 0; i < N_HID; i++) {
        float dh = delta_out * p[base_wo + i] * h[i]
            * (1.0f - h[i]);
        for (j = 0; j < 2; j++)
            g[i * 2 + j] = dh * x[j];
        g[base_bh + i] = dh;
    }
}

typedef struct { float *m, *v; float b1, b2, eps, lr, 
    b1t, b2t;
    int n;
    }
    Adam;
static Adam adam_create(int n, float lr) {
    Adam o;
    o.m = (float*)calloc(n, sizeof(float));
    o.v = (float*)calloc(n, sizeof(float));
    o.b1 = 0.9f;
    o.b2 = 0.999f;
    o.eps = 1e-8f;
    o.lr = lr;
    o.b1t = 1;
    o.b2t = 1;
    o.n = n;
    return o;
    }
static void adam_update(Adam *o, float *p, 
    const float *g) {
    int i;
    o->b1t *= o->b1;
    o->b2t *= o->b2;
    for(i = 0;i<o->n;i++) {
        o->m[i] = o->b1*o->m[i]+(1-o->b1)*g[i];
        o->v[i] = o->b2*o->v[i]+(1-o->b2)*g[i]*g[i];
        float mh = o->m[i]/(1-o->b1t), 
            vh = o->v[i]/(1-o->b2t);
        p[i] -= o->lr*mh/(sqrtf(vh)+o->eps);
        }
        }
static void adam_free(Adam *o)
{
    free(o->m);
    free(o->v);
}

typedef struct { float x[2]; float t; } Sample;

static void generate_data(Sample *data, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        data[i].x[0] = randf() * 4.0f - 2.0f;
        data[i].x[1] = randf() * 4.0f - 2.0f;
        float dist = data[i].x[0]*data[i].x[0]
            + data[i].x[1]*data[i].x[1];
        float noise = randf() * 0.4f - 0.2f;
        data[i].t = (dist + noise < 1.44f)
            ? 1.0f
            : 0.0f;
    }
}

static float eval_loss(const float *p, 
    const Sample *data, int n)
{
    float total = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float h[N_HID];
        float y = forward(p, data[i].x, h);
        float d = data[i].t - y;
        total += d * d;
    }
    return total / n;
}

int main(void)
{
    Sample data[200];
    float params[N_PARAMS], grads[N_PARAMS], 
        grad_acc[N_PARAMS];
    Adam opt;
    int epoch, s, i;

    srand(42);
    generate_data(data, 200);

    /* Initialize params */
    for (i = 0; i < N_PARAMS; i++)
        params[i] = randf() * 0.4f - 0.2f;

    opt = adam_create(N_PARAMS, 0.001f);

    printf("Training 20-hidden network (oversized for "
           "200 samples)\n");
    printf("  epoch   train_loss  val_loss    gap\n");
    for (epoch = 0; epoch < 500; epoch++) {
        /* Train on first 160 samples */
        for (i = 0; i < N_PARAMS; i++)
            grad_acc[i] = 0.0f;
        for (s = 0; s < 160; s++) {
            float h[N_HID];
            float y = forward(params, data[s].x, h);
            backward(params, data[s].x, h, y, 
                data[s].t, grads);
            for (i = 0; i < N_PARAMS; i++)
                grad_acc[i] += grads[i];
        }
        for (i = 0; i < N_PARAMS; i++)
            grad_acc[i] /= 160.0f;
        adam_update(&opt, params, grad_acc);

        if ((epoch + 1) % 50 == 0) {
            float tl = eval_loss(params, data, 160);
            float vl = eval_loss(params, 
                data + 160, 40);
            printf("  %4d    %.6f    %.6f    %+.6f\n",
                   epoch + 1, tl, vl, vl - tl);
        }
    }

    adam_free(&opt);
    return 0;
}
