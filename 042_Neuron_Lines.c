/* 042_Neuron_Lines.c */
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

#define N_HID 4
#define N_PARAMS (2*N_HID + N_HID + N_HID + 1)

static float forward(const float *p, const float x[2], 
    float h[N_HID])
{
    int i, j;
    for (i = 0; i < N_HID; i++) {
        float z = p[2*N_HID + i];
        for (j = 0; j < 2; j++) z += p[i*2+j] * x[j];
        h[i] = sigmoid(z);
    }
    { int bw = 3*N_HID; float z = p[bw+N_HID];
      for (i = 0; i < N_HID; i++) z += p[bw+i] * h[i];
      return sigmoid(z);
      }
}

static void backward(const float *p, const float x[2], 
    const float h[N_HID], 
                      float y, float t, float *g)
{
    int i, j, bh = 2*N_HID, bw = 3*N_HID;
    float d_out = -2.0f * (t - y) * y * (1.0f - y);
    for (i = 0; i < N_PARAMS; i++) g[i] = 0;
    for (i = 0; i < N_HID; i++) g[bw+i] = d_out * h[i];
    g[bw+N_HID] = d_out;
    for (i = 0; i < N_HID; i++) {
        float dh = d_out * p[bw+i] * h[i]
            * (1.0f - h[i]);
        for (j = 0; j < 2; j++) g[i*2+j] = dh * x[j];
        g[bh+i] = dh;
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

int main(void)
{
    Sample data[200];
    float params[N_PARAMS], grads[N_PARAMS], 
        ga[N_PARAMS];
    Adam opt;
    int epoch, s, i;

    srand(42);
    for (i = 0; i < 200; i++) {
        data[i].x[0] = randf() * 4 - 2;
        data[i].x[1] = randf() * 4 - 2;
        float dist = data[i].x[0]*data[i].x[0]
            + data[i].x[1]*data[i].x[1];
        data[i].t = (dist < 1.44f) ? 1.0f : 0.0f;
    }
    for (i = 0; i < N_PARAMS; i++)
        params[i] = randf() * 0.4f - 0.2f;
    opt = adam_create(N_PARAMS, 0.01f);

    for (epoch = 0; epoch < 2000; epoch++) {
        for (i = 0; i < N_PARAMS; i++) ga[i] = 0;
        for (s = 0; s < 200; s++) {
            float h[N_HID], y;
            y = forward(params, data[s].x, h);
            backward(params, data[s].x, h, y, 
                data[s].t, grads);
            for (i = 0; i < N_PARAMS; i++)
                ga[i] += grads[i];
        }
        for (i = 0; i < N_PARAMS; i++) ga[i] /= 200.0f;
        adam_update(&opt, params, ga);
    }

    printf("Each hidden neuron defines a decision "
           "line:\n");
    printf("  w0*x0 + w1*x1 + b = 0\n\n");

    for (i = 0; i < N_HID; i++) {
        float w0 = params[i*2];
        float w1 = params[i*2+1];
        float b = params[2*N_HID+i];
        float angle = atan2f(w1, 
            w0) * 180.0f / 3.14159f;

        printf("  Neuron %d: %+.3f*x0 + %+.3f*x1 + "
               "%+.3f = 0\n",
               i, w0, w1, b);
        printf("            direction: "
               "%.0f degrees", angle);

        /* Output weight tells us if this neuron
           votes for or against */
        float ow = params[3*N_HID + i];
        printf("  output weight: %+.3f (%s)\n",
               ow, ow > 0 ? "votes IN" : "votes OUT");
    }

    /* Accuracy check */
    int correct = 0;
    for (s = 0; s < 200; s++) {
        float h[N_HID], y;
        y = forward(params, data[s].x, h);
        int pred = y > 0.5f ? 1 : 0;
        int actual = data[s].t > 0.5f ? 1 : 0;
        if (pred == actual) correct++;
    }
    printf("\nAccuracy: %d/200 (%.1f%%)\n", correct,
        correct / 2.0f);

    adam_free(&opt);
    return 0;
}
