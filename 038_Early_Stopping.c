/* 038_Early_Stopping.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

#define N_HID 20
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
    float delta_out = -2.0f * (t - y) * y * (1.0f - y);
    for (i = 0; i < N_PARAMS; i++) g[i] = 0.0f;
    for (i = 0; i < N_HID; i++)
        g[bw+i] = delta_out * h[i];
    g[bw+N_HID] = delta_out;
    for (i = 0; i < N_HID; i++) {
        float dh = delta_out * p[bw+i] * h[i]
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
static void generate_data(Sample *d, int n) {
    int i;
    for(i = 0;i<n;i++) {
        d[i].x[0] = randf()*4
        -2;
        d[i].x[1] = randf()*4-2;
        float dist = d[i].x[0]*d[i].x[0]
            +d[i].x[1]*d[i].x[1];
        d[i].t = (dist+randf()*0.4f-0.2f<1.44f)
            ?1.0f
            :0.0f;
        }
        }
static float eval_loss(const float *p, 
    const Sample *d, int n) {
    float t = 0;
    int i;
    for(i = 0;i<n;i++) {
        float h[N_HID];
        float y = forward(p, d[i].x, h);
        float diff = d[i].t-y;
        t += diff*diff;
        }
        return t/n;
        }

int main(void)
{
    Sample data[200];
    float params[N_PARAMS], grads[N_PARAMS], 
        grad_acc[N_PARAMS];
    float best_params[N_PARAMS];
    Adam opt;
    float best_val = 1e30f;
    int best_epoch = 0;
    int patience = 50, streak = 0;
    int epoch, s, i;

    srand(42);
    generate_data(data, 200);
    for (i = 0; i < N_PARAMS; i++)
        params[i] = randf() * 0.4f - 0.2f;
    opt = adam_create(N_PARAMS, 0.001f);

    printf("Training with early stopping "
           "(patience=%d)\n", patience);
    printf("  epoch   train_loss  val_loss    best_val "
           "  status\n");
    for (epoch = 0; epoch < 1000; epoch++) {
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

        float vl = eval_loss(params, data + 160, 40);

        if (vl < best_val) {
            best_val = vl;
            best_epoch = epoch + 1;
            memcpy(best_params, params, sizeof(params));
            streak = 0;
        }
        else {
            streak++;
        }

        if ((epoch + 1) % 50 == 0) {
            float tl = eval_loss(params, data, 160);
            printf("  %4d    %.6f    %.6f    %.6f   "
                   "%s\n",
                   epoch + 1, tl, vl, best_val, 
            streak > 0 ? "waiting..."
                       : "new best");
        }

        if (streak >= patience) {
            printf("\n  Early stop at epoch %d. Best "
                   "was epoch %d.\n",
                   epoch + 1, best_epoch);
            break;
        }
    }

    /* Restore best weights */
    memcpy(params, best_params, sizeof(params));
    float final_tl = eval_loss(params, data, 160);
    float final_vl = eval_loss(params, data + 160, 40);
    printf("\n  Restored best weights (epoch "
           "%d)\n", best_epoch);
    printf("  Final train_loss=%.6f  val_loss=%.6f\n",
        final_tl, final_vl);

    adam_free(&opt);
    return 0;
}
