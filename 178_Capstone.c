/* 178_Capstone.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#define VOCAB 28
#define D_MODEL 32
#define N_HEADS 4
#define D_HEAD 8
#define D_FF 128
#define N_LAYERS 2
#define MAX_SEQ 64

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

static float gelu(float x)
{
    float u = 0.7978846f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}

static float gelu_grad(float x)
{
    float a = 0.044715f, c = 0.7978846f;
    float u = c * (x + a * x * x * x);
    float t = tanhf(u);
    return 0.5f * (1.0f + t)
         + 0.5f * x * (1.0f - t * t) * c
           * (1.0f + 3.0f * a * x * x);
}

static int char_to_id(char c)
{
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c == ' ') return 26;
    return 27;
}

static char id_to_char(int id)
{
    if (id < 26) return 'a' + id;
    if (id == 26) return ' ';
    return '.';
}

static void softmax(float *x, int n)
{
    int i;
    float mx = x[0], s = 0;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

/* ---- Parameters ---- */

typedef struct {
    float tok_emb[VOCAB][D_MODEL];
    float pos_emb[MAX_SEQ][D_MODEL];
    float W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
    float W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
    float W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
    float W_O[N_LAYERS][D_MODEL][D_MODEL];
    float W_up[N_LAYERS][D_FF][D_MODEL];
    float W_down[N_LAYERS][D_MODEL][D_FF];
    float norm1_g[N_LAYERS][D_MODEL];
    float norm2_g[N_LAYERS][D_MODEL];
    float norm_f_g[D_MODEL];
}
GPT;

#define NPARAM (sizeof(GPT) / sizeof(float))

/* ---- Activation cache, one per forward pass ---- */

typedef struct {
    int T;
    int tok[MAX_SEQ];
    /* embeddings */
    float X0[MAX_SEQ][D_MODEL];
    /* block input */
    float Xa[N_LAYERS][MAX_SEQ][D_MODEL];
    /* normed */
    float n1[N_LAYERS][MAX_SEQ][D_MODEL];
    /* 1/rms */
    float s1[N_LAYERS][MAX_SEQ];
    float q[N_LAYERS][N_HEADS][MAX_SEQ][D_HEAD];
    float k[N_LAYERS][N_HEADS][MAX_SEQ][D_HEAD];
    float v[N_LAYERS][N_HEADS][MAX_SEQ][D_HEAD];
    /* post softmax */
    float p[N_LAYERS][N_HEADS][MAX_SEQ][MAX_SEQ];
    /* concat heads */
    float cc[N_LAYERS][MAX_SEQ][D_MODEL];
    /* after attn add */
    float Xb[N_LAYERS][MAX_SEQ][D_MODEL];
    float n2[N_LAYERS][MAX_SEQ][D_MODEL];
    float s2[N_LAYERS][MAX_SEQ];
    /* pre GELU */
    float hf[N_LAYERS][MAX_SEQ][D_FF];
    /* post GELU */
    float gf[N_LAYERS][MAX_SEQ][D_FF];
    /* after last block */
    float Xc[MAX_SEQ][D_MODEL];
    float sf[MAX_SEQ];
    /* final normed */
    float Xf[MAX_SEQ][D_MODEL];
    float logit[MAX_SEQ][VOCAB];
}
Cache;

static void gpt_init(GPT *m)
{
    int i, l, d;
    float *w = (float *)m;

    for (i = 0; i < (int)NPARAM; i++)
        w[i] = (randf() * 2 - 1) * 0.08f;

    for (l = 0; l < N_LAYERS; l++)
        for (d = 0; d < D_MODEL; d++) {
            m->norm1_g[l][d] = 1.0f;
            m->norm2_g[l][d] = 1.0f;
        }
    for (d = 0; d < D_MODEL; d++)
        m->norm_f_g[d] = 1.0f;
}

/* y = g * x / rms, returns 1/rms */
static float rmsnorm(const float *x, const float *g, 
                     float *y, int n)
{
    int i;
    float ss = 0;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / n + 1e-5f);
    for (i = 0; i < n; i++) y[i] = g[i] * x[i] * s;
    return s;
}

/* dx from dy, accumulating dg */
static void rmsnorm_back(const float *x, 
    const float *g, 
                         float s, const float *dy, 
                         float *dx, float *dg, int n)
{
    int i;
    float dot = 0;
    for (i = 0; i < n; i++) {
        float xhat = x[i] * s;
        dg[i] += dy[i] * xhat;
        dot += dy[i] * g[i] * xhat;
    }
    dot /= n;
    for (i = 0; i < n; i++)
        dx[i] += s * (dy[i] * g[i] - x[i] * s * dot);
}

static void forward(const GPT *m, 
    const int *tok, int T, 
                    Cache *c)
{
    int i, j, l, h, d, e;
    float X[MAX_SEQ][D_MODEL];

    c->T = T;
    for (i = 0; i < T; i++) c->tok[i] = tok[i];

    for (i = 0; i < T; i++)
        for (d = 0; d < D_MODEL; d++)
            X[i][d] = c->X0[i][d] = 
                m->tok_emb[tok[i]][d] + m
                    ->pos_emb[i][d];

    for (l = 0; l < N_LAYERS; l++) {
        memcpy(c->Xa[l], X, sizeof(X));

        for (i = 0; i < T; i++)
            c->s1[l][i] = rmsnorm(X[i], m->norm1_g[l], 
                                  c->n1[l][i], D_MODEL);

        for (h = 0; h < N_HEADS; h++) {
            for (i = 0; i < T; i++)
                for (d = 0; d < D_HEAD; d++) {
                    float qa = 0, ka = 0, va = 0;
                    for (e = 0; e < D_MODEL; e++) {
                        float u = c->n1[l][i][e];
                        qa += m->W_Q[l][h][d][e] * u;
                        ka += m->W_K[l][h][d][e] * u;
                        va += m->W_V[l][h][d][e] * u;
                    }
                    c->q[l][h][i][d] = qa;
                    c->k[l][h][i][d] = ka;
                    c->v[l][h][i][d] = va;
                }

            float scale = 1.0f / sqrtf((float)D_HEAD);
            for (i = 0; i < T; i++) {
                float row[MAX_SEQ];
                for (j = 0; j <= i; j++) {
                    float s = 0;
                    for (d = 0; d < D_HEAD; d++)
                        s += c->q[l][h][i][d]
                           * c->k[l][h][j][d];
                    row[j] = s * scale;
                }
                softmax(row, i + 1);
                for (j = 0; j <= i; j++)
                    c->p[l][h][i][j] = row[j];
                for (j = i + 1; j < T; j++)
                    c->p[l][h][i][j] = 0;

                for (d = 0; d < D_HEAD; d++) {
                    float ctx = 0;
                    for (j = 0; j <= i; j++)
                        ctx += row[j] * c
                            ->v[l][h][j][d];
                    c->cc[l][i][h * D_HEAD + d] = ctx;
                }
            }
        }

        for (i = 0; i < T; i++)
            for (d = 0; d < D_MODEL; d++) {
                float o = 0;
                for (e = 0; e < D_MODEL; e++)
                    o += m->W_O[l][d][e] * c
                        ->cc[l][i][e];
                X[i][d] += o;
            }
        memcpy(c->Xb[l], X, sizeof(X));

        for (i = 0; i < T; i++)
            c->s2[l][i] = rmsnorm(X[i], m->norm2_g[l], 
                                  c->n2[l][i], D_MODEL);

        for (i = 0; i < T; i++) {
            for (d = 0; d < D_FF; d++) {
                float a = 0;
                for (e = 0; e < D_MODEL; e++)
                    a += m->W_up[l][d][e] * c
                        ->n2[l][i][e];
                c->hf[l][i][d] = a;
                c->gf[l][i][d] = gelu(a);
            }
            for (d = 0; d < D_MODEL; d++) {
                float o = 0;
                for (e = 0; e < D_FF; e++)
                    o += m->W_down[l][d][e]
                       * c->gf[l][i][e];
                X[i][d] += o;
            }
        }
    }

    memcpy(c->Xc, X, sizeof(X));
    for (i = 0; i < T; i++)
        c->sf[i] = rmsnorm(X[i], m->norm_f_g, c->Xf[i], 
                           D_MODEL);

    for (i = 0; i < T; i++)
        for (j = 0; j < VOCAB; j++) {
            float z = 0;
            for (d = 0; d < D_MODEL; d++)
                z += c->Xf[i][d] * m->tok_emb[j][d];
            c->logit[i][j] = z;
        }
}

static float loss_and_backward(const GPT *m, Cache *c, 
                               GPT *gr)
{
    int i, j, l, h, d, e;
    int T = c->T;
    float total = 0;
    static float dX[MAX_SEQ][D_MODEL];
    static float dXf[MAX_SEQ][D_MODEL];

    memset(dX, 0, sizeof(dX));
    memset(dXf, 0, sizeof(dXf));

    float inv = 1.0f / (T - 1);

    for (i = 0; i < T - 1; i++) {
        float pr[VOCAB];
        int tgt = c->tok[i + 1];
        memcpy(pr, c->logit[i], sizeof(pr));
        softmax(pr, VOCAB);
        total += -logf(pr[tgt] + 1e-9f);

        /* The loss is a mean, so the gradient is too */
        for (j = 0; j < VOCAB; j++) {
            float hot = (j == tgt) ? 1.0f : 0.0f;
            float dz = (pr[j] - hot) * inv;
            for (d = 0; d < D_MODEL; d++) {
                gr->tok_emb[j][d] += dz * c->Xf[i][d];
                dXf[i][d] += dz * m->tok_emb[j][d];
            }
        }
    }

    for (i = 0; i < T; i++)
        rmsnorm_back(c->Xc[i], m->norm_f_g, c->sf[i], 
                     dXf[i], dX[i], gr->norm_f_g, 
                     D_MODEL);

    for (l = N_LAYERS - 1; l >= 0; l--) {
        static float dn2[MAX_SEQ][D_MODEL];
        static float dn1[MAX_SEQ][D_MODEL];
        memset(dn2, 0, sizeof(dn2));
        memset(dn1, 0, sizeof(dn1));

        /* FFN */
        for (i = 0; i < T; i++) {
            float dg[D_FF] = {0};
            for (d = 0; d < D_MODEL; d++)
                for (e = 0; e < D_FF; e++) {
                    gr->W_down[l][d][e] += dX[i][d]
                          * c->gf[l][i][e];
                    dg[e] += m
                        ->W_down[l][d][e] * dX[i][d];
                }
            for (e = 0; e < D_FF; e++) {
                float da = dg[e]
                         * gelu_grad(c->hf[l][i][e]);
                for (d = 0; d < D_MODEL; d++) {
                    gr->W_up[l][e][d] += da
                                       * c->n2[l][i][d];
                    dn2[i][d] += m->W_up[l][e][d] * da;
                }
            }
        }
        for (i = 0; i < T; i++)
            rmsnorm_back(c->Xb[l][i], m->norm2_g[l], 
                         c->s2[l][i], dn2[i], dX[i], 
                         gr->norm2_g[l], D_MODEL);

        /* Attention */
        static float dcc[MAX_SEQ][D_MODEL];
        memset(dcc, 0, sizeof(dcc));
        for (i = 0; i < T; i++)
            for (d = 0; d < D_MODEL; d++)
                for (e = 0; e < D_MODEL; e++) {
                    gr->W_O[l][d][e] += dX[i][d]
                                      * c->cc[l][i][e];
                    dcc[i][e] += m
                        ->W_O[l][d][e] * dX[i][d];
                }

        float scale = 1.0f / sqrtf((float)D_HEAD);
        for (h = 0; h < N_HEADS; h++) {
            static float dq[MAX_SEQ][D_HEAD];
            static float dk[MAX_SEQ][D_HEAD];
            static float dv[MAX_SEQ][D_HEAD];
            memset(dq, 0, sizeof(dq));
            memset(dk, 0, sizeof(dk));
            memset(dv, 0, sizeof(dv));

            for (i = 0; i < T; i++) {
                float dp[MAX_SEQ] = {0}, ds[MAX_SEQ];
                for (j = 0; j <= i; j++) {
                    float acc = 0;
                    for (d = 0; d < D_HEAD; d++) {
                        acc += dcc[i][h * D_HEAD + d]
                             * c->v[l][h][j][d];
                        dv[j][d] += c->p[l][h][i][j]
                      * dcc[i][h * D_HEAD + d];
                    }
                    dp[j] = acc;
                }
                float dot = 0;
                for (j = 0; j <= i; j++)
                    dot += c->p[l][h][i][j] * dp[j];
                for (j = 0; j <= i; j++)
                    ds[j] = c->p[l][h][i][j]
                          * (dp[j] - dot) * scale;
                for (j = 0; j <= i; j++)
                    for (d = 0; d < D_HEAD; d++) {
                        dq[i][d] += ds[j]
                                  * c->k[l][h][j][d];
                        dk[j][d] += ds[j]
                                  * c->q[l][h][i][d];
                    }
            }

            for (i = 0; i < T; i++)
                for (d = 0; d < D_HEAD; d++)
                    for (e = 0; e < D_MODEL; e++) {
                        float u = c->n1[l][i][e];
                        gr->W_Q[l][h][d][e]
                                += dq[i][d] * u;
                        gr->W_K[l][h][d][e]
                                += dk[i][d] * u;
                        gr->W_V[l][h][d][e]
                                += dv[i][d] * u;
                        dn1[i][e] += 
                  m->W_Q[l][h][d][e] * dq[i][d]
                + m->W_K[l][h][d][e] * dk[i][d]
                + m->W_V[l][h][d][e] * dv[i][d];
                    }
        }

        for (i = 0; i < T; i++)
            rmsnorm_back(c->Xa[l][i], m->norm1_g[l], 
                         c->s1[l][i], dn1[i], dX[i], 
                         gr->norm1_g[l], D_MODEL);
    }

    for (i = 0; i < T; i++)
        for (d = 0; d < D_MODEL; d++) {
            gr->tok_emb[c->tok[i]][d] += dX[i][d];
            gr->pos_emb[i][d] += dX[i][d];
        }

    return total / (T - 1);
}

static double loss_only(const GPT *m, const int *tok, 
                        int T, Cache *c)
{
    int i;
    double total = 0;
    forward(m, tok, T, c);
    for (i = 0; i < T - 1; i++) {
        float pr[VOCAB];
        memcpy(pr, c->logit[i], sizeof(pr));
        softmax(pr, VOCAB);
        total += -logf(pr[c->tok[i + 1]] + 1e-9f);
    }
    return total / (T - 1);
}

/* ---- Adam ---- */

static GPT Grad, Mom, Vel;

static void adam_step(GPT *m, float lr, int t)
{
    float *w = (float *)m, *g = (float *)&Grad;
    float *u = (float *)&Mom, *v = (float *)&Vel;
    float b1 = 0.9f, b2 = 0.999f;
    float c1 = 1.0f - powf(b1, t);
    float c2 = 1.0f - powf(b2, t);
    int i;

    for (i = 0; i < (int)NPARAM; i++) {
        u[i] = b1 * u[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        w[i] -= lr * (u[i] / c1)
              / (sqrtf(v[i] / c2) + 1e-8f);
    }
}

/* ---- Gradient check ----
   Perturb one whole block along a random direction and
   compare the measured change in loss against the
   analytic
   gradient projected onto that direction. Averaging
   over a
   direction removes the float32 noise that ruins single
   element checks. */

static double check_block(GPT *m, 
    const int *tok, int T, 
                          Cache *c, int lo, int hi, 
                          double *out_num)
{
    static float dir[NPARAM];
    float *w = (float *)m, *g = (float *)&Grad;
    double ana = 0, eps = 1e-3, Lp, Lm;
    int i;

    for (i = lo; i < hi; i++) {
        dir[i] = randf() * 2 - 1;
        ana += (double)g[i] * dir[i];
    }
    for (i = lo; i < hi; i++) w[i] += eps * dir[i];
    Lp = loss_only(m, tok, T, c);
    for (i = lo; i < hi; i++) w[i] -= 2 * eps * dir[i];
    Lm = loss_only(m, tok, T, c);
    for (i = lo; i < hi; i++) w[i] += eps * dir[i];

    *out_num = (Lp - Lm) / (2 * eps);
    return ana;
}

/* ---- Corpus and training ---- */

static float unigram_entropy(const int *tok, int n)
{
    int count[VOCAB] = {0};
    float h = 0;
    int i;

    for (i = 0; i < n; i++) count[tok[i]]++;
    for (i = 0; i < VOCAB; i++)
        if (count[i]) {
            float p = (float)count[i] / n;
            h -= p * logf(p);
        }
    return h;
}

#define WINDOW 48
#define STRIDE 16

int main(void)
{
    static GPT model;
    static Cache c;
    static int tok[512];
    const char *names[] = {
        "tok_emb", "pos_emb", "W_Q", "W_K",
            "W_V", "W_O",
        "W_up", "W_down", "norm1_g", "norm2_g",
        "norm_f_g" };
    size_t off[] = {
        offsetof(GPT, tok_emb), offsetof(GPT, pos_emb), 
        offsetof(GPT, W_Q), offsetof(GPT, W_K), 
        offsetof(GPT, W_V), offsetof(GPT, W_O), 
        offsetof(GPT, W_up), offsetof(GPT, W_down), 
        offsetof(GPT, norm1_g), offsetof(GPT, norm2_g), 
        offsetof(GPT, norm_f_g), sizeof(GPT) };
    const char *corpus = 
        "the cat sat on the mat. "
        "the dog ran in the park. "
        "a big cat sat on a big mat. the small dog "
        "ran fast. "
        "the cat and the dog sat on the mat. "
        "a cat ran. "
        "the big dog sat. a small cat ran in the park. "
        "the dog and the cat ran fast. a big dog sat "
        "on a mat. ";
    int n = 0, i, b, e, s, step = 0, g;
    double worst = 0;

    for (i = 0; corpus[i] && n < 512; i++)
        tok[n++] = char_to_id(corpus[i]);

    srand(42);
    gpt_init(&model);

    printf("=== Capstone: a complete GPT ===\n\n");
    printf("  vocab %d  d_model %d  heads %d\n",
           VOCAB, D_MODEL, N_HEADS);
    printf("  d_ff %d  layers %d\n", D_FF, N_LAYERS);
    printf("  parameters %d   corpus %d characters\n",
           (int)NPARAM, n);
    printf("  uniform guess scores %.3f\n",
        logf(VOCAB));
    printf("  character frequency scores %.3f\n\n",
           unigram_entropy(tok, n));

    /* Verify every gradient before trusting any of
       them */
    memset(&Grad, 0, sizeof(Grad));
    forward(&model, tok, WINDOW, &c);
    loss_and_backward(&model, &c, &Grad);

    printf("  Gradient check, "
           "analytic vs numerical:\n");
    printf("    block       analytic    numerical  "
           "err\n");
    for (b = 0; b < 11; b++) {
        double num;
        double ana = check_block(&model, 
            tok, WINDOW, &c, 
                                 off[b]
                                     / sizeof(float), 
                                 off[b + 1]
                                     / sizeof(float), 
                                 &num);
        double rel = fabs(num - ana)
                   / (fabs(num) + fabs(ana) + 1e-12);
        if (rel > worst) worst = rel;
        printf("    %-9s %11.6f %11.6f %8.1e\n",
               names[b], ana, num, rel);
    }
    printf("    worst relative error %.1e against a\n",
           worst);
    printf("    tolerance of 5e-3, the limit of "
           "float32\n");
    printf("    forward passes allow. %s\n\n",
           worst < 5e-3 ? "Every gradient agrees."
                        : "A gradient is wrong.");

    /* Train every weight in the model */
    memset(&Mom, 0, sizeof(Mom));
    memset(&Vel, 0, sizeof(Vel));
    printf("  Training all %d parameters:\n",
        (int)NPARAM);
    for (e = 1; e <= 100; e++) {
        double total = 0;
        int nb = 0;
        for (s = 0; s + WINDOW < n; s += STRIDE) {
            memset(&Grad, 0, sizeof(Grad));
            forward(&model, tok + s, WINDOW, &c);
            total
                    += loss_and_backward(&model, &c,
                                         &Grad);
            adam_step(&model, 0.01f, ++step);
            nb++;
        }
        if (e == 1 || e % 20 == 0)
            printf("    epoch %3d   loss %.4f\n",
                   e, total / nb);
    }

    printf("\n  Generation (temperature 0.8):\n");
    for (g = 0; g < 3; g++) {
        int seq[MAX_SEQ], len = 4;
        const char *p = (g == 1) ? "a bi" : "the ";
        for (i = 0; i < 4; i++)
            seq[i] = char_to_id(p[i]);
        printf("    %s", p);
        for (i = 0; i < WINDOW - 4; i++) {
            float pr[VOCAB], r, acc = 0;
            int pick = VOCAB - 1, j;
            forward(&model, seq, len, &c);
            for (j = 0; j < VOCAB; j++)
                pr[j] = c.logit[len - 1][j] / 0.8f;
            softmax(pr, VOCAB);
            r = randf();
            for (j = 0; j < VOCAB; j++) {
                acc += pr[j];
                if (r < acc) {
                    pick = j;
                    break;
                    }
            }
            putchar(id_to_char(pick));
            seq[len++] = pick;
        }
        printf("\n");
    }

    printf("\n  Every component, and its chapter:\n");
    printf("    Token embedding          Ch. 10\n");
    printf("    Position embedding       Ch. 22\n");
    printf("    RMSNorm and its backward Ch. 23\n");
    printf("    Multi-head attention     Ch. 21\n");
    printf("    Causal mask              Ch. 25\n");
    printf("    GELU and its derivative  Ch. 24\n");
    printf("    Feed-forward network     Ch. 24\n");
    printf("    Residual connections     Ch. 24\n");
    printf("    Weight tying             Ch. 26\n");
    printf("    Softmax, cross entropy   Ch. 4\n");
    printf("    Backpropagation          Ch. 5\n");
    printf("    Adam                     Ch. 8\n");
    printf("    Temperature sampling     Ch. 26\n");

    return 0;
}
