/* 174_Router.c */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

#define DM 6
#define N_EXPERTS 8
#define TOP_K 2
#define N_DOMAIN 3
#define PER_DOMAIN 10
#define N_TOKENS (N_DOMAIN * PER_DOMAIN)

/* Which domain each expert has specialised in */
static const int owner[N_EXPERTS] = { 0, 0, 0, 1, 
    1, 2, 2, 2 };

/* Three unit directions standing in for code, math
   and prose after the router has been trained */
static const float proto[N_DOMAIN][DM] = {
    { 0.707f, 0.707f, 0, 0, 0, 0 }, 
    { 0, 0, 0.707f, 0.707f, 0, 0 }, 
    { 0, 0, 0, 0, 0.707f, 0.707f }, 
};

/* Score every expert, select top K, normalise the
   gates, and report how far apart the scores were */
static float route(const float c[N_EXPERTS][DM], 
                   const float x[DM], int sel[TOP_K], 
                   float gate[TOP_K], float *raw)
{
    float s[N_EXPERTS], lo = 2, hi = -2, sum = 0;
    int i, j, k;

    for (i = 0; i < N_EXPERTS; i++) {
        float dot = 0;
        for (j = 0; j < DM; j++)
            dot += x[j] * c[i][j];
        s[i] = sigmoid(dot);
        if (s[i] < lo) lo = s[i];
        if (s[i] > hi) hi = s[i];
    }

    for (k = 0; k < TOP_K; k++) {
        int best = -1;
        float bs = -1;
        for (i = 0; i < N_EXPERTS; i++) {
            int used = 0;
            for (j = 0; j < k; j++)
                if (sel[j] == i) used = 1;
            if (!used && s[i] > bs) {
                best = i;
                bs = s[i];
                }
        }
        sel[k] = best;
        gate[k] = s[best];
    }

    *raw = gate[0];
    for (k = 0; k < TOP_K; k++) sum += gate[k];
    for (k = 0; k < TOP_K; k++) gate[k] /= sum;
    return hi - lo;
}

/* Run every token and count how many land on an
   expert that owns the token's own domain */
static void evaluate(const char *label, 
                     const float c[N_EXPERTS][DM], 
                     const float tok[N_TOKENS][DM], 
            const int dom[N_TOKENS],
            int verbose)
{
    float spread = 0, top_gate = 0, top_raw = 0;
    int hits = 0, t;

    printf("  %s\n", label);

    for (t = 0; t < N_TOKENS; t++) {
        int sel[TOP_K];
        float gate[TOP_K], raw;
        spread += route(c, tok[t], sel, gate, &raw);
        top_gate += gate[0];
        top_raw += raw;
        if (owner[sel[0]] == dom[t]) hits++;
        if (verbose && t % PER_DOMAIN < 2)
            printf("    dom %d -> [%d, %d]  score %.3f"
                   "  gates %.3f %.3f\n",
                   dom[t], sel[0], sel[1], raw, 
                   gate[0], gate[1]);
    }

    printf("    mean score spread    %.3f\n",
           spread / N_TOKENS);
    printf("    mean winning score   %.3f\n",
           top_raw / N_TOKENS);
    printf("    mean winning gate    %.3f\n",
           top_gate / N_TOKENS);
    printf("    routed to own domain %d of %d\n\n",
           hits, N_TOKENS);
}

int main(void)
{
    float rnd[N_EXPERTS][DM], trained[N_EXPERTS][DM];
    float tok[N_TOKENS][DM];
    int dom[N_TOKENS];
    int i, j, t;

    srand(42);

    /* An untrained router: centroids are noise */
    for (i = 0; i < N_EXPERTS; i++)
        for (j = 0; j < DM; j++)
            rnd[i][j] = (randf()*2-1) * 0.3f;

    /* A trained router: each centroid sits on the
       prototype of the domain that expert owns */
    for (i = 0; i < N_EXPERTS; i++)
        for (j = 0; j < DM; j++)
            trained[i][j] = 2.5f * proto[owner[i]][j]
                          + (randf()*2-1) * 0.4f;

    /* Tokens are prototypes plus noise */
    for (t = 0; t < N_TOKENS; t++) {
        dom[t] = t / PER_DOMAIN;
        for (j = 0; j < DM; j++)
            tok[t][j] = proto[dom[t]][j]
                      + (randf()*2-1) * 0.25f;
    }

    printf("Router: %d experts, top-%d, %d domains\n\n",
           N_EXPERTS, TOP_K, N_DOMAIN);

    evaluate("Untrained router", rnd, tok, dom, 0);
    evaluate("Trained router", trained, tok, dom, 1);

    printf("  Training moves the winning score, not "
           "the\n");
    printf("  gate. Normalising two close scores "
           "always\n");
    printf("  returns something near one half, so "
           "the\n");
    printf("  routing decision carries the "
           "information\n");
    printf("  and the gate barely varies at all.\n");

    return 0;
}
