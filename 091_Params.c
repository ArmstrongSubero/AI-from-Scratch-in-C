/* 091_Params.c */
#include <stdio.h>

int main(void)
{
    int n = 64;   /* hidden size */
    int m = 32;   /* input size */

    /* W_x + W_h + b */
    int rnn_params = n*m + n*n + n;
    /* z, r and the candidate s */
    int gru_params = 3 * (n*m + n*n + n);
    /* f, i, o and the candidate g */
    int lstm_params = 4 * (n*m + n*n + n);

    printf("Parameter count comparison "
           "(hidden=%d, input=%d):\n\n", n, m);
    printf("  Architecture   Gate sets   Parameters"
           "   Ratio\n");
    printf("  ------------   ---------   ----------"
           "   -----\n");
    printf("  Basic RNN      1           %6d       "
           "1.0x\n", rnn_params);
    printf("  GRU            3           %6d       "
           "%.1fx\n", gru_params,
           (float)gru_params/rnn_params);
    printf("  LSTM           4           %6d       "
           "%.1fx\n", lstm_params,
           (float)lstm_params/rnn_params);

    printf("\n  GRU uses 75%% of LSTM's parameters.\n");
    printf("  On many tasks, GRU matches LSTM "
           "performance.\n");
    printf("  KANN uses GRU for its rnn-bit "
           "example.\n");

    /* multiplies per step, approximately */
    int rnn_per_step = n*m + n*n;
    int gru_per_step = 3 * (n*m + n*n);
    int lstm_per_step = 4 * (n*m + n*n);

    printf("\n  Compute per step (multiplies, "
           "approx):\n");
    printf("    RNN:  %6d\n", rnn_per_step);
    printf("    GRU:  %6d (%.1fx RNN)\n",
           gru_per_step, 
           (float)gru_per_step/rnn_per_step);
    printf("    LSTM: %6d (%.1fx RNN)\n",
           lstm_per_step, 
           (float)lstm_per_step/rnn_per_step);

    return 0;
}
