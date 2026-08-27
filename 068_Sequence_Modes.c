/* 068_Sequence_Modes.c */
#include <stdio.h>

int main(void)
{
    /* Process a sequence one element at a time */
    int sequence[] = { 1, 0, 1, 1, 0, 
                       1, 0, 1, 1, 0 };
    int length = 10;
    int state = 0;  /* the "memory" */
    int t;

    printf("Sequential processing with state:\n\n");
    printf("  step  input  state\n");
    for (t = 0; t < length; t++) {
        state = state + sequence[t];   /* update rule */
        printf("  %3d     %d      %d\n", t,
               sequence[t], state);
    }
    printf("\n  Final state (count of 1s): "
           "%d\n", state);

    printf("\nThe key properties:\n");
    printf("  1. Process one input at a time\n");
    printf("  2. Maintain a state (memory) across "
           "steps\n");
    printf("  3. The update rule is the SAME at "
           "every step\n");
    printf("  4. Works for ANY sequence length\n");
    printf("\nThis is exactly what a recurrent "
           "neural network does.\n");
    printf("Replace the fixed update rule with "
           "a learned one.\n");

    return 0;
}
