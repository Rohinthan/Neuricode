#include "tensor.h"
#include <stdio.h>
#include <math.h>

static int close(float a, float b) { return fabsf(a-b) < 1e-4f; }

int main(void) {
    /* ---- Test A: simple scalar chain: L = sum(a*b) ---- */
    int shape2[1] = {3};
    Tensor *a = tensor_create(shape2, 1);
    Tensor *b = tensor_create(shape2, 1);
    a->data[0]=2; a->data[1]=3; a->data[2]=4;
    b->data[0]=5; b->data[1]=6; b->data[2]=7;
    tensor_requires_grad_(a, 1);
    tensor_requires_grad_(b, 1);

    Tensor *prod = tensor_create(shape2, 1);
    tensor_mul(a, b, prod);       /* prod = a*b, tracked */

    /* scalar-ize via reshape (view!) into a same-size [3] -> keep same
     * shape, then use matmul with a ones[3,1] vector to reduce to a
     * true scalar while staying inside the tape (matmul is tracked). */
    int ones_shape[2] = {3,1};
    Tensor *ones_col = tensor_create(ones_shape, 2);
    tensor_fill(ones_col, 1.0f);

    int row_shape[2] = {1,3};
    Tensor *prod_row = tensor_reshape(prod, row_shape, 2); /* view of prod */

    int out_shape[2] = {1,1};
    Tensor *loss = tensor_create(out_shape, 2);
    tensor_matmul(prod_row, ones_col, loss);  /* loss = sum(a*b), scalar [1,1] */

    tensor_backward(loss);

    printf("[Test A] loss=%.4f (expect 2*5+3*6+4*7=56)\n", loss->data[0]);
    printf("  a.grad = [%.2f %.2f %.2f] (expect b = [5 6 7])\n", a->grad[0], a->grad[1], a->grad[2]);
    printf("  b.grad = [%.2f %.2f %.2f] (expect a = [2 3 4])\n", b->grad[0], b->grad[1], b->grad[2]);

    int ok = close(loss->data[0],56.0f) && close(a->grad[0],5) && close(a->grad[1],6) &&
              close(a->grad[2],7) && close(b->grad[0],2) && close(b->grad[1],3) && close(b->grad[2],4);
    printf("  %s\n", ok ? "PASS" : "FAIL");

    /* ---- Test B: gradient through a VIEW (prod_row) lands on `prod`'s
     * grad buffer, not some orphaned buffer on the view itself. ---- */
    printf("[Test B] prod_row->grad (view, should stay NULL) = %p\n", (void*)prod_row->grad);
    printf("          prod->grad (true owner) = [%.2f %.2f %.2f] (expect ones, since d(matmul)/d(prod)=ones_col row)\n",
           prod->grad[0], prod->grad[1], prod->grad[2]);
    int okB = (prod_row->grad == NULL) && close(prod->grad[0],1) && close(prod->grad[1],1) && close(prod->grad[2],1);
    printf("  %s\n", okB ? "PASS" : "FAIL");

    tensor_tape_clear();

    /* ---- Test C: permute forward + backward round trip ---- */
    int bsf_shape[3] = {2,3,4}; /* [Batch=2, Seq=3, Feat=4] */
    Tensor *x = tensor_create(bsf_shape, 3);
    for (size_t i = 0; i < x->size; i++) x->data[i] = (float)i;
    tensor_requires_grad_(x, 1);

    int axis_order[3] = {1,0,2}; /* -> [Seq=3, Batch=2, Feat=4] */
    Tensor *xp = tensor_permute(x, axis_order, 3);
    printf("[Test C] permuted shape = [%d,%d,%d] (expect 3,2,4)\n", xp->shape[0], xp->shape[1], xp->shape[2]);

    /* spot check: x[b=1,s=2,f=3] should equal xp[s=2,b=1,f=3] */
    int idx_x[3]  = {1,2,3};
    int idx_xp[3] = {2,1,3};
    float vx  = tensor_get(x, idx_x);
    float vxp = tensor_get(xp, idx_xp);
    printf("  x[1,2,3]=%.1f  xp[2,1,3]=%.1f\n", vx, vxp);

    /* backward: fake an upstream grad of all-ones on xp, check it lands
     * back on x with the same total (permute doesn't touch values). */
    int ones_shape3[3] = {3,2,4};
    Tensor *g_out = tensor_create(ones_shape3, 3);
    tensor_fill(g_out, 1.0f);
    Tensor *g_in = tensor_zeros(bsf_shape, 3);
    tensor_permute_backward(g_out, axis_order, g_in);
    float total = tensor_sum(g_in);
    printf("  sum(g_in) = %.1f (expect 24 = numel)\n", total);

    int okC = close(vx, vxp) && close(total, 24.0f) &&
              xp->shape[0]==3 && xp->shape[1]==2 && xp->shape[2]==4;
    printf("  %s\n", okC ? "PASS" : "FAIL");

    printf("\nOVERALL: %s\n", (ok && okB && okC) ? "ALL PASS" : "SOME FAILED");
    return (ok && okB && okC) ? 0 : 1;
}
