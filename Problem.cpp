// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-11-05

/** C++ Example NLP for interfacing a problem with IPOPT.
 *  Problem implements a C++ example showing how to interface with IPOPT
 *  through the TNLP interface. This example is designed to go along with
 *  the tutorial document (see Examples/CppTutorial/).
 *  This class implements the following NLP.
 *
 * min_x f(x) = -(x2-2)^2
 *  s.t.
 *       g(x) = x1^2 + x2 - 1 = 0
 *       -1 <= x1 <= 1
 *
 *
 * -> f(x) = -(x2-2)^2
 *    g(x) = x1^2 + x2 - 1
 *    g_L = 0
 *    g_U = 0
 *    x_L = [-1 -inf]^T
 *    x_U = [1 inf]^T
 *  -----------------------
 *  grad f(x) = [0,  -2*(x2-2)]^T
 *
 */

#include "Problem.hpp"

#include <cassert>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

using namespace Ipopt;

/* Constructor. */
Problem::Problem() {}

Problem::~Problem() {}

bool Problem::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                           Index& nnz_h_lag, IndexStyleEnum& index_style) {
    // The problem described in Problem.hpp has 2 variables, x1, & x2,
    n = 2;

    // one equality constraint,
    m = 1;

    // 2 nonzeros in the jacobian (one for x1, and one for x2),
    nnz_jac_g = 2;

    // and 2 nonzeros in the hessian of the lagrangian
    // (one in the hessian of the objective for x2,
    //  and one in the hessian of the constraints for x1)
    nnz_h_lag = 2;

    // We use the standard c-style index style for row/col entries (0-based)
    index_style = C_STYLE;

    return true;
}

bool Problem::get_bounds_info(Index n, Number* x_l, Number* x_u, Index m,
                              Number* g_l, Number* g_u) {
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(n == 2);
    assert(m == 1);

    // x1 has a lower bound of -1 and an upper bound of 1
    x_l[0] = -1.0;
    x_u[0] = 1.0;

    // x2 has no upper or lower bound, so we set them to
    // a large negative and a large positive number.
    // The value that is interpretted as -/+infinity can be
    // set in the options, but it defaults to -/+1e19
    x_l[1] = -1.0e19;
    x_u[1] = +1.0e19;

    // we have one equality constraint, so we set the bounds on this constraint
    // to be equal (and zero).
    g_l[0] = g_u[0] = 0.0;

    return true;
}

bool Problem::get_starting_point(Index n, bool init_x, Number* x, bool init_z,
                                 Number* z_L, Number* z_U, Index m,
                                 bool init_lambda, Number* lambda) {
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the others if
    // you wish.
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // we initialize x in bounds, in the upper right quadrant
    x[0] = 0.5;
    x[1] = 1.5;

    return true;
}

bool Problem::eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
    // calculate the value of the objective function
    obj_value = -(x[1] - 2.0) * (x[1] - 2.0);
    return true;
}

bool Problem::eval_grad_f(Index n, const Number* x, bool new_x,
                          Number* grad_f) {
    // evaluate the gradient of the objective function
    // grad f(x) = [ 0,  -2*(x2-2) ]^T
    grad_f[0] = 0.0;
    grad_f[1] = -2.0 * (x[1] - 2.0);
    return true;
}

bool Problem::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
    // evaluate the constraint:
    g[0] = x[0] * x[0] + x[1] - 1.0;
    return true;
}

bool Problem::eval_jac_g(Index n, const Number* x, bool new_x, Index m,
                         Index nele_jac, Index* iRow, Index* jCol,
                         Number* values) {
    // either evaluate the jacobian of the constraint function g,
    // or set the sparsity structure of the jacobian:

    // Sei X ein 2D C-array. Ipopt's sparse matrix format expects
    // iRow, jCol and values, such that: X[ iRow[k] ][ jCol[k] ] = values[k]
    // bzw. X[ m * iRow[k] + jCol[k] ] = values[k].

    // See p.46 here:
    // https://projects.coin-or.org/Ipopt/browser/stable/3.10/Ipopt/doc/documentation.pdf?format=raw

    // Here, J_ij = derivative of g_i(x) w.r.t x_j
    // J_g(x) = [ 2*x[0],  1 ]^T

    //
    if (values == NULL) {
        // return the structure of the jacobian of the constraints
        iRow[0] = 0;  // J_00
        jCol[0] = 0;  // J_00
        iRow[1] = 0;  // J_01
        jCol[1] = 1;  // J_01
    } else {
        // evaluate the jacobian of the constraint function g
        values[0] = 2.0 * x[0];  // J_00 = 2*x1
        values[1] = 1.0;         // J_01 = 1.0
    }

    return true;
}

// Since we plan to use a quasi-newton method, it's not necessary
// to implement this function :)

// bool Problem::eval_h(Index n, const Number* x, bool new_x, Number obj_factor,
//                      Index m, const Number* lambda, bool new_lambda,
//                      Index nele_hess, Index* iRow, Index* jCol,
//                      Number* values) {
//     if (values == NULL) {
//         // return the structure. This is a symmetric matrix, fill the lower
//         left
//         // triangle only.

//         // element at 1,1: grad^2_{x1,x1} L(x,lambda)
//         iRow[0] = 1;
//         jCol[0] = 1;

//         // element at 2,2: grad^2_{x2,x2} L(x,lambda)
//         iRow[1] = 2;
//         jCol[1] = 2;

//         // Note: off-diagonal elements are zero for this problem
//     } else {
//         // return the values

//         // element at 1,1: grad^2_{x1,x1} L(x,lambda)
//         values[0] = -2.0 * lambda[0];

//         // element at 2,2: grad^2_{x2,x2} L(x,lambda)
//         values[1] = -2.0 * obj_factor;

//         // Note: off-diagonal elements are zero for this problem
//     }

//     return true;
// }

void Problem::finalize_solution(SolverReturn status, Index n, const Number* x,
                                const Number* z_L, const Number* z_U, Index m,
                                const Number* g, const Number* lambda,
                                Number obj_value, const IpoptData* ip_data,
                                IpoptCalculatedQuantities* ip_cq) {
    // here is where we would store the solution to variables, or write to a
    // file, etc so we could use the solution. Since the solution is displayed
    // to the console, we currently do nothing here.
}