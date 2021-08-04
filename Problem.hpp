#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include "IpTNLP.hpp"

using namespace Ipopt;

/*
For f: R^n -> R and g:R^n -> R^m we solve:
min     f(x)
s.t.
      g_LU <= g(x) <= g_UB,
      x_LU <=  x   <= x_UB
*/
// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-11-05

using namespace Ipopt;

class Problem : public TNLP {
   public:
    /** default constructor */
    Problem();

    /** default destructor */
    virtual ~Problem();

    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style);

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m,
                                 Number* g_l, Number* g_u);

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda);

    /** Method to return the objective value */
    virtual bool eval_f(Index n, const Number* x, bool new_x,
                        Number& obj_value);

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x,
                             Number* grad_f);

    /** Method to return the constraint residuals */
    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m,
                        Number* g);

    /** Method to return:
     *   1) The structure of the Jacobian (if "values" is NULL)
     *   2) The values of the Jacobian (if "values" is not NULL)
     */
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x, Index m,
                            Index nele_jac, Index* iRow, Index* jCol,
                            Number* values);

    /** Method to return:
     *   1) The structure of the Hessian of the Lagrangian (if "values" is NULL)
     *   2) The values of the Hessian of the Lagrangian (if "values" is not
     * NULL)
     */
    // virtual bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor,
    //                     Index m, const Number* lambda, bool new_lambda,
    //                     Index nele_hess, Index* iRow, Index* jCol,
    //                     Number* values);

    /** This method is called when the algorithm is complete so the TNLP can
     * store/write the solution */
    virtual void finalize_solution(SolverReturn status, Index n,
                                   const Number* x, const Number* z_L,
                                   const Number* z_U, Index m, const Number* g,
                                   const Number* lambda, Number obj_value,
                                   const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq);
};

#endif