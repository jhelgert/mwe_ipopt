#include "IpIpoptApplication.hpp"
#include "Problem.hpp"
#include "iostream"

/*
clang++ main.cpp Problem.cpp -std=c++17 -Wall -O3
-I/usr/local/Cellar/ipopt/3.14.1/include/coin-or
-L/usr/local/Cellar/ipopt/3.14.1/lib -lipopt -o main
*/

int main() {
    Ipopt::SmartPtr<Problem> nlp = new Problem();

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    // Process the options
    auto status = app->Initialize();

    // Solve the problem
    status = app->OptimizeTNLP(nlp);

    if (status == Ipopt::Solve_Succeeded) {
        std::cout << "Problem solved!" << '\n';
    }

    return (int)status;
}