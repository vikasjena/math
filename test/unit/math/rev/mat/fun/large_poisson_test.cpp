
#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <vector>

/*
TEST(large, poisson_lpmf) {
  using stan::math::value_of;
  using stan::math::var;
  using std::vector;

  const int terms = 1000000;
  const int iter = 100;

  using stan::math::vector_v;

  std::vector<int> n(terms, 3);

  for(int i = 0; i < iter; ++i) {
    vector_v lambda(terms);

    for (int j = 0; j < terms; j++)
      lambda(j) = 3.0;

    var lp = stan::math::poisson_lpmf(n, lambda);

    lp.grad();
    stan::math::recover_memory();
  }
}

TEST(large, poisson_lpmf_n1) {
  using stan::math::value_of;
  using stan::math::var;
  using std::vector;

  const int terms = 100000;
  const int iter = 1000;

  using stan::math::vector_v;

  for(int i = 0; i < iter; ++i) {
    vector_v lambda(terms);
    // std::vector<int> n(terms, 3);

    for (int j = 0; j < terms; j++)
      lambda(j) = i;

    var lp = stan::math::poisson_lpmf(3, lambda);

    lp.grad();
    stan::math::recover_memory();
  }
}
*/

struct eval_poisson {
  eval_poisson() {}
  template <typename T1, typename T2>
  Eigen::Matrix<typename stan::return_type<T1, T2>::type, Eigen::Dynamic, 1>
  operator()(const Eigen::Matrix<T1, Eigen::Dynamic, 1>& eta,
             const Eigen::Matrix<T2, Eigen::Dynamic, 1>& lambda,
             const std::vector<double>& x_r, const std::vector<int>& n,
             std::ostream* msgs = 0) const {
    typedef typename stan::return_type<T1, T2>::type result_type;
    Eigen::Matrix<result_type, Eigen::Dynamic, 1> res(1);
    res(0) = stan::math::poisson_lpmf(n, lambda);
    ;
    return (res);
  }
};

STAN_REGISTER_MAP_RECT(0, eval_poisson)

TEST(large, mr_poisson_lpmf) {
  using stan::math::value_of;
  using stan::math::var;
  using std::vector;

  const int chunks_terms = 10000;
  const int chunks = 5;
  const int iter = 1000;

  using stan::math::vector_d;
  using stan::math::vector_v;

  for (int i = 0; i < iter; ++i) {
    vector_d eta;
    std::vector<std::vector<double>> x_r(chunks, std::vector<double>());
    std::vector<std::vector<int>> n(chunks, std::vector<int>(chunks_terms, 3));

    std::vector<vector_v> lambda(chunks, vector_v(chunks_terms));

    for (int j = 0; j < chunks; ++j) {
      for (int k = 0; k < chunks_terms; ++k)
        lambda[j](k) = i;
    }

    var lp = stan::math::sum(
        stan::math::map_rect<0, eval_poisson>(eta, lambda, x_r, n, 0));

    lp.grad();
    stan::math::recover_memory();
  }
}

TEST(large, poisson_lpmf) {
  using stan::math::value_of;
  using stan::math::var;
  using std::vector;

  const std::size_t terms = 50000;
  const int iter = 1000;

  using stan::math::vector_v;

  std::vector<int> n(terms, 3);

  for (int i = 0; i < iter; ++i) {
    vector_v lambda(terms);

    for (int j = 0; j < terms; ++j)
      lambda(j) = i;

    var lp = stan::math::poisson_lpmf(n, lambda);

    lp.grad();

    stan::math::recover_memory();
  }
}
