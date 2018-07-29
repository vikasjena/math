#ifndef STAN_MATH_PRIM_SCAL_PROB_POISSON_LPMF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_POISSON_LPMF_HPP

#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/operands_and_partials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/size_zero.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/is_inf.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/scalar_seq_view.hpp>
#include <stan/math/prim/mat/functor/map_rect_concurrent.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <limits>

#include <thread>
#include <future>

namespace stan {
namespace math {

// Poisson(n|lambda)  [lambda > 0;  n >= 0]
template <bool propto, typename T_n, typename T_rate>
typename return_type<T_rate>::type poisson_lpmf(const T_n& n,
                                                const T_rate& lambda) {
  typedef
      typename stan::partials_return_type<T_n, T_rate>::type T_partials_return;

  static const char* function = "poisson_lpmf";

  if (size_zero(n, lambda))
    return 0.0;

  T_partials_return logp(0.0);

  check_nonnegative(function, "Random variable", n);
  check_not_nan(function, "Rate parameter", lambda);
  check_nonnegative(function, "Rate parameter", lambda);
  check_consistent_sizes(function, "Random variable", n, "Rate parameter",
                         lambda);

  if (!include_summand<propto, T_rate>::value)
    return 0.0;

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_rate> lambda_vec(lambda);
  size_t size = max_size(n, lambda);
  size_t n_size = n_vec.size();

  for (size_t i = 0; i < size; i++)
    if (is_inf(lambda_vec[i]))
      return LOG_ZERO;
  for (size_t i = 0; i < size; i++)
    if (lambda_vec[i] == 0 && n_vec[i] != 0)
      return LOG_ZERO;

  operands_and_partials<T_rate> ops_partials(lambda);

  auto execute_chunk = [&](int start, int size) -> T_partials_return {
    const int end = start + size;
    T_partials_return logp_chunk(0.0);
    for (int i = start; i != end; i++) {
      if (!(lambda_vec[i] == 0 && n_vec[i] == 0)) {
        if (include_summand<propto>::value) {
          if (n_size != 1)
            logp_chunk -= lgamma(n_vec[i] + 1.0);
        }
        if (include_summand<propto, T_rate>::value)
          logp_chunk += multiply_log(n_vec[i], value_of(lambda_vec[i]))
                        - value_of(lambda_vec[i]);
      }

      if (!is_constant_struct<T_rate>::value)
        ops_partials.edge1_.partials_[i]
            += n_vec[i] / value_of(lambda_vec[i]) - 1.0;
    }
    return logp_chunk;
  };

  std::vector<std::future<T_partials_return>> futures;

  int num_threads = internal::get_num_threads(size);
  int num_jobs_per_thread = size / num_threads;
  futures.emplace_back(
      std::async(std::launch::deferred, execute_chunk, 0, num_jobs_per_thread));

#ifdef STAN_THREADS
  if (num_threads > 1) {
    const int num_big_threads
        = (size - num_jobs_per_thread) % (num_threads - 1);
    const int first_big_thread = num_threads - num_big_threads;
    for (int i = 1, job_start = num_jobs_per_thread, job_size = 0;
         i < num_threads; ++i, job_start += job_size) {
      job_size = i >= first_big_thread ? num_jobs_per_thread + 1
                                       : num_jobs_per_thread;
      futures.emplace_back(
          std::async(std::launch::async, execute_chunk, job_start, job_size));
    }
  }
#endif

  for (std::size_t i = 0; i < futures.size(); ++i)
    logp += futures[i].get();

  if (include_summand<propto>::value)
    if (n_size == 1)
      logp -= size * lgamma(n_vec[1] + 1.0);

  return ops_partials.build(logp);
}

template <typename T_n, typename T_rate>
inline typename return_type<T_rate>::type poisson_lpmf(const T_n& n,
                                                       const T_rate& lambda) {
  return poisson_lpmf<false>(n, lambda);
}

}  // namespace math
}  // namespace stan
#endif
