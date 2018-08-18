#ifndef STAN_MATH_GPU_CL_HPP
#define STAN_MATH_GPU_CL_HPP
#ifdef STAN_OPENCL
#include <stan/math/gpu/opencl_context.hpp>
#include <stan/math/gpu/constants.hpp>
#include <CL/cl.hpp>
#include <string>
#include <algorithm>
#include <map>
#include <vector>

namespace stan {
namespace math {
namespace {

std::string helpers =                      // Helper macros for the kernels.
#include <stan/math/gpu/kernels/helpers.cl>  // NOLINT
    ;                                      // NOLINT
// Holds Default parameter values for each Kernel.
typedef std::map<const char*, int> map_base_opts;
map_base_opts base_opts
= {{"LOWER", static_cast<int>(TriangularViewGPU::Lower)},
   {"UPPER", static_cast<int>(TriangularViewGPU::Upper)},
   {"ENTIRE", static_cast<int>(TriangularViewGPU::Entire)},
   {"UPPER_TO_LOWER", static_cast<int>(TriangularMapGPU::UpperToLower)},
   {"LOWER_TO_UPPER", static_cast<int>(TriangularMapGPU::LowerToUpper)}};

auto compile_kernel(const char* name, const char* source) {
  std::string kernel_opts = "";
  for (auto&& comp_opts : base_opts) {
    kernel_opts += std::string(" -D") + comp_opts.first + "="
                   + std::to_string(comp_opts.second);
  }
  std::string kernel_source(helpers);
  kernel_source.append(source);
  try {
    cl::Program::Sources src(
        1,
        std::make_pair(kernel_source.c_str(), strlen(kernel_source.c_str())));
    cl::Program program = cl::Program(opencl_context.context(), src);
    program.build({opencl_context.device()}, kernel_opts.c_str());

    return cl::Kernel(program, name);
  } catch (const cl::Error& e) {
    check_opencl_error("Kernel Compilation", e);
  }
  return cl::Kernel(); // never reached because check_opencl_error throws
}
}

namespace opencl_kernels {

template <typename ...Args>
class kernel_functor {
 private:
  const char* name_;
  const char* source_;
  cl::Kernel kernel_;

 public:
  kernel_functor(const char* name, const char* source)
      : name_(name), source_(source), kernel_(compile_kernel(name_, source_)) {};

  auto operator()() const {
    return cl::make_kernel<Args...>(kernel_);
  };
};

template <typename ...Args>
struct rows_cols_kernel {
  const kernel_functor<...Args, int, int> make_functor;
}

struct nullary_kernel {
  const kernel_functor<cl::Buffer, int, int> make_functor;
  nullary_kernel(const char* name, const char* source) : make_functor(name, source) {};
  void operator()(cl::Buffer dest, int rows, int cols) const {
    auto f = make_functor();
    cl::EnqueueArgs eargs(opencl_context.queue(), cl::NDRange(rows, cols));
    f(eargs, dest, rows, cols).wait();
  }
};

const nullary_kernel identity("identity",
#include <stan/math/gpu/kernels/identity_matrix.cl>  // NOLINT
                              );

struct unary_kernel {
  const kernel_functor<cl::Buffer, cl::Buffer, int, int> make_functor;
  unary_kernel(const char* name, const char* source) : make_functor(name, source) {};
  void operator()(cl::Buffer src, cl::Buffer dest, int rows, int cols) const {
    auto f = make_functor();
    cl::EnqueueArgs eargs(opencl_context.queue(), cl::NDRange(rows, cols));
    f(eargs, src, dest, rows, cols).wait();
  }
};

const unary_kernel copy("copy",
#include <stan/math/gpu/kernels/copy_matrix.cl>  // NOLINT
                         );

const unary_kernel copy_triangular("copy_triangular",
#include <stan/math/gpu/kernels/copy_triangular_matrix.cl>  // NOLINT
                         );

const unary_kernel transpose("transpose",
#include <stan/math/gpu/kernels/transpose_matrix.cl>  // NOLINT
                             );

struct binary_kernel {
  const kernel_functor<cl::Buffer, cl::Buffer, cl::Buffer, int, int> make_functor;
  unary_kernel(const char* name, const char* source) : make_functor(name, source) {};
  void operator()(cl::Buffer lhs, cl::Buffer rhs, cl::Buffer dest, int rows, int cols) const {
    auto f = make_functor();
    cl::EnqueueArgs eargs(opencl_context.queue(), cl::NDRange(rows, cols));
    f(eargs, lhs, rhs, dest, rows, cols).wait();
  }
};

const binary_kernel add("add",
#include <stan/math/gpu/kernels/add_matrix.cl>  // NOLINT
                        );

struct check_kernel {
  const kernel_functor<cl::Buffer, int, int, int*> make_functor;
  check_kernel(const char* name, const char* source) : make_functor(name, source) {};
  void operator()(cl::Buffer mat, int rows, int cols int* flag) const {
    auto f = make_functor();
    cl::EnqueueArgs eargs(opencl_context.queue(), cl::NDRange(rows, cols));
    f(eargs, mat, rows, cols, flag).wait();
  }
};

const check_kernel check_diagonal_zeros("check_diagonal_zeros",
#include <stan/math/gpu/kernels/check_diagonal_zeros.cl>  // NOLINT
                           );

const check_kernel check_nan("check_nan",
#include <stan/math/gpu/kernels/check_nan.cl>  // NOLINT
                                        );

const check_kernel check_symmetric("check_symmetric",
#include <stan/math/gpu/kernels/check_symmetric.cl>  // NOLINT
                             );

const check_kernel check_nan("check_nan",
#include <stan/math/gpu/kernels/check_nan.cl>  // NOLINT
                             );

struct unary_sub_kernel {
  const kernel_functor<cl::Buffer, cl::Buffer, int, int, int, int, int, int, int, int>
  make_functor;

  check_kernel(const char* name, const char* source) : make_functor(name, source) {};

  void operator()(cl::Buffer src, cl::Buffer dest,
                  int dst_offset_row, int dst_offset_col,
                  int sub_rows, int sub_cols,
                  int src_rows, int src_cols,
                  int dst_rows, int dst_cols) const {
    auto f = make_functor();
    cl::EnqueueArgs eargs(opencl_context.queue(), cl::NDRange(rows, cols));
    f(eargs, mat, rows, cols, flag).wait();
  }
}

}  // namespace math
}  // namespace stan

#endif
#endif
