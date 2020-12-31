#include <pybind11/pybind11.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace py = pybind11;

struct ExampleFunctor {
  template<typename T>
  bool operator()(const T *const x, T *residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<ExampleFunctor,
                                           1,
                                           1>(new ExampleFunctor);
  }
};

struct PnPFunctor {
    PnPFunctor(double img_x, double img_y, double wld_x, double wld_y, double wld_z):
        img_x(img_x), img_y(img_y), wld_x(wld_x), wld_y(wld_y), wld_z(wld_z) {}
  template<typename T>
  bool operator()(const T *const camera, T *residuals) const {
    T p[3];
    T point[3];
    point[0] = T(wld_x); point[1] = T(wld_y); point[2] = T(wld_z);

    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    // camera 6 rvec + tvec
    residuals[0] =  xp - T(img_x);
    residuals[1] = yp - T(img_y);
    return true;
  }

  static ceres::CostFunction *Create(
          const double img_x, const double img_y,
          const double wld_x, const double wld_y, const double wld_z) {
    return new ceres::AutoDiffCostFunction<PnPFunctor, 2, 6>(
            new PnPFunctor(img_x, img_y, wld_x, wld_y, wld_z));
  }
  double img_x;
  double img_y;
  double wld_x;
  double wld_y;
  double wld_z;
};

void add_custom_cost_functions(py::module &m) {

  // Use pybind11 code to wrap your own cost function which is defined in C++s


  // Here is an example
  m.def("CreateCustomExampleCostFunction", &ExampleFunctor::Create);
  m.def("CreatePnPCostFunction", &PnPFunctor::Create);

}
