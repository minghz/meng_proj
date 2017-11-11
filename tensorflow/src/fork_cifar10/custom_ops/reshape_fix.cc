#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("ReshapeFix")
.Input("to_fix: float") //input tensor
.Input("fdefinition: float") // range and precision bits (m, n)
.Input("bdefinition: float")
.Input("foverflow: float")
.Input("boverflow: float")
.Output("fixed: float")    
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});

class ReshapeFixOp : public OpKernel {
  public:
    explicit ReshapeFixOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      const Tensor& range_precision = context->input(1);
      // have to preform a const_cast to be able to pass by reference
      Tensor& foverflow = const_cast<Tensor&>(context->input(3));

      auto m_n = range_precision.flat<float>(); //range - precision
      auto input = input_tensor.flat<float>();
      auto overflow = foverflow.flat<float>();

      float range_min = -1 * pow(2, (m_n(0) - 1));
      float range_max = pow(2, (m_n(0) - 1)) - pow(2, -1 * (m_n(1)));
      float resolution = pow(2, -1 * (m_n(1)));

      std::cout << "range: [" << range_min << ", " << range_max << "]"
        << " | resolution: " << resolution
        << std::endl;

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, input_tensor.shape(), &output_tensor));
      auto output = output_tensor->flat<float>();

      // convert input tensor to fixed point equivalent range
      // and precision with a 5% resolution tolerance
      // counts times when range is clipped, or tolerance exceeded
      int overflow_count = 0;
      int unprecise_count = 0;
      const int input_count = input.size();
      for (int i = 0; i < input_count; i++) {

        // clip on max and min of allowed range
        if (input(i) > range_max || input(i) < range_min ) {
          overflow_count++;
          if (input(i) > range_max) { output(i) = range_max; }
          if (input(i) < range_min) { output(i) = range_min; }

        // convert resolution to fixed point equivalent
        } else {
          float fix_equivalent = resolution * trunc(input(i) / resolution);
          float deviation_from_orig = abs(fix_equivalent - input(i)) / input(i);
          if(deviation_from_orig > 0.05){ // more than 5% deviation
            unprecise_count++;
          }
        }
      }
      overflow(0) = (float) (overflow_count / input_count);
      overflow(1) = (float) (unprecise_count / input_count);
    }
};

REGISTER_KERNEL_BUILDER(Name("ReshapeFix").Device(DEVICE_CPU), ReshapeFixOp);
