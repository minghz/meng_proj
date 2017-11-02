#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("ReshapeFix")
    .Input("to_round: float")
    .Input("fdefinition: float")
    .Input("bdefinition: float")
    .Input("foverflow: float")
    .Input("boverflow: float")
    .Output("roundeded: float")    
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
    const Tensor& ILFL = context->input(1);
    Tensor& foverflow = const_cast<Tensor&>(context->input(3)); //have to preform a const_cast to be able to pass by reference

    auto range_precision = ILFL.flat<float>();

    auto input = input_tensor.flat<float>();

    auto overflow = foverflow.flat<float>();

    int range = pow(2,(int)range_precision(0));
    int precision = pow(2,(int)range_precision(1));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    auto output = output_tensor->flat<float>();
    
    // round all elements to fix point.
    const int N = input.size();
    int overflow_count = 0;
    int underflow_count = 0;
    double sum,avg = 0;
    double tmp;

    // loops through the tensor and applies fixedpoint rounding based on the given range and precision
    // counts overflows and underflows during the process, also saves percent change in rounding
    for (int i = 0; i < N; i++){
      if(input(i) >  (range)){

        output(i) =  (range);
        overflow_count++;
      }else if(input(i) <  (-range)){

        output(i) =  (-range);
        overflow_count++;
      }else{

        output(i) = input(i)* precision;
        output(i) =  ((int)(output(i)+ 0.5));
        output(i) = output(i)/ precision;

        if (output(i) != input(i)){
          underflow_count++;
          tmp = fabs((output(i) - input(i))/input(i));
          sum += tmp;
        }
      }
    }

    avg = sum/N;
    overflow(0) += (float)overflow_count/N; //saves the percentage of elements that overflowed
    overflow(1) = avg;   //saves average percent change during rounding
  }
};

REGISTER_KERNEL_BUILDER(Name("ReshapeFix").Device(DEVICE_CPU), ReshapeFixOp);
