#ifndef PYTORCH_HNet_H
#define PYTORCH_HNet_H

#include <string>

#include <Eigen/StdVector>
#include <Eigen/Eigen>

#include <torch/torch.h> // pytorch
#include <torch/script.h> // for reading .pt model

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/date_time/posix_time/posix_time.hpp> // for timing
#include <c10/cuda/CUDAStream.h>

#define BLUE   "\033[34m" // for printing
#define RESET  "\033[0m"

namespace pytorch {

    class HomographyNet {
    
    public:
        HomographyNet(std::string &network_model_path, std::string &network_model_iterative_path, bool use_prior, int num_of_iteration, bool show_imgs);
        ~HomographyNet();
        void load_current_img(const cv::Mat &img, const double &time_stamp);
        Eigen::Matrix<double, 8, 1> get_pred_mean() {return _pred_mean.cast<double>();}
        Eigen::Matrix<double, 8, 8> get_pred_Cov() {return _pred_Cov.cast<double>();}
        double get_latest_inference_time() {return _latest_inference_time_stamp;}  
        void network_inference(Eigen::Matrix<double, 8, 1> &prior_4pt_offset_vec, int iteration);
        int img_counter = 0;

    private:
        int inference_counting = 0;
        void load_network_model(std::string &network_model_path);
        void load_network_model_iterative(std::string &network_model_iterative_path);
        bool cv_imshow;
        bool use_prior_4pt_offset;
        bool show_phtometric_error;

        torch::jit::script::Module HomographyNet_model;
        torch::jit::script::Module HomographyNet_model_iterative;

        torch::DeviceType device_type;

        boost::posix_time::ptime rT1, rT2, rT3; // timing the network

        std::vector<torch::jit::IValue> nn_inputs; // a vector of inputs.

        at::Tensor tensor_img_prev;
        at::Tensor tensor_img_curr;
        at::Tensor prior_4pt_offset_tensor;

        double sum_nn_time = 0;
        double _latest_inference_time_stamp = -1.0;
        
        Eigen::Matrix<float, 8, 1> _pred_mean;
        Eigen::Matrix<float, 8, 8> _pred_Cov;

        // only for debugging
        // std::ofstream network_output_txt;

        bool iteration = false;
        
    };
}

#endif // PYTORCH_HNet_H