#include <iostream>
#include "HomographyNet.h"
#include <fstream> // for debugging file printing

namespace pytorch {

HomographyNet::HomographyNet(std::string &network_model_path, std::string &network_model_iterative_path, bool use_prior, int num_of_iteration, bool show_imgs) {

    use_prior_4pt_offset = use_prior;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Running on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Running on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    load_network_model(network_model_path);

    if (num_of_iteration > 1) {
        iteration = true;
        load_network_model_iterative(network_model_iterative_path);
        std::cout << "IEKF! Load the Network for Iteration!" << std::endl;
    }

    nn_inputs.clear();
    
    // run the network once because the first time inference cost far longer time
    nn_inputs.push_back(torch::ones({1, 1, 224, 320}).to(torch::Device(device_type))*0.2);
    nn_inputs.push_back(torch::ones({1, 1, 224, 320}).to(torch::Device(device_type))*0.5);
    if (use_prior_4pt_offset) {
        nn_inputs.push_back(torch::ones({1, 1, 4, 2}).to(torch::Device(device_type)));
    }

    at::Tensor mean;
    at::Tensor Cov; 

    rT2 =  boost::posix_time::microsec_clock::local_time();

    // Execute the model and turn its output into a tensor (or a tuple of tensors)
    // when we output a tuple
    auto output = HomographyNet_model.forward(nn_inputs);

    mean = output.toTuple()->elements()[0].toTensor(); // 1st output
    Cov = output.toTuple()->elements()[1].toTensor(); // 2nd output

    // std::cout << "Network Output (testing after init): " << mean << "    " << Cov << '\n';

    if (iteration) {
        auto output_iterative = HomographyNet_model_iterative.forward(nn_inputs);

        mean = output_iterative.toTuple()->elements()[0].toTensor(); // 1st output
        Cov = output_iterative.toTuple()->elements()[1].toTensor(); // 2nd output

        // std::cout << "Network (for iteration) Output (testing after init): " << mean << "    " << Cov << '\n';
    }

    nn_inputs.clear();

    rT3 =  boost::posix_time::microsec_clock::local_time();

    double time_network = (rT3-rT2).total_microseconds() * 1e-3; // Timing information
    printf(BLUE "[TIME]: %.4f milliseconds for the first network inference\n" RESET, time_network);
    // NOTE the tx2 MAXP CORE ARM (3) mode is fast than the MAX-N mode (0) for network inference 

    cv_imshow = show_imgs;
    if (cv_imshow) {
        namedWindow("Image", cv::WINDOW_KEEPRATIO);// Create a window for display.
        namedWindow("Photometric Error", cv::WINDOW_KEEPRATIO);// Create a window for display.
    }

    // network_output_txt.open("network_output.txt");

}

HomographyNet::~HomographyNet() {
    // network_output_txt.close();
    std::cout << "HomographyNet Object is being deleted! End of this run ..." << std::endl;
}

void HomographyNet::load_network_model(std::string &network_model_path) {

    std::cout << "Loading the Network Model from Torch Script ..." << std::endl;

    // https://pytorch.org/tutorials/advanced/cpp_export.html

    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      HomographyNet_model = torch::jit::load(network_model_path); // better use the absolute path...
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model !!!\n";
    }

    std::cerr << network_model_path << std::endl;
    if (network_model_path.find("_showError") == std::string::npos ) {
        show_phtometric_error = false;
    } else {
        show_phtometric_error = true;
    }

}

void HomographyNet::load_network_model_iterative(std::string &network_model_iterative_path) {

    std::cout << "Loading the Network Model for IEKF from Torch Script ..." << std::endl;
    // https://pytorch.org/tutorials/advanced/cpp_export.html

    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      HomographyNet_model_iterative = torch::jit::load(network_model_iterative_path); 
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model !!!\n";
    }

    std::cerr << network_model_iterative_path << std::endl;
    if (network_model_iterative_path.find("_showError") == std::string::npos ) {
        show_phtometric_error = false;
    } else {
        show_phtometric_error = true;
    }

}


void HomographyNet::load_current_img(const cv::Mat &img, const double &time_stamp) {

    if (cv_imshow) {
        cv::imshow("Image", img);
        cv::waitKey(1);
    }

    img_counter++;

    // convert cv::Mat into tensor
    if (img_counter == 1) { // first img
        std::cout << "First Image Comes into the Network Object!" << std::endl;
        tensor_img_curr = torch::from_blob(img.data, { img.rows, img.cols, 1 }, at::kByte).to(torch::Device(device_type));
        tensor_img_curr = tensor_img_curr.permute({ 2, 0, 1 }); // H, W, C --> C, H, W
        tensor_img_curr = tensor_img_curr.toType(c10::kFloat) / 255.0;
    } else {
        tensor_img_prev = tensor_img_curr.clone();
        tensor_img_curr = torch::from_blob(img.data, { img.rows, img.cols, 1 }, at::kByte).to(torch::Device(device_type));
        tensor_img_curr = tensor_img_curr.permute({ 2, 0, 1 }); // H, W, C --> C, H, W
        tensor_img_curr = tensor_img_curr.toType(c10::kFloat) / 255.0;

        _latest_inference_time_stamp = time_stamp;
    }

}

void HomographyNet::network_inference(Eigen::Matrix<double, 8, 1> &prior_4pt_offset_vec, int num_of_inference) { 

    if (img_counter < 2) { // first img
        std::cout << "HNet cannot inference! Only has one image!" << std::endl;
        return;
    }

    if (use_prior_4pt_offset) {
        prior_4pt_offset_tensor = torch::tensor({{prior_4pt_offset_vec[0], prior_4pt_offset_vec[1]}, 
                                                 {prior_4pt_offset_vec[2], prior_4pt_offset_vec[3]}, 
                                                 {prior_4pt_offset_vec[4], prior_4pt_offset_vec[5]}, 
                                                 {prior_4pt_offset_vec[6], prior_4pt_offset_vec[7]}}).unsqueeze(0).unsqueeze(0).toType(c10::kFloat).to(torch::Device(device_type));
    }

    nn_inputs.push_back(tensor_img_prev.unsqueeze(0)); // [1, 1, 224, 320]
    nn_inputs.push_back(tensor_img_curr.unsqueeze(0)); // [1, 1, 224, 320]

    if (use_prior_4pt_offset) {
        nn_inputs.push_back(prior_4pt_offset_tensor); // [1, 1, 4, 2]
    }

    // Execute the model and turn its output into a tensor.
    // when we output a tuple
    if (num_of_inference == 0) {

        if (torch::cuda::is_available()) {
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        rT1 =  boost::posix_time::microsec_clock::local_time();
        auto output = HomographyNet_model.forward(nn_inputs); 
        if (torch::cuda::is_available()) {
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        rT2 =  boost::posix_time::microsec_clock::local_time();
        inference_counting = inference_counting + 1;
        at::Tensor mean_tensor = output.toTuple()->elements()[0].toTensor().to(torch::Device(torch::kCPU)); // 1st output mean
        at::Tensor Cov_tensor = output.toTuple()->elements()[1].toTensor().to(torch::Device(torch::kCPU)); // 2nd output Cov

        float* data_mean = mean_tensor.data_ptr<float>();
        _pred_mean = Eigen::Map<Eigen::Matrix<float, 8, 1>>(data_mean, 8, 1);

        float* data_Cov = Cov_tensor.data_ptr<float>();
        _pred_Cov = Eigen::Map<Eigen::Matrix<float, 8, 8>>(data_Cov, 8, 8);

        if (cv_imshow & show_phtometric_error & (! iteration)) {
            at::Tensor photometric_error_tensor = output.toTuple()->elements()[2].toTensor().to(torch::Device(torch::kCPU)); // 3rd output photometric_error
            photometric_error_tensor = photometric_error_tensor.squeeze(0).detach().permute({1, 2, 0}).clamp(0, 255).to(torch::kU8);
            cv::Mat resultImg(224, 320, CV_8UC1);
            std::memcpy((void *) resultImg.data, photometric_error_tensor.data_ptr(), sizeof(torch::kU8) * photometric_error_tensor.numel());

            cv::imshow("Photometric Error", resultImg);
            cv::waitKey(1); // The function waitKey waits for a key event infinitely (when delay≤0 ) or for delay milliseconds, when it is positive. 
        }

    } else {

        auto output = HomographyNet_model_iterative.forward(nn_inputs); 
        
        at::Tensor mean_tensor = output.toTuple()->elements()[0].toTensor().to(torch::Device(torch::kCPU)); // 1st output mean
        at::Tensor Cov_tensor = output.toTuple()->elements()[1].toTensor().to(torch::Device(torch::kCPU)); // 2nd output Cov

        float* data_mean = mean_tensor.data_ptr<float>();
        _pred_mean = Eigen::Map<Eigen::Matrix<float, 8, 1>>(data_mean, 8, 1);

        float* data_Cov = Cov_tensor.data_ptr<float>();
        _pred_Cov = Eigen::Map<Eigen::Matrix<float, 8, 8>>(data_Cov, 8, 8);
        
        if (cv_imshow & show_phtometric_error) {
            at::Tensor photometric_error_tensor = output.toTuple()->elements()[2].toTensor().to(torch::Device(torch::kCPU)); // 3rd output photometric_error
            photometric_error_tensor = photometric_error_tensor.squeeze(0).detach().permute({1, 2, 0}).clamp(0, 255).to(torch::kU8);
            cv::Mat resultImg(224, 320, CV_8UC1);
            std::memcpy((void *) resultImg.data, photometric_error_tensor.data_ptr(), sizeof(torch::kU8) * photometric_error_tensor.numel());

            cv::imshow("Photometric Error", resultImg);
            cv::waitKey(1);
        }
    }
    
    double pred_Cov_sum = 0.0;
    double pred_mean_sum = 0.0;
    for (int i = 0; i < 8; i++) {
        // network_output_txt << _latest_inference_time_stamp << " " << _pred_mean(i, 0) << " " << _pred_Cov(i, i) << std::endl;
        pred_Cov_sum = pred_Cov_sum + _pred_Cov(i, i);
        pred_mean_sum = pred_mean_sum + abs(_pred_mean(i, 0));
    }
    // network_output_txt.precision(20);
    // network_output_txt << _latest_inference_time_stamp << " " << img_counter << " " << pred_mean_sum / 8.0 << " " << pred_Cov_sum / 8.0 << std::endl;

    nn_inputs.clear();

    if (num_of_inference == 0) {
        double time_network = (rT2-rT1).total_microseconds() * 1e-3;
        if (inference_counting > 100) {
            sum_nn_time = sum_nn_time + time_network;
            printf(BLUE "[TIME]: %.3f (avg. = %.3f) milliseconds for pure network inference\n" RESET, time_network, sum_nn_time/(inference_counting-100));
        }
    }
}

}
