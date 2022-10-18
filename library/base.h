#ifndef BASE_H
#define BASE_H
#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>

/**
 * @brief 
 * (M):= depth of a layer output.. number of desired filters
 * (N):= # of neurons in a layer output
 * (d):= depth of input to a layer
 * (F):= filter demension
 * (B):= Batch size
 * (C):= # of convolutions
 */

using namespace std;

namespace nn
{
    class nn{ // Neural Network
    private:
    // hyperparameters
        const Eigen::Matrix<int,3,1> input_type = Eigen::ArrayXXi::Zero(3,1); // (height, width, depth)
        const Eigen::Matrix<int,4,1> output_type = Eigen::ArrayXXi::Zero(4,1); // (height, width, depth, act func ENUM)
        // Vector of convolution layer hyperparameters
        vector<Eigen::Matrix<int,7,1>> hidden_layer_type {}; // ((filter_height0, filter_width0, filter_depth0, act func ENUM0, F0, P0, S0), (filter_height1, filter_width1, filter_depth1, act func ENUM1, F1, P1, S1), .... )
    
    // Memory
        // Weights
        vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> hidden_weights {{{{}}}}; // C(M(N(d(Fx x Fy))))
        // Biases
        vector< vector< vector< vector<float>>>> bias {{{{}}}}; // C(M(N(d(b))))
        // Nueron Values
        vector<vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>> neuron_outputs {{{}}}; // b(C(M(Fx x Fy))
        // Input
        vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> input_train {{}}; // b(d(matrix)) 
        // Output
        vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> output_train {{}}; // b(d(matrix))
        // Cost
        vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> cost {{}}; // b(d(matrix))
        // Cost Gradient
        vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> cost_grad {{}}; // b(d(matrix))
        // Average Loss
        float loss_avg {};
        // Total Loss
        float loss_tot {};

    // System
        bool parameter_sharing = true;
        int batch_size = 0;
        
    public:

        enum size {height, width, depth, act, F, P, S};
        enum func {LINEAR, RELU, TANH, SIGMOID, FILL4, FILL5, FILL6, FILL7, FILL8, FILL9, LS_LOSS};

        nn();

        nn(const Eigen::Matrix<int,3,1>& input_t , vector<Eigen::Matrix<int,7,1>>& hidden_t , const Eigen::Matrix<int,4,1>& output_t );   

        ~nn();
        
        void init_weights(bool param_share_layer = true, bool param_share_depth = false);

        void init_neurons();

        void init_output();

        void init_cost();

        void forward(vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>>&, vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>>&);

        void backward();

        void activate(float&,int);

        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> deactivate(const Eigen::Block<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>&, int);

        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> loss(const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>&, const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>&, int);

        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> dloss(const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>&, const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>&, int);

        void calc_average_loss();

        void print_neuron_outputs();

        void print_output();

        void print_layer_sizes(bool param_share_layer = true, bool param_share_depth = false);

        void print_hidden_weights(bool param_share_layer = true, bool param_share_depth = false);

        void print_network(); 
        
        void print_hidden_layers();
    };
    
} // namespace nn


#endif // BASE_H
