#ifndef BASE_H
#define BASE_H
#include <Eigen/Dense>
#include <vector>
#include <iostream>

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

    class fp { 
    private:
        enum Activation {RELU, TANH, SIGMOID};

    public:
        fp();
        ~fp();

        void relu();
        void tanh();
        void sigmoid();
    };

    class bp { // Back propogation
    private:

    public:
        bp();
        ~bp();
    };


    class nn { // Neural Network
    private:
    // hyperparameters
        const Eigen::Matrix<int,3,1> input_type = Eigen::ArrayXXi::Zero(3,1); // (height, width, depth)
        const Eigen::Matrix<int,4,1> output_type = Eigen::ArrayXXi::Zero(4,1); // (height, width, depth, act func ENUM)
        const vector<Eigen::Matrix<int,7,1>> hidden_layer_type {}; // ((filter_height0, filter_width0, filter_depth0, act func ENUM0, F0, P0, S0), (filter_height1, filter_width1, filter_depth1, act func ENUM1, F1, P1, S1), .... )
    
    // Memory
        // Weights
        vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> hidden_weights {{{}}}; // C(M(N(d(Fx x Fy))))
        // Inputs
        vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>> input {{}}; // B(N(d(Fx x Fy)))
        // Outputs
        vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>> output {{}}; // B(N(d(Fx x Fy)))
        
    public:

        enum size {height, width, depth, act, F, P, S};
        
        fp FP; // forward propogation
        bp BP; // backward propogation

        // Constructor
        nn();


        /**
         * @brief Construct a new nn object defining a nural network and its hyperparameters
         * 
         * @param input_t define the input tensor size to the nn
         * @param hidden_t define each hidden layers tensor size and activation function used
         * @param output_t define the output layer tensor size and the activaiton function used
         */
        nn(const Eigen::Matrix<int,3,1>& input_t , const vector<Eigen::Matrix<int,7,1>>& hidden_t , const Eigen::Matrix<int,4,1>& output_t );   


        /**
         * @brief Destroy the nn object
         * 
         */
        ~nn();

        void init_input();


        void init_output();


        void init_weights(bool param_share_layer = true, bool param_share_depth = false);

        vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> get_hidden_weights();

        void print_layer_sizes(bool param_share_layer = true, bool param_share_depth = false);

        void print_hidden_weights(bool param_share_layer = true, bool param_share_depth = false);

        /**
         * @brief Print network hyperparameters
         * 
         */
        void print_network(); 
        

        /**
         * @brief Print hidden layer hyperparameters
         * 
         */
        void print_hidden_layers();
    };
    
} // namespace nn


#endif // BASE_H
