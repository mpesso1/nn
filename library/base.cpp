#include "base.h"
#include <iostream>
#include <vector>
#include <random>

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

// Default Constructor
nn::nn::nn() {}

// Overload Constructor
nn::nn::nn(const Eigen::Matrix<int,3,1>& input, const vector<Eigen::Matrix<int,7,1>>& hidden, const Eigen::Matrix<int,4,1>& output) : 
    input_type(input), output_type(output), hidden_layer_type(hidden)
    {
        nn::init_weights();
    }

// Default Destructor
nn::nn::~nn() {}

// init weights to random values C(M(N(d(Fx x Fy)))) vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> hidden_weights
void nn::nn::init_weights(bool param_share_layer, bool param_share_depth) {
    
        hidden_weights.resize(hidden_layer_type.size()); // resize weight memory to the correct # of convolutions

        int c_itr = 0; // convolution counter

        for (auto c = hidden_layer_type.begin(); c != hidden_layer_type.end(); c++) { // loop over each convolution

            hidden_weights[c_itr].resize((*c)(nn::depth)); // resize the number of desired filters for the specific convolution

            for (int m = 0; m < (*c)(nn::depth); m++){ // loop over each filter of the convolution

                int N = 1;
                if (!param_share_layer && !param_share_depth) {
                    N = (input_type[nn::height]*input_type[nn::width] - (*c)(nn::F) + 2*(*c)(nn::P))/(*c)(nn::S) + 1;
                }

                hidden_weights[c_itr][m].resize(N); // for a certain covolution and filter, specify the number neurons for that filter

                for (int n = 0; n < N; n++) { // one loop to produce weights for neurons of the filter
                    
                    hidden_weights[c_itr][m][n].resize(input_type[nn::depth]); // for a certain convolution, filter, and neuron, specify the depth of the input

                    for (int d = 0; d < input_type[nn::depth]; d++) { // loop over all input depths
                        
                        random_device rd;
                        mt19937 gen(rd());
                        uniform_real_distribution<> distr(-1,1);

                        if ((*c)(nn::F)/2 == 1) { // if the filter size is of size 1

                            hidden_weights[c_itr][m][n][d].resize((*c)(nn::F)/2, 0); // for a conv, filter, neuron, and input depth, specify the weight matrix
                            hidden_weights[c_itr][m][n][d](1,0) = distr(gen);
                        }

                        else { // if filter is greater than 1... next size should be 4 (2x2 filter)
                            
                            hidden_weights[c_itr][m][n][d].resize((*c)(nn::F)/2, (*c)(nn::F)/2); // for a conv, filter, neuron, and input depth, specify the weight matrix

                            for(int i = 0; i < hidden_weights[c_itr][m][n][d].rows(); i++) { // loop over each weight and assign it a random # between -1, 1

                                for(int j = 0; j < hidden_weights[c_itr][m][n][d].cols(); j++) {

                                    hidden_weights[c_itr][m][n][d](i,j) = distr(gen);
                                }
                            }
                        }
                    }
                }
            }
            c_itr++; // increase convolution pointer
        }
};

vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> nn::nn::get_hidden_weights(){
    return hidden_weights;
}

void nn::nn::print_layer_sizes(bool param_share_layer, bool param_share_depth ) {

    cout << "# of convolutions: " << hidden_weights.size() << endl;
    int c_itr = 0; // convolution counter
    for (auto c = hidden_layer_type.begin(); c != hidden_layer_type.end(); c++) { // loop over each convolution

        cout << "# of filters for conv " << c_itr << " : " << hidden_weights[c_itr].size() << endl;

        for (int m = 0; m < (*c)(nn::depth); m++){ // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth) {
                N = (input_type[nn::height]*input_type[nn::width] - (*c)(nn::F) + 2*(*c)(nn::P))/(*c)(nn::S) + 1;
            }
            cout << "# of neurons for conv " << c_itr << " and filter " << m << " : " << hidden_weights[c_itr][m].size() << endl; // for a certain covolution and filter, specify the number neurons for that filter

            // int N = 1; // all neurons will have the same weights..

            for (int n = 0; n < N; n++) { // one loop to produce weights for neurons of the filter
                
                cout << "# of input depth for conv " << c_itr << " and filter " << m << " and neuron " << n << " : " << hidden_weights[c_itr][m][n].size() << endl;; // for a certain convolution, filter, and neuron, specify the depth of the input

                for (int d = 0; d < input_type[nn::depth]; d++) { // loop over all input depths
                    
                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> distr(-1,1);

                    if ((*c)(nn::F) == 1) { // if the filter size is of size 1
                        // cout << 5 << endl;
                        // cout << (*c)(nn::F) << endl;
                        hidden_weights[c_itr][m][n][d].resize((*c)(nn::F), 1); // for a conv, filter, neuron, and input depth, specify the weight matrix
                        // cout << hidden_weights[c_itr][m][n][d].rows() << endl;
                        // cout << hidden_weights[c_itr][m][n][d].cols() << endl;
                        // cout << distr(gen) << endl;
                        hidden_weights[c_itr][m][n][d](0,0) = distr(gen);
                    }

                    else { // if filter is greater than 1... next size should be 4 (2x2 filter)
                        
                        hidden_weights[c_itr][m][n][d].resize((*c)(nn::F)/2, (*c)(nn::F)/2); // for a conv, filter, neuron, and input depth, specify the weight matrix

                        for(int i = 0; i < hidden_weights[c_itr][m][n][d].rows(); i++) { // loop over each weight and assign it a random # between -1, 1

                            for(int j = 0; j < hidden_weights[c_itr][m][n][d].cols(); j++) {

                                hidden_weights[c_itr][m][n][d](i,j) = distr(gen);
                            }
                        }
                    }
                    cout << "# of weights for conv " << c_itr << " and filter " << m << " and neuron " << n << " and depth " << d << " : " << hidden_weights[c_itr][m][n][d].size() << endl;
                }
            }
        }
        c_itr++; // increase convolution pointer
    }
}

void nn::nn::print_hidden_weights(bool param_share_layer, bool param_share_depth) {

    int c_itr = 0; // convolution counter
    for (auto c = hidden_layer_type.begin(); c != hidden_layer_type.end(); c++) { // loop over each convolution

        for (int m = 0; m < (*c)(nn::depth); m++){ // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth) {
                N = (input_type[nn::height]*input_type[nn::width] - (*c)(nn::F) + 2*(*c)(nn::P))/(*c)(nn::S) + 1;
            }
            
            for (int n = 0; n < N; n++) { // one loop to produce weights for neurons of the filter
                
                for (int d = 0; d < input_type[nn::depth]; d++) { // loop over all input depths
                    
                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> distr(-1,1);

                    if ((*c)(nn::F) == 1) { // if the filter size is of size 1
                        // cout << 5 << endl;
                        // cout << (*c)(nn::F) << endl;
                        hidden_weights[c_itr][m][n][d].resize((*c)(nn::F), 1); // for a conv, filter, neuron, and input depth, specify the weight matrix
                        // cout << hidden_weights[c_itr][m][n][d].rows() << endl;
                        // cout << hidden_weights[c_itr][m][n][d].cols() << endl;
                        // cout << distr(gen) << endl;
                        hidden_weights[c_itr][m][n][d](0,0) = distr(gen);
                    }

                    else { // if filter is greater than 1... next size should be 4 (2x2 filter)
                        
                        hidden_weights[c_itr][m][n][d].resize((*c)(nn::F)/2, (*c)(nn::F)/2); // for a conv, filter, neuron, and input depth, specify the weight matrix

                        for(int i = 0; i < hidden_weights[c_itr][m][n][d].rows(); i++) { // loop over each weight and assign it a random # between -1, 1

                            for(int j = 0; j < hidden_weights[c_itr][m][n][d].cols(); j++) {

                                hidden_weights[c_itr][m][n][d](i,j) = distr(gen);
                            }
                        }
                    }

                }
                for(auto zebra : hidden_weights[c_itr][m][n]) {
                    cout << "Conv: " << c_itr << " filter: " << m << " neuron: " << n << endl;
                    cout << zebra << endl;
                }
            }
        }
        c_itr++; // increase convolution pointer
    }
}

// print network hyperparameters
void nn::nn::print_network() {
    cout << "INPUT: " << input_type.transpose() << endl;
    cout << "OUTPUT: " << output_type.transpose() << endl;
    nn::nn::print_hidden_layers();
}

// print hidden layer hyperparameters
void nn::nn::print_hidden_layers() {
    cout << "HIDDEN: \n";
    for (auto idx = hidden_layer_type.begin(); idx != hidden_layer_type.end(); idx++) {
        cout << (*idx).transpose() << endl;
    }
}



nn::fp::fp() {}
nn::fp::~fp() {}

nn::bp::bp() {}
nn::bp::~bp() {}
// int nn::nn::print_num(int num) { std::cout << num << std::endl; }




