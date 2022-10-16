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

/**
 * @brief Construct a new nn::nn::nn object
 * 
 * @param input 
 * @param hidden 
 * @param output 
 */
nn::nn::nn(const Eigen::Matrix<int,3,1>& input, vector<Eigen::Matrix<int,7,1>>& hidden, const Eigen::Matrix<int,4,1>& output) : 
    input_type(input), output_type(output), hidden_layer_type(hidden)
    {
        cout << "\033[1;31m NETWORK ARCHITECTURE SUCCESFULLY ADDED \033[0m\n";
        
        nn::init_weights();
        cout << "\033[1;34m WEIGHTS SUCCESFULLY SET \033[0m\n";
        
        nn::init_neurons();
        cout << "\033[1;32m NEURONS SUCCESFULLY SET \033[0m\n";
    }

/**
 * @brief Destroy the nn::nn::nn object
 * 
 */
nn::nn::~nn() {}


/**
 * @brief init weights to random values C(M(N(d(Fx x Fy)))) vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> hidden_weights
 * 
 * @param param_share_layer 
 * @param param_share_depth 
 */
void nn::nn::init_weights(bool param_share_layer, bool param_share_depth) {
    
    hidden_weights.resize(hidden_layer_type.size()); // resize weight memory to the correct # of convolutions
    bias.resize(hidden_layer_type.size());

    int c_idx = 0; // convolution counter


    for (auto c_type = hidden_layer_type.begin(); c_type != hidden_layer_type.end(); c_type++) { // loop over each convolution

        hidden_weights[c_idx].resize((*c_type)(depth)); // resize the number of desired filters for the specific convolution
        bias[c_idx].resize((*c_type)(depth));

        for (int m = 0; m < (*c_type)(depth); m++){ // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth) {
                if (c_idx == 0) {
                    N = (input_type[height]*input_type[width] - pow((*c_type)(nn::F),2) + 2*(*c_type)(nn::P))/(*c_type)(nn::S) + 1;
                }
                else {
                    N = ((*(c_type-1))(height)*(*(c_type-1))(width) - pow((*c_type)(nn::F),2) + 2*(*c_type)(nn::P))/(*c_type)(nn::S) + 1;
                }
            }
            
            if (N != round(N)) {
                cout << "Error: number of neurons not obtainable.\n";
                cout << "Check hyperparameters of layer.\n";
                cout << "N: " << N << endl;
            }

            hidden_weights[c_idx][m].resize(N); // for a certain convolution and filter, specify the number of neurons for that filter
            bias[c_idx][m].resize(N);

            for (int n = 0; n < N; n++) { // for each neuron of that filter of that convolution

                int Depth = 0;
                
                if (c_idx ==  0) {
                    Depth = input_type[depth];
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }
                else {
                    Depth = hidden_weights[c_idx-1].size();
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }
                
                for (int d = 0; d < Depth; d++) { // loop over all input depths
                    
                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> distr(-1,1);

                    bias[c_idx][m][n][d] = distr(gen); // give a bias to each depth

                    if ((*c_type)(nn::F) == 1) { // if the filter size is of size 1
                        hidden_weights[c_idx][m][n][d].resize((*c_type)(nn::F), 1); // for a conv, filter, neuron, and input depth, specify the weight matrix
                        hidden_weights[c_idx][m][n][d](0,0) = distr(gen);
                    }

                    else { // if filter is greater than 1... next size should be 4 (2x2 filter)
                        
                        hidden_weights[c_idx][m][n][d].resize((*c_type)(nn::F), (*c_type)(nn::F)); // for a conv, filter, neuron, and input depth, specify the weight matrix

                        for(int i = 0; i < hidden_weights[c_idx][m][n][d].rows(); i++) { // loop over each weight and assign it a random # between -1, 1

                            for(int j = 0; j < hidden_weights[c_idx][m][n][d].cols(); j++) {


                                hidden_weights[c_idx][m][n][d](i,j) = distr(gen);
                            }
                        }
                    }
                }
            }
        }
        c_idx++; // increase convolution pointer
    }
};


void nn::nn::init_neurons() {
    
    neuron_outputs.resize(hidden_layer_type.size()); // resize to the number of convolutions

    int c_idx = 0; // convolution counter

    for (auto c = neuron_outputs.begin(); c != neuron_outputs.end(); c++) { // loop through each convolution

        neuron_outputs[c_idx].resize(hidden_layer_type[c_idx](depth)); // resize to the number of filters in the convolution

        for (auto m = c->begin(); m != c->end(); m++) { // loop through each filter of the convolution

            (*m) = Eigen::ArrayXXf::Zero( hidden_layer_type[c_idx](height), hidden_layer_type[c_idx](width) ); // define filter height and width
            // cout << m->rows() << "," << m->cols() << endl;  // check the number of rows and columns is correct
        }

        c_idx++; // increment the 
    
    }
}


void nn::nn::forward(vector< vector< Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>>& in, vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>>& out) {
    int c_idx = 0; // convolution
    int m_idx = 0; // filter
    int n_idx = 0; // neuron
    int d_idx = 0; // input depth
    int w_idx = 0; // weight / weight matrix

    batch_size = in.size();
    init_output();
    cout << "\033[1;37m OUTPUT SUCCESFULLY SET \033[0m\n";

    int b {}; // temporary

    int j_ref = 0; // to be used later.. sollution to parameter sharing
    int i_ref = 0; // to be used later..

    vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> store;
    for (auto c = hidden_weights.begin(); c != hidden_weights.end(); c++) { // loop over convolutions
        
        store.resize(c->size()); // set size to the number of filters of the convolution

        int filter_dem = hidden_layer_type[c_idx](F);
        
        m_idx = 0;
        for (auto m = c->begin(); m != c->end(); m++) { // loop over each filter of convolution
            
            store[m_idx] = Eigen::ArrayXXf::Zero(neuron_outputs[c_idx][m_idx].size(),1); // set the size of matrix to number of nerons in the filter
            // cout << "m\n";
            w_idx = 0;
            for (auto n = m->begin(); n != m->end(); n++) { // loop over each neuron of filter
                
                // loop over input
                n_idx = 0;
                for (int j = 0; j != neuron_outputs[c_idx][m_idx].cols(); j++) { // loop over width of input
                    for (int i = 0; i != neuron_outputs[c_idx][m_idx].rows(); i++) { // loop over height of input
                        // cout << "n\n";
                        
                        
                        d_idx = 0;
                        for (auto d = n->begin(); d != n->end(); d++) { // loop over each input depth
                            // cout << ((in[b][d_idx].block(i,j,filter_dem,filter_dem)).cwiseProduct((*d))).sum() + bias[c_idx][m_idx][w_idx][d_idx] << endl;
                            // cout << (in[b][d_idx].block(i,j,filter_dem,filter_dem)).size() << endl;
                            if (c_idx == 0) {
                                // cout << "p\n";
                                // cout << bias[c_idx][m_idx][w_idx][d_idx] << endl;
                                store[m_idx](n_idx,0) += ((in[b][d_idx].block(i,j,filter_dem,filter_dem)).cwiseProduct((*d))).sum() + bias[c_idx][m_idx][w_idx][d_idx];
                                // cout << "e\n";
                            }
                            else {
                                
                                store[m_idx](n_idx,0) += ((neuron_outputs[c_idx-1][m_idx].block(i,j,filter_dem,filter_dem)).cwiseProduct((*d))).sum() + bias[c_idx][m_idx][w_idx][d_idx];
                            }
                            activate(store[m_idx](n_idx,0), hidden_layer_type[c_idx](act));
                            d_idx++;
                        }
                        n_idx++;
                    }
                }

                neuron_outputs[c_idx][m_idx] = store[m_idx].reshaped(sqrt(neuron_outputs[c_idx][m_idx].size()), sqrt(neuron_outputs[c_idx][m_idx].size()));
                if (c_idx == (hidden_weights.size() - 1)) {
                    output_train[b][m_idx] = neuron_outputs[c_idx][m_idx];
                    cost[b][m_idx] = output_train[b][m_idx] - out[b][m_idx]; // NEEDS MORE ABILITY... calculate the cost comparing the output to the desired output specified
                }
                w_idx++;
            }
            m_idx++;
        }
        cout << "Neuron outputs for conv: " << c_idx << endl;
        print_neuron_outputs();
        c_idx++;
    }

    cout << "OUTPUT: " << endl;
    print_output();
    
}

void nn::nn::backward() {

}

void nn::nn::init_output() {
    output_train.resize(batch_size);
    for (int b = 0; b != batch_size; b++) {
        output_train[b].resize(output_type(depth));
        for (int d = 0; d != output_type(depth); d++) {
            output_train[b][d] = Eigen::ArrayXXf::Zero(output_type(height),output_type(width));
        }
    }
}

void nn::nn::activate(float& val, int func) {
    if (func == 0) {
        if (val>0) {
            val = val;
        }
        else {
            val = 0;
        }
    }
}

void nn::nn::print_neuron_outputs() {
    for (auto& c : neuron_outputs) {
        for (auto& m : c) {
            cout << m.transpose() << endl;
        }
    }
}

void nn::nn::print_output() {
    for (auto& b : output_train) {
        for (auto& d : b) {
            cout << d << endl;
        }
    }
}

void nn::nn::train(vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> input, vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> output, int batch) {
    if (input.size() != output.size()) {
        cout << "Input data size: " << input.size() << endl;
        cout << "Output data size: " << output.size() << endl;
        cout << "MUST MATCH\n";
        return;
    }
    else {

    }
}


void nn::nn::propogate(vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> i, vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> o) {

    int c_itr = 0; // convolution counter
    int d_idx = 0; // input depth counter
    int m_idx = 0; // filter depth counter
    int n_idx = 0; // ountput neuron counter
    auto neuron_input = i;
    int N = 0;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> next_input {};

    // vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> next_input{};

    for (auto c = hidden_weights.begin(); c != hidden_weights.end(); c++) { // loop over each convolution
        for (auto m = c->begin(); m != c->end(); m++) { // loop over each filter of the convolution
            for(auto n = m->begin(); n != m->end(); n++) {
                next_input.resize(nn::height);

                if (n->size() == 1) { // parameter sharing... only one weight matrix for each filter
                    int filter_dem = sqrt(hidden_layer_type[c_itr](nn::F));
                    for (int i = 0; i != hidden_layer_type[c_itr](nn::width); i++) {
                        for (int j = 0; j != hidden_layer_type[c_itr](nn::height); j++) {
                            for (auto d = n->begin(); d != n->end(); d++) {
                                next_input[m_idx](n_idx) += (neuron_input[d_idx].block(i,j,filter_dem,filter_dem).cwiseProduct((*d))).sum();
                            }
                        }
                    }
                }
                else if (n->size() > 1) { // no parameter sharing

                }
                else { // error

                }
                

            }
            next_input[m_idx].reshaped<Eigen::RowMajor>(sqrt(m->size()),sqrt(m->size()));

            m_idx++;
        }
        c_itr++; // increase convolution pointer
    }
}


vector< vector< vector< vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> nn::nn::get_hidden_weights(){
    return hidden_weights;
}

void nn::nn::print_layer_sizes(bool param_share_layer, bool param_share_depth ) {

    cout << "# of convolutions: " << hidden_weights.size() << endl;
    int c_idx = 0; // convolution counter

    for (auto c_type = hidden_layer_type.begin(); c_type != hidden_layer_type.end(); c_type++) { // loop over each convolution

        cout << "# of filters for conv " << c_idx << " : " << hidden_weights[c_idx].size() << endl;

        for (int m = 0; m < (*c_type)(nn::depth); m++){ // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth) {
                if (c_idx == 0) {
                    N = (input_type[height]*input_type[width] - (*c_type)(nn::F) + 2*(*c_type)(nn::P))/(*c_type)(nn::S) + 1;
                }
                else {
                    N = ((*(c_type-1))(height)*(*(c_type-1))(width) - (*c_type)(nn::F) + 2*(*c_type)(nn::P))/(*c_type)(nn::S) + 1;
                }
            }
            if (N != round(N)) {
                cout << "Error: number of neurons not obtainable.\n";
                cout << "Check hyperparameters of layer.";
                cout << "N: " << N << endl;
            }
            
            cout << "# of weight tensors for conv " << c_idx << " and filter " << m << " : " << hidden_weights[c_idx][m].size() << endl; // for a certain covolution and filter, specify the number neurons for that filter

            // int N = 1; // all neurons will have the same weights..

            for (int n = 0; n < N; n++) { // one loop to produce weights for neurons of the filter
                
                int Depth = 0;
                
                if (c_idx ==  0) {
                    Depth = input_type[depth];
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }
                else {
                    Depth = hidden_weights[c_idx-1].size();
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }

                cout << "# of input depth for conv " << c_idx << " and filter " << m << " and neuron " << n << " : " << hidden_weights[c_idx][m][n].size() << endl;; // for a certain convolution, filter, and neuron, specify the depth of the input

                for (int d = 0; d < Depth; d++) { // loop over all input depths

                    cout << "# of weights for conv " << c_idx << " and filter " << m << " and neuron " << n << " and depth " << d << " : " << hidden_weights[c_idx][m][n][d].size() << endl;
                    cout << "# of biases for conv " << c_idx << " and filter " << m << " and neuron " << n << " and depth " << d << " : " << bias[c_idx][m][n].size() << endl;
                }
            }
        }
        c_idx++; // increase convolution pointer
    }
}

void nn::nn::print_hidden_weights(bool param_share_layer, bool param_share_depth) {

    int c_idx = 0; // convolution counter
    for (auto c_type = hidden_layer_type.begin(); c_type != hidden_layer_type.end(); c_type++) { // loop over each convolution

        for (int m = 0; m < (*c_type)(nn::depth); m++){ // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth) {
                if (c_idx == 0) {
                    N = (input_type[height]*input_type[width] - (*c_type)(nn::F) + 2*(*c_type)(nn::P))/(*c_type)(nn::S) + 1;
                }
                else {
                    N = ((*(c_type-1))(height)*(*(c_type-1))(width) - (*c_type)(nn::F) + 2*(*c_type)(nn::P))/(*c_type)(nn::S) + 1;
                }
            }
            if (N != round(N)) {
                cout << "Error: number of neurons not obtainable.\n";
                cout << "Check hyperparameters of layer.";
                cout << "N: " << N << endl;
            }
            
            for (int n = 0; n < N; n++) { // one loop to produce weights for neurons of the filter
                int d = 0;
                for(auto zebra : hidden_weights[c_idx][m][n]) {
                    cout << "Conv: " << c_idx << " filter: " << m << " neuron: " << n << " depth: " << d << endl;
                    cout << zebra << endl;
                    // cout << 
                    d++;
                }

            }
        }
        c_idx++; // increase convolution pointer
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




