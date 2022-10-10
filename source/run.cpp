#include <iostream>
#include "base.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
using namespace std;

int main() {
    const Eigen::Matrix<int,3,1> input {2,2,2}; // height, width, depth
    const Eigen::Matrix<int,4,1> output {1,1,1,1}; // height, width, depth, activation
    vector<Eigen::Matrix<int,7,1>> hidden {}; // height, width, depth ,activation
    for (int i = 0; i < 2; i++) {
        Eigen::Matrix<int,7,1> layer;
        layer(0,0) = 1;
        layer(1,0) = 1;
        layer(2,0) = 10;
        layer(3,0) = 1;
        layer(4,0) = 1;
        layer(5,0) = 0;
        layer(6,0) = 1;
        hidden.push_back(layer);
    }

    // nn::nn FC = nn::nn(input, hidden, output);
    // FC.print_network();

    unique_ptr<nn::nn> FC_ptr(new nn::nn(input,hidden,output));

    // FC_ptr->print_network();

    // auto test = FC_ptr->get_hidden_weights();

    // FC_ptr->print_layer_sizes();
    // FC_ptr->print_hidden_weights(false);
    // for (auto i : test) {
    //     for (auto j : i) {
    //         for (auto k : j) {
    //             for (auto u : k) {
    //                 cout << u << endl;
    //             }
    //         }
    //     }
    // }

    vector<float> i {0,1,2,3};

    for (auto z : i) {
        cout << z << endl;
    }

    // for(auto z = i.begin(); z != i.end(); z++) {
    //     // cout << "Z: " << (*z) << endl;
    //     // cout << "i.end(): " << (i.end() << endl;
    //     cout << i.end() - z << endl;
    //     // cout << (*z) << endl;
    // }

    // try {
    //     cout << "Mason\n";
    //     cout << << endl;
    //     // cout << i << endl;
    // } catch(...) {
    //     cout << "Pesson" << endl;
    // }



    // vector<vector<vector<float>>> test;
    // test.resize(3,vector<vector<float>>(4,vector<float>(5)));

    // for (auto i : test) {
    //     for (auto j : i) {
    //         for (auto k : j) {
    //             cout << k << endl;
    //         }
    //     }
    // }



    // Eigen::Matrix<float,3,3> m1 = Eigen::Array33f::Ones()*-1;
    // Eigen::Matrix<float,3,3> m2 = Eigen::Array33f::Zero(3,3);

    // cout << m1.cwiseProduct(m2) << endl;
    // cout << m1.cwiseMax(m2) << endl;

    // cout << Eigen::Rand::balanced
    
    // random_device rd;
    // mt19937 gen(rd());
    // uniform_real_distribution<> distr(-1,1);

    // for (int i=0; i<10; i++) {
    //     cout << distr(gen) << endl;
    // }

    // FC.print_num(5);
    return 0;
}








