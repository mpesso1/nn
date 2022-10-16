#include <iostream>
#include "base.h"
#include <Eigen/Core>
// #include "/usr/include/eigen-3.4.0/Eigen/src/Core/DenseBase.h"
#include <vector>
#include <memory>
#include <random>
using namespace std;

int main() {
    const Eigen::Matrix<int,3,1> input {4,4,2}; // height, width, depth
    const Eigen::Matrix<int,4,1> output {2,2,2,1}; // height, width, depth, activation
    vector<Eigen::Matrix<int,7,1>> hidden {}; // height, width, depth ,activation
    for (int i = 0; i < 1; i++) {
        Eigen::Matrix<int,7,1> layer;
        layer(0,0) = 3; // height
        layer(1,0) = 3; // width
        layer(2,0) = 4; // depth
        layer(3,0) = 0; // activation function
        layer(4,0) = 2; // F
        layer(5,0) = 0; // P
        layer(6,0) = 1; // S
        hidden.push_back(layer);
    }
    for (int i = 0; i < 1; i++) {
        Eigen::Matrix<int,7,1> layer;
        layer(0,0) = 2; // height
        layer(1,0) = 2; // width
        layer(2,0) = 2; // depth
        layer(3,0) = 1; // activation function
        layer(4,0) = 2; // F 
        layer(5,0) = 0; // P
        layer(6,0) = 1; // S
        hidden.push_back(layer);
    }

    // nn::nn FC = nn::nn(input, hidden, output);
    // FC.print_network();

    unique_ptr<nn::nn> FC_ptr(new nn::nn(input,hidden,output));

    Eigen::Matrix<float,4,4> test {{1,2,1,1},{1,1,1,6},{3,1,1,1},{1,1,1,7}};
    // cout<< test.size() << endl;
    vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> test_o {test,test};
    vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> test_o2 {test_o};

    Eigen::Matrix<float,2,2> test_out {{1,1},{1,1}};
    vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> test_out1 {test_out,test_out};
    vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> test_out2 {test_out1};

    FC_ptr->forward(test_o2, test_out2);



    // vector<float> tt {0,1,2,3,4};

    // for (auto i = tt.end()-1; i != tt.begin()-1; i--) {
    //     cout << *i << endl;
    // }

    // auto test = FC_ptr->get_hidden_weights();

    // FC_ptr->print_layer_sizes();
    // FC_ptr->print_hidden_weights();
    // for (auto i : test) {
    //     for (auto j : i) {
    //         for (auto k : j) {
    //             for (auto u : k) {
    //                 cout << u << endl;
    //             }
    //         }
    //     }
    // }

    // vector<float> i {0,1,2,3};

    // for (auto z : i) {
    //     cout << z << endl;
    // }

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

    // Eigen::Matrix<float,9,1> m;

    // for (int i = 0; i != m.size(); i++) {
    //     m(i,0) = i;
    // }
    // cout << m << endl;

    // cout << m.reshaped<Eigen::RowMajor>(3,3) << endl;

    // for (int i=0; i<10; i++) {
    //     if (i == 3) {
    //         if (i == 3)
    //             continue;
    //     }
    //     cout << i << endl;
    // }
    
    // FC.print_num(5);
    return 0;
}








