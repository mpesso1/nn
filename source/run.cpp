#include <iostream>
#include "base.hpp"
#include <Eigen/Core>
#include <vector>
#include <memory>
#include <random>
#include <opencv2/core.hpp>
#include </usr/include/opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <boost/version.hpp>
#include <boost/asio.hpp>
// #include <boost/asio/io_service.hpp>
// #include <boost/asio/io_context.hpp>
// #include <boost/asio/steady_timer.hpp>

void print(const boost::system::error_code& ) {
    std::cout << "hello\n";
}


using namespace std;
using namespace boost::json;

int main() {
    const Eigen::Matrix<int,3,1> input {1,1,2}; // height, width, depth
    const Eigen::Matrix<int,4,1> output {1,1,2,10}; // height, width, depth, activation
    vector<Eigen::Matrix<int,7,1>> hidden {}; // height, width, depth ,activation
    for (int i = 0; i < 1; i++) {
        Eigen::Matrix<int,7,1> layer;
        layer(0,0) = 1; // height
        layer(1,0) = 1; // width
        layer(2,0) = 30; // depth
        layer(3,0) = 1; // activation function
        layer(4,0) = 1; // F
        layer(5,0) = 0; // P
        layer(6,0) = 1; // S
        hidden.push_back(layer);
    }
    for (int i = 0; i < 1; i++) {
        Eigen::Matrix<int,7,1> layer;
        layer(0,0) = 1; // height
        layer(1,0) = 1; // width
        layer(2,0) = 30; // depth
        layer(3,0) = 1; // activation function
        layer(4,0) = 1; // F
        layer(5,0) = 0; // P
        layer(6,0) = 1; // S
        hidden.push_back(layer);
    }
    for (int i = 0; i < 1; i++) {
        Eigen::Matrix<int,7,1> layer;
        layer(0,0) = 1; // height
        layer(1,0) = 1; // width
        layer(2,0) = 2; // depth
        layer(3,0) = 0; // activation function
        layer(4,0) = 1; // F 
        layer(5,0) = 0; // P
        layer(6,0) = 1; // S
        hidden.push_back(layer);
    }

    // nn::nn FC = nn::nn(input, hidden, output);
    // FC.print_network();
    std::cout << "Using Boost "     
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;

    unique_ptr<nn::nn> FC_ptr(new nn::nn(input,hidden,output));

    // Eigen::Matrix<float,4,4> test {{1,2,1,1},{1,1,1,6},{3,1,1,1},{1,1,1,7}};
    // // cout<< test.size() << endl;
    // vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> test_o {test,test};
    // vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> test_o2 {test_o};

    // Eigen::Matrix<float,2,2> test_out {{1,1},{1,1}};
    // vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> test_out1 {test_out,test_out};
    // vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> test_out2 {test_out1};

    // FC_ptr->forward(test_o2, test_out2);
    // std::cout << "Using Boost "     
    //         << BOOST_VERSION / 100000     << "."  // major version
    //         << BOOST_VERSION / 100 % 1000 << "."  // minor version
    //         << BOOST_VERSION % 100                // patch level
    //         << std::endl;


    Eigen::Matrix<float,1,1> test {1};
    Eigen::Matrix<float,1,1> testb {3};
    // // cout<< test.size() << endl;
    vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> test_o {test,testb};
    vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> test_o2 {test_o};

    Eigen::Matrix<float,1,1> test_out {5};
    Eigen::Matrix<float,1,1> test_outb {15};
    vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>> test_out1 {test_out,test_outb};
    vector<vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>>> test_out2 {test_out1};

    // for (int i = 0; i != 500; i++) {
    //     FC_ptr->forward(test_o2, test_out2);
    //     FC_ptr->backward();
    //     FC_ptr->print_loss();
    // }

    ofstream file_id;
    file_id.open("file.txt");

    // boost::json::value v;

    value v = {
    { "pi", 3.141 },
    { "happy", true },
    { "name", "Boost" },
    { "nothing", nullptr },
    { "answer", {
        { "everything", 42 } } },
    {"list", {1, 0, 2}},
    {"object", {
        { "currency", "USD" },
        { "value", 42.99 }
            } }
    };

    // cout << v << endl;
    // boost::asio::io_service io;
    boost::asio::io_context io;
    boost::asio::steady_timer t(io, boost::asio::chrono::seconds(5));
    t.async_wait(&print);
    cout << "Hey\n";
    // t.wait();
    io.run();
    cout << "Hey\n";

    return 0;
}


