#include "base.hpp"

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
nn::nn::nn(const Eigen::Matrix<int, 3, 1> &input,
           vector<Eigen::Matrix<int, 7, 1>> &hidden,
           const Eigen::Matrix<int, 4, 1> &output) : input_type(input),
                                                     output_type(output),
                                                     hidden_layer_type(hidden)
{
    cout << "\033[1;31m NETWORK ARCHITECTURE SUCCESFULLY ADDED \033[0m\n";

    nn::init_weights();
    cout << "\033[1;34m WEIGHTS SUCCESFULLY SET \033[0m\n";
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
void nn::nn::init_weights(bool param_share_layer,
                          bool param_share_depth)
{
    hidden_weights.resize(hidden_layer_type.size()); // resize weight memory to the correct # of convolutions
    bias.resize(hidden_layer_type.size());

    int c_idx = 0; // convolution counter

    for (auto c_type = hidden_layer_type.begin();
         c_type != hidden_layer_type.end();
         c_type++)
    { // loop over each convolution

        hidden_weights[c_idx].resize((*c_type)(depth)); // resize the number of desired filters for the specific convolution
        bias[c_idx].resize((*c_type)(depth));

        for (int m = 0;
             m < (*c_type)(depth);
             m++)
        { // loop over each filter of the convolution

            int N = 1; // Specifying the number of weight matices needed... if one
                       // then this means there is parameter sharing over a matrix layer.
                       // In the case that you have a FC (fully connected) between layers
                       // then the number of weight would equal the number of input layer matrix elements
            if (!param_share_layer && !param_share_depth)
            {
                cout << "HEYYY WTF\n";
                if (c_idx == 0)
                {
                    N = (input_type[height] * input_type[width] - pow((*c_type)(nn::F), 2) + 2 * (*c_type)(nn::P)) / (*c_type)(nn::S) + 1;
                }
                else
                {
                    N = ((*(c_type - 1))(height) * (*(c_type - 1))(width)-pow((*c_type)(nn::F), 2) + 2 * (*c_type)(nn::P)) / (*c_type)(nn::S) + 1;
                }

                if (N != round(N))
                {
                    cout << "Error: number of neurons not obtainable.\n";
                    cout << "Check hyperparameters of layer.\n";
                    cout << "N: " << N << endl;
                    exit(0);
                }
            }

            hidden_weights[c_idx][m].resize(N); // for a certain convolution and filter, specify the number of neurons for that filter
            bias[c_idx][m].resize(N);

            for (int n = 0; n < N; n++)
            { // for each neuron of that filter of that convolution

                int Depth = 0;

                if (c_idx == 0)
                {
                    Depth = input_type[depth];
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }
                else
                {
                    Depth = hidden_weights[c_idx - 1].size();
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }

                for (int d = 0; d < Depth; d++)
                { // loop over all input depths

                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> distr(-1, 1);

                    bias[c_idx][m][n][d] = distr(gen); // give a bias to each depth

                    if ((*c_type)(nn::F) == 1)
                    {                                                               // if the filter size is of size 1
                        hidden_weights[c_idx][m][n][d].resize((*c_type)(nn::F), 1); // for a conv, filter, neuron, and input depth, specify the weight matrix
                        hidden_weights[c_idx][m][n][d](0, 0) = distr(gen);
                    }

                    else
                    { // if filter is greater than 1... next size should be 4 (2x2 filter)

                        hidden_weights[c_idx][m][n][d].resize((*c_type)(nn::F), (*c_type)(nn::F)); // for a conv, filter, neuron, and input depth, specify the weight matrix

                        for (int i = 0; i < hidden_weights[c_idx][m][n][d].rows(); i++)
                        { // loop over each weight and assign it a random # between -1, 1

                            for (int j = 0; j < hidden_weights[c_idx][m][n][d].cols(); j++)
                            {

                                hidden_weights[c_idx][m][n][d](i, j) = distr(gen);
                            }
                        }
                    }
                }
            }
        }
        c_idx++; // increase convolution pointer
    }
};

void nn::nn::init_neurons()
{

    neuron_outputs.resize(batch_size);
    int b_idx = 0;

    for (auto b = neuron_outputs.begin(); b != neuron_outputs.end(); b++)
    {

        neuron_outputs[b_idx].resize(hidden_layer_type.size()); // resize to the number of convolutions

        int c_idx = 0; // convolution counter

        for (auto c = b->begin(); c != b->end(); c++)
        { // loop through each convolution

            neuron_outputs[b_idx][c_idx].resize(hidden_layer_type[c_idx](depth)); // resize to the number of filters in the convolution

            for (auto m = c->begin(); m != c->end(); m++)
            { // loop through each filter of the convolution

                (*m) = Eigen::ArrayXXf::Zero(hidden_layer_type[c_idx](height),
                                             hidden_layer_type[c_idx](width)); // define filter height and width
                // cout << m->rows() << "," << m->cols() << endl;  // check the number of rows and columns is correct
            }

            c_idx++; // increment the
        }
    }
}

void nn::nn::init_output()
{
    output_train.resize(batch_size);
    for (int b = 0; b != batch_size; b++)
    {
        output_train[b].resize(output_type(depth));
        for (int d = 0; d != output_type(depth); d++)
        {
            output_train[b][d] = Eigen::ArrayXXf::Zero(output_type(height),
                                                       output_type(width));
        }
    }
}

void nn::nn::init_cost()
{
    cost.resize(batch_size);
    cost_grad.resize(batch_size);
    for (int b = 0; b != batch_size; b++)
    {
        cost[b].resize(output_type(depth));
        cost_grad[b].resize(output_type(depth));
        for (int d = 0; d != output_type(depth); d++)
        {
            cost[b][d] = Eigen::ArrayXXf::Zero(output_type(height), output_type(width));
            cost_grad[b][d] = Eigen::ArrayXXf::Zero(output_type(height), output_type(width));
        }
    }
}

void nn::nn::forward(vector<vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> &in,
                     vector<vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> &out)
{
    int c_idx = 0; // convolution
    int m_idx = 0; // filter
    int n_idx = 0; // neuron
    int d_idx = 0; // input depth
    int w_idx = 0; // weight / weight matrix

    input_train = in;

    if (!memory_ready)
    {
        batch_size = in.size();
        nn::init_neurons();
        cout << "\033[1;32m NEURONS SUCCESFULLY SET \033[0m\n";
        init_output();
        cout << "\033[1;37m OUTPUT SUCCESFULLY SET \033[0m\n";
        init_cost();
        cout << "\033[1;39m COST SUCCESFULLY SET \033[0m\n";

        memory_ready = true;
    }

    int j_ref = 0; // to be used later.. sollution to parameter sharing
    int i_ref = 0; // to be used later..

    loss_tot = 0;

    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> store;

    for (int b = 0; b != batch_size; b++)
    {

        for (auto c = hidden_weights.begin(); c != hidden_weights.end(); c++)
        { // loop over convolutions

            store.resize(c->size()); // set size to the number of filters of the convolution

            int filter_dem = hidden_layer_type[c_idx](F);

            m_idx = 0;
            for (auto m = c->begin(); m != c->end(); m++)
            { // loop over each filter of convolution

                store[m_idx] = Eigen::ArrayXXf::Zero(neuron_outputs[b][c_idx][m_idx].size(), 1); // set the size of matrix to number of nerons in the filter

                w_idx = 0;
                for (auto n = m->begin(); n != m->end(); n++)
                { // loop over each neuron of filter

                    // loop over input
                    n_idx = 0;
                    for (int j = 0; j != neuron_outputs[b][c_idx][m_idx].cols(); j++)
                    { // loop over width of input
                        for (int i = 0; i != neuron_outputs[b][c_idx][m_idx].rows(); i++)
                        { // loop over height of input

                            d_idx = 0;
                            for (auto d = n->begin(); d != n->end(); d++)
                            { // loop over each input depth

                                store[m_idx](n_idx, 0) += (c_idx == 0) ? ((in[b][d_idx].block(i, j, filter_dem, filter_dem)).cwiseProduct((*d))).sum() + bias[c_idx][m_idx][w_idx][d_idx] : ((neuron_outputs[b][c_idx - 1][m_idx].block(i, j, filter_dem, filter_dem)).cwiseProduct((*d))).sum() + bias[c_idx][m_idx][w_idx][d_idx];

                                d_idx++;
                            }

                            activate(store[m_idx](n_idx, 0), hidden_layer_type[c_idx](act));
                            n_idx++;
                        }
                    }

                    neuron_outputs[b][c_idx][m_idx] = store[m_idx].reshaped(sqrt(neuron_outputs[b][c_idx][m_idx].size()),
                                                                            sqrt(neuron_outputs[b][c_idx][m_idx].size()));

                    if (c_idx == (hidden_weights.size() - 1))
                    {

                        output_train[b][m_idx] = neuron_outputs[b][c_idx][m_idx];
                        cost[b][m_idx] = loss(output_train[b][m_idx], out[b][m_idx], output_type(act));
                        cost_grad[b][m_idx] = dloss(output_train[b][m_idx], out[b][m_idx], output_type(act));
                    }
                    w_idx++;
                }
                m_idx++;
            }

            // cout << "Neuron outputs for conv: " << c_idx << endl;
            // print_neuron_outputs();
            c_idx++;
        }

        // cout << "OUTPUT: " << endl;
        // print_output();
    }

    cout << "\033[1;30m FORWARD SUCCESFULL \033[0m\n";
}

void nn::nn::backward()
{
    int c_idx{}; // convolution
    int m_idx{}; // filter
    int n_idx{}; // neuron
    int d_idx{}; // input depth
    int w_idx{}; // weight / weight matrix

    vector<vector<vector<vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>>>> dW = hidden_weights;
    vector<vector<vector<vector<float>>>> dB = bias;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> dX_in;
    vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> dX_out;

    for (int b = 0; b != batch_size; b++)
    {

        c_idx = hidden_weights.size() - 1; // convolution
        for (auto c = hidden_weights.end() - 1; c != hidden_weights.begin() - 1; c--)
        { // for each convolution. c->size() := number of filters in conv / depth of conv

            int filter_dem = hidden_layer_type[c_idx](F); // filter size used to compute output neurons

            // computationally messy needs to be cleaned... shouldnt be copying this much... same goes for dW and dB
            dX_in.resize(c->size(),
                         Eigen::ArrayXXf::Zero(hidden_layer_type[c_idx](height),
                                               hidden_layer_type[c_idx](width))); // set the size of gradient input to current depth

            if (c_idx == hidden_weights.size() - 1)
            { // if on the first layer of back propogation then input is cost gradient
                dX_in = cost_grad[b];
                cout << "COST_GRAD: " << dX_in[0] << endl;
                cout << "Size of cost grad: " << dX_in.size() << endl;
            }
            else
            {
                dX_in = dX_out;
            }
            if (c_idx != 0)
            { // no need to set output once we reach the inputs of forward propogation.. will just be updating the weights
                dX_out.resize((c - 1)->size(),
                              Eigen::ArrayXXf::Zero(hidden_layer_type[c_idx - 1](height),
                                                    hidden_layer_type[c_idx - 1](width))); // set ouput gradient to the size of input depth on forward prop.
                // could also get the same value from current n->size() but dont want to be iterativley re calling size and potentially changing values
            }
            // --

            m_idx = c->size() - 1;
            for (auto m = c->end() - 1; m != c->begin() - 1; m--)
            { // for each filter. m->size() := number of weight matrices in filter

                w_idx = m->size() - 1;
                for (auto n = m->end() - 1; n != m->begin() - 1; n--)
                { // for each weight matrix. n->size() := depth of the input neurons to the filter

                    n_idx = neuron_outputs[b][c_idx][m_idx].size() - 1;
                    for (int j = 0; j != neuron_outputs[b][c_idx][m_idx].cols(); j++)
                    { // for width of matrix
                        for (int i = 0; i != neuron_outputs[b][c_idx][m_idx].rows(); i++)
                        { // for height of matrix

                            d_idx = n->size() - 1;
                            for (auto d = n->end() - 1; d != n->begin() - 1; d--)
                            { //

                                if (c_idx != 0)
                                {

                                    dX_out[d_idx].block(i, j, filter_dem, filter_dem) += ((*d) * dX_in[m_idx](i, j)).cwiseProduct(deactivate(neuron_outputs[b][c_idx - 1][m_idx].block(i, j, filter_dem, filter_dem), hidden_layer_type[c_idx](act)));

                                    dW[c_idx][m_idx][w_idx][d_idx] -= neuron_outputs[b][c_idx - 1][d_idx].block(i, j, filter_dem, filter_dem) * dX_in[m_idx](i, j);
                                }
                                else
                                {

                                    dW[c_idx][m_idx][w_idx][d_idx] -= input_train[b][d_idx].block(i, j, filter_dem, filter_dem) * dX_in[m_idx](i, j);
                                }

                                dB[c_idx][m_idx][w_idx][d_idx] -= dX_in[m_idx](i, j);

                                d_idx--;
                            }
                            n_idx--;
                        }
                    }
                    w_idx--;
                }
                m_idx--;
            }
            c_idx--;
        }
    }

    hidden_weights = dW;
    bias = dB;

    cout << "\033[1;30m BACKWARD SUCCESFULL \033[0m\n";
}

void nn::nn::activate(float &val, int func)
{
    switch (func)
    {
    case LINEAR:
        val = val;
        break;

    case RELU:
        if (val > 0)
        {
            val = val;
        }
        else
        {
            val = 0;
        }
        break;

    case TANH:
        val = tanh(val);
        break;

    case SIGMOID:
        val = 1 / (1 + exp(-val));
        break;

    default:
        break;
    }
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> nn::nn::deactivate(const Eigen::Block<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> &val, int func)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> buff = Eigen::ArrayXXf::Zero(val.rows(), val.cols());

    float revert{};

    switch (func)
    {
    case LINEAR:
        for (int i = 0; i != val.rows(); i++)
        {
            for (int j = 0; j != val.cols(); j++)
            {
                buff(i, j) = 1;
            }
        }
        break;

    case RELU:
        for (int i = 0; i != val.rows(); i++)
        {
            for (int j = 0; j != val.cols(); j++)
            {
                if (val(i, j) == 0)
                {
                    buff(i, j) = 0;
                }
                else
                {
                    buff(i, j) = 1;
                }
            }
        }
        break;

    case TANH:
        for (int i = 0; i != val.rows(); i++)
        {
            for (int j = 0; j != val.cols(); j++)
            {
                revert = atanh(val(i, j));
                buff(i, j) = 1 - tanh(revert) * tanh(revert);
            }
        }
        break;

    case SIGMOID:
        for (int i = 0; i != val.rows(); i++)
        {
            for (int j = 0; j != val.cols(); j++)
            {
                revert = -log(1 - val(i, j));
                buff(i, j) = 1 / (1 - exp(-revert)) * (1 - (1 / (1 - exp(-revert))));
            }
        }
        break;
    default:
        break;
    }

    return buff;
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> nn::nn::loss(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &out,
                                                                  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &des,
                                                                  int func)
{
    switch (func)
    {
    case LS_LOSS:
        loss_tot = 0.5 * (-(des - out).cwiseProduct(-(des - out))).sum();
        // cout << "loss tot: " << loss_tot << endl;
        // cout << "des: " << des << endl;
        // cout << "out: " << out << endl;
        // cout << "des-out: " << des-out << endl;
        // cout << "product: " << (des - out).cwiseProduct((des - out)) << endl;
        // cout << "sum: " << (des - out).cwiseProduct((des - out)).sum() << endl;
        return 0.5 * (-(des - out).cwiseProduct(-(des - out)));

    default:
        cout << "Loss funtion undetectable\n";
        break;
    }
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> nn::nn::dloss(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &out,
                                                                   const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &des,
                                                                   int func)
{
    switch (func)
    {
    case LS_LOSS:
        return -(des - out) * 0.0001;

    default:
        cout << "Loss function undetectable\n";
        break;
    }
}

void nn::nn::calc_average_loss()
{
    cout << "loss tot: " << loss_tot << endl;
    cout << (cost.size() * cost[0].size() * cost[0][0].size()) << endl;
    loss_avg = loss_tot / (cost.size() * cost[0].size() * cost[0][0].size());
}

void nn::nn::plot_loss()
{
    system("python3 plot_loss.py");
}

void nn::nn::print_loss()
{
    calc_average_loss();
    cout << "LOSS: " << loss_avg << endl;
    cout << "cost_grad here: " << cost_grad[0][0] << endl;
}

void nn::nn::print_neuron_outputs()
{
    for (auto &b : neuron_outputs)
    {
        for (auto &c : b)
        {
            for (auto &m : c)
            {
                cout << m.transpose() << endl;
            }
        }
    }
}

void nn::nn::print_output()
{
    for (auto &b : output_train)
    {
        for (auto &d : b)
        {
            cout << d << endl;
        }
    }
}

void nn::nn::print_layer_sizes(bool param_share_layer, bool param_share_depth)
{

    cout << "# of convolutions: " << hidden_weights.size() << endl;
    int c_idx = 0; // convolution counter

    for (auto c_type = hidden_layer_type.begin(); c_type != hidden_layer_type.end(); c_type++)
    { // loop over each convolution

        cout << "# of filters for conv " << c_idx << " : " << hidden_weights[c_idx].size() << endl;

        for (int m = 0; m < (*c_type)(nn::depth); m++)
        { // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth)
            {
                if (c_idx == 0)
                {
                    N = (input_type[height] * input_type[width] - (*c_type)(nn::F) + 2 * (*c_type)(nn::P)) / (*c_type)(nn::S) + 1;
                }
                else
                {
                    N = ((*(c_type - 1))(height) * (*(c_type - 1))(width) - (*c_type)(nn::F) + 2 * (*c_type)(nn::P)) / (*c_type)(nn::S) + 1;
                }
            }
            if (N != round(N))
            {
                cout << "Error: number of neurons not obtainable.\n";
                cout << "Check hyperparameters of layer.";
                cout << "N: " << N << endl;
            }

            // for a certain covolution and filter, specify the number neurons for that filter
            cout << "# of weight tensors for conv " << c_idx << " and filter " << m << " : " << hidden_weights[c_idx][m].size() << endl;

            // int N = 1; // all neurons will have the same weights..

            for (int n = 0; n < N; n++)
            { // one loop to produce weights for neurons of the filter

                int Depth = 0;

                if (c_idx == 0)
                {
                    Depth = input_type[depth];
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }
                else
                {
                    Depth = hidden_weights[c_idx - 1].size();
                    hidden_weights[c_idx][m][n].resize(Depth); // for a certain convolution, filter, and neuron, specify the depth of the input
                    bias[c_idx][m][n].resize(Depth);
                }

                cout << "# of input depth for conv " << c_idx
                     << " and filter " << m
                     << " and neuron " << n
                     << " : " << hidden_weights[c_idx][m][n].size() << endl;
                ; // for a certain convolution, filter, and neuron, specify the depth of the input

                for (int d = 0; d < Depth; d++)
                { // loop over all input depths

                    cout << "# of weights for conv " << c_idx
                         << " and filter " << m
                         << " and neuron " << n
                         << " and depth " << d
                         << " : " << hidden_weights[c_idx][m][n][d].size() << endl;
                    cout << "# of biases for conv " << c_idx
                         << " and filter " << m
                         << " and neuron " << n
                         << " and depth " << d
                         << " : " << bias[c_idx][m][n].size() << endl;
                }
            }
        }
        c_idx++; // increase convolution pointer
    }
}

void nn::nn::print_hidden_weights(bool param_share_layer, bool param_share_depth)
{

    int c_idx = 0; // convolution counter
    for (auto c_type = hidden_layer_type.begin(); c_type != hidden_layer_type.end(); c_type++)
    { // loop over each convolution

        for (int m = 0; m < (*c_type)(nn::depth); m++)
        { // loop over each filter of the convolution

            int N = 1;
            if (!param_share_layer && !param_share_depth)
            {
                if (c_idx == 0)
                {
                    N = (input_type[height] * input_type[width] - (*c_type)(nn::F) + 2 * (*c_type)(nn::P)) / (*c_type)(nn::S) + 1;
                }
                else
                {
                    N = ((*(c_type - 1))(height) * (*(c_type - 1))(width) - (*c_type)(nn::F) + 2 * (*c_type)(nn::P)) / (*c_type)(nn::S) + 1;
                }
            }
            if (N != round(N))
            {
                cout << "Error: number of neurons not obtainable.\n";
                cout << "Check hyperparameters of layer.";
                cout << "N: " << N << endl;
            }

            for (int n = 0; n < N; n++)
            { // one loop to produce weights for neurons of the filter
                int d = 0;
                for (auto zebra : hidden_weights[c_idx][m][n])
                {
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

void nn::nn::print_network()
{
    cout << "INPUT: " << input_type.transpose() << endl;
    cout << "OUTPUT: " << output_type.transpose() << endl;
    nn::nn::print_hidden_layers();
}

void nn::nn::print_hidden_layers()
{
    cout << "HIDDEN: \n";
    for (auto idx = hidden_layer_type.begin(); idx != hidden_layer_type.end(); idx++)
    {
        cout << (*idx).transpose() << endl;
    }
}
