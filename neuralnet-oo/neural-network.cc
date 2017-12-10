#include "neural-network.hh"

#include <utility>
#include <cstdlib>
#include <cmath>

namespace nn {

double randomWeight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

void Neuron::addInput(Neuron *n) {
    addInput(n, randomWeight());
}


void Neuron::addInput(Neuron *n, double weight) {
    inputs.emplace_back(new Link(n, this, weight));
    n->outputs.push_back(inputs.back().get());
}

double Neuron::g(double z) {
    return 1 / (1 + exp(-z));
}

double Neuron::g_(double z) {
    return g(z) * (1 - g(z));
}

void Neuron::update() {
    if (inputs.empty())
        // This is a neuron without inputs (i.e. bias).
        return;

    sum = 0;
    for (const auto &l : inputs)
        sum += l->src->value * l->weight;

    value = g(sum);
}

Net::Net(uint inputCount,
         uint outputCount,
         uint hiddenCount,
         uint neuronsPerHiddenLayer) {

    layers.resize(2 + hiddenCount);
    layers.front().resize(inputCount + 1);
    layers.back().resize(outputCount);
    for (size_t i = 1; i < layers.size() - 1; ++i) {
        layers[i].resize(neuronsPerHiddenLayer);
        layers[i][0].setValue(-1);
    }

    layers.front()[0].setValue(-1);

    connect();
}

void Net::connect() {
    for (size_t i = 1; i < layers.size(); i++)
        for (auto &n: layers[i])
            for (auto &n2: layers[i-1])
                n.addInput(&n2);
}

void Net::connect(const Connection &c) {
    getNeuron(c.dstL, c.dstN)
        .addInput(&getNeuron(c.srcL, c.srcN),
                  c.weight);
}
void Net::connect(const std::vector<Net::Connection> &cs) {
    for (const auto &c : cs)
        connect(c);
}

std::vector<double> Net::run(const std::vector<double> &input) {
    // Set input neurons.
    for (size_t i = 1; i < layers[0].size(); i++)
        layers[0][i].setValue(input[i-1]);

    // Update all other neurons.
    for (size_t i = 1; i < layers.size(); i++) {
        for (auto &n : layers[i])
            n.update();
    }

    // Collect output values.
    std::vector<double> res;
    for (auto &n : layers.back())
        res.push_back(n.getValue());

    return res;
}

void Neuron::doeNouEensEvenConformDeMaatschappelijkeNormenEnWaarden(double y) {

    auto a = value;

    if (outputs.size()) {
        // Wiskunde == leesbaar
        double sigmaDeltaPAccent = 0;
        for (auto ol : outputs)
            sigmaDeltaPAccent += ol->weight * ol->dst->delta;

        delta = g_(sum) * sigmaDeltaPAccent;

    } else {
        delta = g_(sum) * (y - a);
    }

    if (inputs.size()) {
        for (auto &l : inputs) {
            l->weight_ = l->weight + eta * l->src->value * delta;
        }
    }

}

void Neuron::flush() {
    for (auto &l : inputs)
        l->weight = l->weight_;
}


void Net::train(const std::vector<double> &input,
                const std::vector<double> &expected) {

    auto results = run(input);

    // double delta = 0;

    // for (size_t i = 0; i < layers.back().size(); ++i) {
    //     auto &n = layers.back()[i];
    //     auto a = n.getValue();
    //     auto y = expected[i];
    //     delta += pow(y - a, 2);
    // }
    // double mse = delta / (2 * layers.back().size());
    // std::cout << "MSE:" << mse << "\n";

    for (int i = layers.size()-1; i >= 0; --i) {

        for (size_t j = 0; j < layers[i].size(); ++j) {
            auto &n = layers[i][j];
            if (i == (int)layers.size()-1)
                n.doeNouEensEvenConformDeMaatschappelijkeNormenEnWaarden(expected[j]);
            else
                n.doeNouEensEvenConformDeMaatschappelijkeNormenEnWaarden();
        }
    }

    for (size_t i = 0; i < layers.size(); ++i)
        for (size_t j = 0; j < layers[i].size(); ++j)
            layers[i][j].flush();

}

}
