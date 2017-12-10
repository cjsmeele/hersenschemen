#pragma once

// Simple fully connected neural network implementation.

#include <iostream>
#include <vector>
#include <tuple>

namespace nn {

class Neuron;
using layer_t = std::vector<Neuron>;


class Neuron {

    double outputValue;

    using input_t = std::pair<Neuron*, double>; //Neuron and weight
    std::vector<input_t> inputs;


public:
    Neuron(double initialVal = 0)
        : outputValue(initialVal)
        {
    }

    void addInput(layer_t &l);
    void addInput(layer_t &l, const std::vector<double> &weights);
    void addInput(Neuron &n);
    void addInput(Neuron &n, double weight);
    void setWeight(Neuron *np, double weight);
    void setWeights(std::vector<double> w);
    void removeInput(Neuron *np);

    void removeAllInputs()     { inputs.clear(); }
    double getOutput() const   { return outputValue; }
    void   setOutput(double v) { outputValue = v; }

    void calculate();
};

class Net {

    std::vector<layer_t> the_net;

public:
    Net() = default;
    Net(const std::vector<layer_t> &net)
        : the_net(net)
        { }

    void addNeuron(uint layer, Neuron n);
    void addLayer(uint size);
    std::vector<double> run(std::vector<double> input);
    void interConnect();
    Neuron &gibNeuron(uint layer, uint neuron);
    layer_t &gibLayer(uint layer);
};

}
