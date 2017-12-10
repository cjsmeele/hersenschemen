#pragma once

// Simple fully connected neural network implementation.

#include <iostream>
#include <vector>
#include <tuple>

namespace nn {

class Neuron;
using layer_t = std::vector<Neuron>;


class Neuron {

    /// The current value.
    double outputValue;

    using input_t = std::pair<Neuron*, double>; //Neuron and weight
    std::vector<input_t> inputs;


public:
    Neuron(double initialValue = 0)
        : outputValue(initialValue)
        { }

    void addInput(layer_t &l);
    void addInput(layer_t &l, const std::vector<double> &weights);
    void addInput(Neuron *n);
    void addInput(Neuron *n, double weight);
    void setWeight(Neuron *np, double weight);
    void setWeights(const std::vector<double> &w);
    void removeInput(Neuron *np);

    void removeAllInputs()     { inputs.clear(); }
    double getOutput() const   { return outputValue; }
    void   setOutput(double v) { outputValue = v; }

    void calculate();
};

class Net {

    std::vector<layer_t> layers;

public:
    Net() = default;
    Net(const std::vector<layer_t> &layers)
        : layers(layers)
        { }

    Net(uint inputCount,
        uint hiddenCount,
        uint outputCount,
        uint neuronsPerHiddenLayer);

    void addNeuron(uint layer, Neuron n);
    void addLayer(uint size);

    struct Connection {
        uint srcL, srcN;
        uint dstL, dstN;
        double weight;
    };
    void connect();
    void connect(const Connection &c);
    void connect(const std::vector<Connection> &c);

    std::vector<double> run(std::vector<double> input);

    const Neuron &getNeuron(uint layer, uint neuron) const { return layers[layer][neuron]; }
          Neuron &getNeuron(uint layer, uint neuron)       { return layers[layer][neuron]; }
    const layer_t &getLayer(uint layer) const { return layers[layer]; }
          layer_t &getLayer(uint layer)       { return layers[layer]; }
};

}
