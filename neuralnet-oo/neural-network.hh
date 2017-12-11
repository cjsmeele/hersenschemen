#pragma once

// Simple fully connected neural network implementation.

#include "common.hh"
#include <vector>
#include <tuple>
#include <memory>

namespace nn {

class Neuron;
using layer_t = std::vector<Neuron>;


class Neuron {

    static constexpr double eta = 0.1;

    struct Link {
        Neuron *src, *dst;
        double weight;
        double weight_; // Temporary value during updates.

        Link(Neuron *src, Neuron *dst, double weight)
            : src(src),
              dst(dst),
              weight(weight),
              weight_(weight)
            { }
    };

    /// Used during training.
    double sum   = 0;
    double delta = 0;

    /// The current value.
    double value;

    std::vector<std::unique_ptr<Link>>  inputs;
    std::vector<Link*> outputs;

    // Wiskunde == leesbaar
    static double g(double z);
    static double g_(double z);

public:
    Neuron(double initialValue = 0)
        : value(initialValue)
        { }

    void addInput(Neuron *n);
    void addInput(Neuron *n, double weight);

    const std::vector<std::unique_ptr<Link>> &getInputs() const { return inputs; }

    double getValue() const   { return value; }
    void   setValue(double v) { value = v; }

    void propagateForward();
    void propagateBackward(double y = 0);

    /// Flush weight changes after all weights have been calculated by
    /// propagateBackward.
    void flush();
};

class Net {

    std::vector<layer_t> layers;

public:
    Net() = default;
    Net(std::vector<layer_t> &&layers)
        : layers(std::move(layers))
        { }

    Net(uint inputCount,
        uint hiddenCount,
        uint outputCount,
        uint neuronsPerHiddenLayer);

    struct Connection {
        uint srcL, srcN;
        uint dstL, dstN;
        double weight;
    };
    void connect();
    void connect(const Connection &c);
    void connect(const std::vector<Connection> &c);

    std::vector<double> run(const std::vector<double> &input);

    void train(const std::vector<double> &input,
               const std::vector<double> &expected);

    const std::vector<layer_t> &getLayers() const { return layers; }

    const Neuron &getNeuron(uint layer, uint neuron) const { return layers[layer][neuron]; }
          Neuron &getNeuron(uint layer, uint neuron)       { return layers[layer][neuron]; }
    const layer_t &getLayer(uint layer) const { return layers[layer]; }
          layer_t &getLayer(uint layer)       { return layers[layer]; }
};

template<typename S>
S &operator<<(S& s, const Net &v) {
    for (const auto &l : v.getLayers()) {
        s << "[";
        for (const auto &n : l) {
            s << ' ';
            if (n.getInputs().size()) {
                s << '<';
                for (auto &o : n.getInputs()) {
                    s << ' ' << o->weight;
                }
                s << '>';
            }
            // s << n.getValue();
        }
        s << " ]\n";
    }
    return s;
}

}
