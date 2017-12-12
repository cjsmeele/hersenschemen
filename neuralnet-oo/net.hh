#pragma once

// Simple fully connected neural network implementation.

#include "common.hh"
#include <vector>
#include <tuple>
#include <memory>
#include "neuron.hh"

namespace nn {

    template<typename T = double,
             typename ActivationPolicy = SigmoidActivationPolicy,
             typename Eta = std::ratio<1,10>>
    class Net : ActivationPolicy {

    public:
        using Neuron_t = Neuron<T,ActivationPolicy,Eta>;
        using layer_t  = std::vector<Neuron_t>;

    private:
        std::vector<layer_t> layers;

    public:
        Net() = default;
        Net(std::vector<layer_t> &&layers)
            : layers(layers)
            { }

        Net(uint inputCount,
            uint outputCount,
            uint hiddenCount,
            uint neuronsPerHiddenLayer) {

            layers.resize(2 + hiddenCount);
            layers.front().resize(inputCount + 1);
            layers.back().resize(outputCount);
            for (size_t i = 1; i < layers.size() - 1; ++i) {
                layers[i].resize(neuronsPerHiddenLayer + 1);
                layers[i][0].setValue(Neuron_t::biasValue);
            }

            layers.front()[0].setValue(Neuron_t::biasValue);

            connect();
        }

        struct Connection {
            uint srcL, srcN;
            uint dstL, dstN;
            T weight;
        };
        void connect() {
            for (size_t i = 1; i < layers.size(); ++i) {
                for (size_t j = (i == layers.size()-1 ? 0 : 1);
                     j < layers[i].size(); ++j) {
                    for (auto &n: layers[i-1])
                        layers[i][j].addInput(&n);
                }
            }
        }
        void connect(const Connection &c) {
            getNeuron(c.dstL, c.dstN)
                .addInput(&getNeuron(c.srcL, c.srcN),
                          c.weight);
        }
        void connect(const std::vector<Connection> &cs) {
            for (const auto &c : cs)
                connect(c);
        }

        std::vector<T> run(const std::vector<T> &input) {
            // Set input neurons.
            for (size_t i = 1; i < layers[0].size(); i++)
                layers[0][i].setValue(input[i-1]);

            // propagateForward all other neurons.
            for (size_t i = 1; i < layers.size(); i++) {
                for (auto &n : layers[i])
                    n.propagateForward();
            }

            // Collect output values.
            std::vector<T> res;
            for (auto &n : layers.back())
                res.push_back(n.getValue());

            return res;
        }

        void train(const std::vector<T> &input,
                   const std::vector<T> &expected) {

            auto results = run(input);

            // T delta = 0;

            // for (size_t i = 0; i < layers.back().size(); ++i) {
            //     auto &n = layers.back()[i];
            //     auto a = n.getValue();
            //     auto y = expected[i];
            //     delta += pow(y - a, 2);
            // }
            // T mse = delta / (2 * layers.back().size());
            // std::cout << "MSE:" << mse << "\n";

            for (int i = layers.size()-1; i >= 0; --i) {

                for (size_t j = 0; j < layers[i].size(); ++j) {
                    auto &n = layers[i][j];
                    if (i == (int)layers.size()-1)
                        n.propagateBackward(expected[j]);
                    else
                        n.propagateBackward();
                }
            }

            for (size_t i = 0; i < layers.size(); ++i)
                for (size_t j = 0; j < layers[i].size(); ++j)
                    layers[i][j].flush();

        }

        const std::vector<layer_t> &getLayers() const { return layers; }
              std::vector<layer_t> &getLayers()       { return layers; }

        const layer_t &getLayer(uint layer) const { return layers[layer]; }
              layer_t &getLayer(uint layer)       { return layers[layer]; }

        const Neuron_t &getNeuron(uint layer, uint neuron) const { return layers[layer][neuron]; }
              Neuron_t &getNeuron(uint layer, uint neuron)       { return layers[layer][neuron]; }
    };

    template<typename S, typename... N>
    S &operator<<(S& s, const Net<N...> &v) {
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
