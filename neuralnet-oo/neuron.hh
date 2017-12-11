#include "common.hh"
#include <memory>
#include <ratio>

namespace nn {

    struct SigmoidActivationPolicy {
        double g(double z)  { return 1 / (1 + exp(-z)); }
        double g_(double z) { return g(z) * (1 - g(z)); }
    };

    struct StepActivationPolicy {
        double g(double z)  { return z > 0; }
        double g_(double z) { return z*0; }
    };

    template<typename ActivationPolicy = SigmoidActivationPolicy,
             typename Eta = std::ratio<1,10>>
    class Neuron : ActivationPolicy {

    public:
        static constexpr double eta        = (double)Eta::num / Eta::den;
        static constexpr double biasValue  = -1;

    private:
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
        using ActivationPolicy::g;
        using ActivationPolicy::g_;

    public:
        Neuron(double initialValue = 0)
            : value(initialValue)
            { }

        void addInput(Neuron *n) {
            addInput(n, ((double)rand() / RAND_MAX) * 2 - 1);
        }
        void addInput(Neuron *n, double weight) {
            inputs.emplace_back(new Link(n, this, weight));
            n->outputs.push_back(inputs.back().get());
        }

        const std::vector<std::unique_ptr<Link>> &getInputs() const { return inputs; }

        double getValue() const   { return value; }
        void   setValue(double v) { value = v; }

        void propagateForward() {
            if (inputs.empty())
                // This is a neuron without inputs (i.e. bias).
                return;

            sum = 0;
            for (const auto &l : inputs)
                sum += l->src->value * l->weight;

            value = g(sum);
        }

        void propagateBackward(double y = 0) {

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

        /// Flush weight changes after all weights have been calculated by
        /// propagateBackward.
        void flush() {
            for (auto &l : inputs)
                l->weight = l->weight_;
        }

    };

}
