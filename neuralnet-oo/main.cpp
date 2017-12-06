#include "neural-network.hh"


int main() {
    Net ketnet {{
        layer_t{ Neuron{}, Neuron{}, Neuron{} },
        layer_t{ Neuron{}, Neuron{}, Neuron{} },
        layer_t{ Neuron{}, Neuron{}, Neuron{} },
    }};
    
    std::cout << "Skynet sais hello" << std::endl;
    return 0;
}
