#include "common.hh"
#include "net.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

using namespace nn;

void manual_nor() {
    std::cout << "\nManually trained NOR gate:\n";

    Net<StepActivationPolicy> ketnet;

    auto &layers = ketnet.getLayers();
    layers.resize(2);
    layers[0].resize(4);
    layers[1].resize(1);

    layers[0][0].setValue(1);

    ketnet.connect({ { 0,0, 1,0,  1 },
                     { 0,1, 1,0, -1 },
                     { 0,2, 1,0, -1 },
                     { 0,3, 1,0, -1 } });

    std::cout << ketnet << "\n";

    std::cout << "[ 0 0 0 ] => " << ketnet.run({ 0, 0, 0 }) << "\n";
    std::cout << "[ 0 0 1 ] => " << ketnet.run({ 0, 0, 1 }) << "\n";
    std::cout << "[ 0 1 0 ] => " << ketnet.run({ 0, 1, 0 }) << "\n";
    std::cout << "[ 0 1 1 ] => " << ketnet.run({ 0, 1, 1 }) << "\n";
    std::cout << "[ 1 0 0 ] => " << ketnet.run({ 1, 0, 0 }) << "\n";
    std::cout << "[ 1 0 1 ] => " << ketnet.run({ 1, 0, 1 }) << "\n";
    std::cout << "[ 1 1 0 ] => " << ketnet.run({ 1, 1, 0 }) << "\n";
    std::cout << "[ 1 1 1 ] => " << ketnet.run({ 1, 1, 1 }) << "\n";
}

// Opdracht 4.2
void manual_adder() {
    std::cout << "\nManually trained half adder:\n";

    Net<StepActivationPolicy> ketnet;

    auto &layers = ketnet.getLayers();
    layers.resize(4);
    layers[0].resize(3);
    layers[1].resize(2);
    layers[2].resize(3);
    layers[3].resize(2);

    for (size_t i = 0; i < 3; i++)
        layers[i][0].setValue(1);

    ketnet.connect({ /* 1,1: NAND 1.   */ { 0,0, 1,1,  2 }, { 0,1, 1,1, -1 }, { 0,2, 1,1, -1 },
                     /* 2,1: NAND 2.   */ { 1,0, 2,1,  2 }, { 0,1, 2,1, -1 }, { 1,1, 2,1, -1 },
                     /* 2,2: NAND 3.   */ { 0,2, 2,2, -1 }, { 1,0, 2,2,  2 }, { 1,1, 2,2, -1 },
                     /* 3,0: Inverter. */ { 2,0, 3,0,  1 }, { 1,1, 3,0, -1 },
                     /* 3,1: NAND 4.   */ { 2,0, 3,1,  2 }, { 2,1, 3,1, -1 }, { 2,2, 3,1, -1 }});

    std::cout << ketnet << "\n";

    std::cout << "[ 0 0 ] => " << ketnet.run({ 0, 0 }) << "\n";
    std::cout << "[ 0 1 ] => " << ketnet.run({ 0, 1 }) << "\n";
    std::cout << "[ 1 0 ] => " << ketnet.run({ 1, 0 }) << "\n";
    std::cout << "[ 1 1 ] => " << ketnet.run({ 1, 1 }) << "\n";
}

void inverter() {
    std::cout << "\nInverter:\n";
    Net<SigmoidActivationPolicy,std::ratio<1,10>> net(1, 1, 0, 0);
    for (int i = 0; i < 20000; ++i) {
        net.train({ 0 }, { 1 });
        net.train({ 1 }, { 0 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 ] => " << net.run({ 0 }) << "\n";
    std::cout << "[ 1 ] => " << net.run({ 1 }) << "\n";
}

void eq3() {
    std::cout << "\n3-way equality:\n";
    Net<> net(3, 1, 2, 4);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0, 0, 0 }, { 1 });
        net.train({ 0, 0, 1 }, { 0 });
        net.train({ 0, 1, 0 }, { 0 });
        net.train({ 0, 1, 1 }, { 0 });
        net.train({ 1, 0, 0 }, { 0 });
        net.train({ 1, 0, 1 }, { 0 });
        net.train({ 1, 1, 0 }, { 0 });
        net.train({ 1, 1, 1 }, { 1 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 0 0 ] => " << net.run({ 0, 0, 0 }) << "\n";
    std::cout << "[ 0 0 1 ] => " << net.run({ 0, 0, 1 }) << "\n";
    std::cout << "[ 0 1 0 ] => " << net.run({ 0, 1, 0 }) << "\n";
    std::cout << "[ 0 1 1 ] => " << net.run({ 0, 1, 1 }) << "\n";
    std::cout << "[ 1 0 0 ] => " << net.run({ 1, 0, 0 }) << "\n";
    std::cout << "[ 1 0 1 ] => " << net.run({ 1, 0, 1 }) << "\n";
    std::cout << "[ 1 1 0 ] => " << net.run({ 1, 1, 0 }) << "\n";
    std::cout << "[ 1 1 1 ] => " << net.run({ 1, 1, 1 }) << "\n";
}

void xor_() {
    std::cout << "\n3-way XOR-ish gate:\n";
    Net<> net(3, 1, 2, 4);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0, 0, 0 }, { 0 });
        net.train({ 0, 0, 1 }, { 1 });
        net.train({ 0, 1, 0 }, { 1 });
        net.train({ 0, 1, 1 }, { 0 });
        net.train({ 1, 0, 0 }, { 1 });
        net.train({ 1, 0, 1 }, { 0 });
        net.train({ 1, 1, 0 }, { 0 });
        net.train({ 1, 1, 1 }, { 0 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 0 0 ] => " << net.run({ 0, 0, 0 }) << "\n";
    std::cout << "[ 0 0 1 ] => " << net.run({ 0, 0, 1 }) << "\n";
    std::cout << "[ 0 1 0 ] => " << net.run({ 0, 1, 0 }) << "\n";
    std::cout << "[ 0 1 1 ] => " << net.run({ 0, 1, 1 }) << "\n";
    std::cout << "[ 1 0 0 ] => " << net.run({ 1, 0, 0 }) << "\n";
    std::cout << "[ 1 0 1 ] => " << net.run({ 1, 0, 1 }) << "\n";
    std::cout << "[ 1 1 0 ] => " << net.run({ 1, 1, 0 }) << "\n";
    std::cout << "[ 1 1 1 ] => " << net.run({ 1, 1, 1 }) << "\n";
}

void adder() {
    std::cout << "\nHalf adder:\n";

    Net<> net(2, 2, 1, 2);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0, 0 }, { 0, 0 });
        net.train({ 0, 1 }, { 1, 0 });
        net.train({ 1, 0 }, { 1, 0 });
        net.train({ 1, 1 }, { 0, 1 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 0 ] => " << net.run({ 0, 0 }) << "\n";
    std::cout << "[ 0 1 ] => " << net.run({ 0, 1 }) << "\n";
    std::cout << "[ 1 0 ] => " << net.run({ 1, 0 }) << "\n";
    std::cout << "[ 1 1 ] => " << net.run({ 1, 1 }) << "\n";
}

void divisible_by_three() {
    std::cout << "\n6-bit divisible by three checker (YMMV):\n";

    Net<> net(6, 1, 1, 5);
    for (int i = 0; i < 20000; ++i) {
        for (int j = 0; j < 64; ++j) {
            if (j % 3 == 0) {
                net.train({ (double)((j>>5)&1),
                            (double)((j>>4)&1),
                            (double)((j>>3)&1),
                            (double)((j>>2)&1),
                            (double)((j>>1)&1),
                            (double)( j    &1) },
                        { (double)(j % 3 == 0) });
            }
            net.train({ (double)((j>>5)&1),
                        (double)((j>>4)&1),
                        (double)((j>>3)&1),
                        (double)((j>>2)&1),
                        (double)((j>>1)&1),
                        (double)( j    &1) },
                      { (double)(j % 3 == 0) });
        }
    }

    std::cout << net << "\n";

    for (int i = 0; i < 64; ++i) {
        std::cout << "[";
        for (int j = 0; j < 6; ++j)
            std::cout << ' ' << ((i>>(5-j))&1);
        std::cout << " ] => "
                  << net.run({ (double)((i>>5)&1),
                               (double)((i>>4)&1),
                               (double)((i>>3)&1),
                               (double)((i>>2)&1),
                               (double)((i>>1)&1),
                               (double)( i    &1) }) << "\n";
    }
}

enum class Iris : unsigned {
    setosa = 0,
    versicolor,
    virginica
};

struct iris_t {
    double sepal_l, sepal_w, petal_l, petal_w;
    Iris i;
};

iris_t make_iris(std::string &raw_iris) {
    // std::cout << raw_iris << " this was raw\n"; // DEBUG
    return iris_t {
            std::stof(raw_iris.substr(0,3)), // what is string view?
            std::stof(raw_iris.substr(4,3)), // baby don't hurt me
            std::stof(raw_iris.substr(8,3)), // i swear this works on my dataset
            std::stof(raw_iris.substr(12,3)),//
            [&](){
                // lol who needs generic code when you can just assume your data's always right :)
                switch (raw_iris[24]) { // Yes really efficient 10/10 quality code what are maps amirite?
                    case 'o': return Iris::setosa;
                    case 's': return Iris::versicolor;
                    case 'g': return Iris::virginica;
                    default: std::cout << "IF_YOU_READ_THIS_GET_A_BETTER_DATASET"; return Iris::setosa; // self documenting and everythin
                }
            }()
    };
}

template<typename S> // Printing beautifull flowers
S &operator<<(S& s, iris_t t) {
    s << '[' << t.sepal_l << ' ' << t.sepal_w << ' ' << t.petal_l << ' ' << t.petal_w << ' ';
    switch (t.i) { //lol
        case Iris::setosa:     s << "Iris-setosa";     break;
        case Iris::versicolor: s << "Iris-versicolor"; break;
        case Iris::virginica:  s << "Iris-virginica";  break;
    }
    s << "]\n";
    return s;
}

void iris_dataset() {
    Net<> net(4, 3, 2, 10);
    std::ifstream rdata("../../iris/bezdekIris.data.txt");
    std::vector<iris_t> data;
    std::vector<iris_t> test_data;
    data.reserve(150);
    {
        std::string line;
        while (std::getline(rdata, line) && line.size())
            data.push_back(make_iris(line));
    }
    // std::cout << data;
    // Who needs performance when you can shuffle vectors?
    {
        std::random_device r;
        std::mt19937 engine(r());
        std::shuffle(data.begin(), data.end(), engine); // every day i'm shuffelin'
    }
    // get some validation data
    for (int _ = 0; _ < 50; _++) { test_data.push_back(data.back()); data.pop_back(); }
    // train network
    for (int i = 0; i < 10000; i++) {
        for (const auto &d: data) {
            net.train({ d.sepal_l,
                        d.sepal_w,
                        d.petal_l,
                        d.petal_w },
                      { d.i == Iris::setosa,
                        d.i == Iris::versicolor,
                        d.i == Iris::virginica });
        }
        std::cout << "trained round " << i << " of 10000\n";
    }
    std::cout << "\nNetwork is done:\n" << net << "\nCalculating score:\n";
    {
        int correct, wrong, total;
        for (const auto &d: test_data) {
            auto results = net.run({ d.sepal_l,
                                     d.sepal_w,
                                     d.petal_l,
                                     d.petal_w });
            int sum = 0;
            for (auto x: results) { x = std::round(x); sum += x; }
            if (sum != 1) { wrong++; continue; } // Bad network! You can't pull a 50/50 on me
            if (results[static_cast<unsigned>(d.i)]) correct++;
        }
        total = correct + wrong;
        std::cout << "Network got "
                  << correct
                  << " out of "
                  << total
                  << " for a total score of "
                  << (double) correct / total
                  << '\n';
    }
}

int main() {
    srand(time(NULL));
    std::cout.precision(2);

//    manual_adder();
//    manual_nor();
//    inverter();
//    eq3();
//    xor_();
//    adder();
//    divisible_by_three();

    iris_dataset();

    return 0;
}
