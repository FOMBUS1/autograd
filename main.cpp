#include <iostream>
#include "value.h"

int main() {
    Value x = 1.5;
    Value y = 1.3;
    Value k = 2*x;
    Value z = k+y;
    z.backward();
    std::cout << x << std::endl;
    std::cout << y << std::endl;
}