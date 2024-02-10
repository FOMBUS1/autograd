#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>

struct Value {
public:
    float data;
    float grad = 0;

    std::function<void()> backward_step = nullptr;
    std::vector<Value*> children;

    void build_topo(Value* v, std::vector<Value*> &topo) {
        if (check(v, topo)) {
            for(int i = 0; i < v->children.size(); i++) {
                build_topo(v->children[i], topo);
            }
            topo.push_back(v);
        }
    };

    bool check(Value* v, std::vector<Value*> &topo) {
        bool unique = true;
        for (int i = 0; i < topo.size(); i++) {
            if(v == topo[i]) {
                unique = false;
                return unique;
            }
        }
        return unique;
    };

public:
    Value(float data) {
        this->data = data;
    };

    Value operator+ (Value &obj) {
        Value out = Value(this->data + obj.data);

        out.children.push_back(this);
        out.children.push_back(&obj);

        auto back = [&] () {
            this->grad += out.grad;
            obj.grad += out.grad;
        };

        out.backward_step = back;

        return out;
    };

    Value operator* (Value &obj) {
        Value out = Value(this->data * obj.data);

        out.children.push_back(this);
        out.children.push_back(&obj);

        auto back = [&] () {
            this->grad += obj.data * out.grad;
            obj.grad += this->data * out.grad;
        };
        out.backward_step = back;
        return out;
    };

    void backward() {
        std::vector<Value*> topo;
        build_topo(this, topo);

        this->grad=1;
        std::reverse(topo.begin(), topo.end());
        for(int i = 0; i < topo.size(); i++) {
            if (topo[i]->backward_step != nullptr) {
                topo[i]->backward_step();
            }
        }
    };


    void operator= (int value) {
        this->data = value;
    };

    float get_value() {
        return this->data;
    };

    float get_grad() {
        return this->grad;
    }

};

int main() {
    Value a = -3.0;
    Value b = 2.0;
    Value c = a * b;
    Value e = 10.0;
    Value d = e+c;
    d.backward();
    //da/dd = ((a*b) + e)' = b = 2
    //db/dd = ((a*b) + e)' = a = -3
    std::cout << "a_grad: " << a.get_grad() << std::endl;
    std::cout << "b_grad: " << b.get_grad() << std::endl;
}