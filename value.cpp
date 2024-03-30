#include "value.h"

void Value::build_topo(Value* v, std::vector<Value*> &topo) {
    if (check(v, topo)) {
        for(int i = 0; i < v->children.size(); i++) {
            build_topo(v->children[i], topo);
        }
        topo.push_back(v);
    }
}

bool Value::check(Value* v, std::vector<Value*> &topo) {
    bool unique = true;
    for (int i = 0; i < topo.size(); i++) {
        if(v == topo[i]) {
            unique = false;
            return unique;
        }
    }
    return unique;
}

Value::Value () {
    this->data = 0;
    this->grad = 0;
}

Value::Value(float data) {
    this->data = data;
}

Value Value::operator+ (Value &obj) {
    Value out = Value(this->data + obj.data);
    out.children.push_back(this);
    out.children.push_back(&obj);

    auto back = [&] () {
        this->grad += out.get_grad();
        obj.grad += out.get_grad();
    };

    out.backward_step = back;
    return out;
}

Value Value::operator+ (float obj) {
    Value out = Value(this->data + obj);

    out.children.push_back(this);

    auto back = [&out, obj, this] () {
        this->grad += out.get_grad();
    };

    out.backward_step = back;

    return out;
}

Value Value::operator* (Value &obj) {
    Value out = Value(this->data * obj.data);

    out.children.push_back(this);
    out.children.push_back(&obj);

    auto back = [&] () {
        this->grad += obj.data * out.grad;
        obj.grad += this->data * out.grad;
    };
    out.backward_step = back;
    return out;
}

Value Value::operator* (float obj) {
    Value out = Value(this->data * obj);
    out.children.push_back(this);

    auto back = [&out, obj, this] () {
        this->grad += obj * out.grad;
    };
    out.backward_step = back;
    return out;
}


Value Value::relu() {
    Value out = this->data;
    out.children.push_back(this);
    if (this->data < 0) {
        out.data = 0;
    }
    auto back = [&] () {
        this->grad += (out.data > 0) * out.get_grad(); 
   };
    out.backward_step = back;
    return out;
}

void Value::backward() {
    std::vector<Value*> topo;
    build_topo(this, topo);

    this->grad=1;
    std::reverse(topo.begin(), topo.end());
    for(int i = 0; i < topo.size(); i++) {
        if (topo[i]->backward_step != nullptr) {
            topo[i]->backward_step();
        }
    }
}


void Value::operator= (int value) {
    this->data = value;
}

float Value::get_value() {
    return this->data;
}

float Value::get_grad() {
    return this->grad;
}


Value operator* (float obj, Value &a) {
    Value out = Value(obj + a.data);
    out.children.push_back(&a);

    auto back = [&out, obj, &a] () {
        a.grad += obj * out.grad;
    };
    out.backward_step = back;
    return out;
}

Value operator+ (float obj, Value &a) {
    Value out = Value(a.data + obj);
    out.children.push_back(&a);
    auto back = [&out, obj, &a] () {
        a.grad += out.get_grad();
    };

    out.backward_step = back;

    return out;
};