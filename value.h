#include <vector>
#include <algorithm>

struct Value {
private:
    friend Value operator* (float obj, Value &a);
    friend Value operator+ (float obj, Value &a);
    
    float data;
    float grad = 0;

    std::function<void()> backward_step = nullptr;
    std::vector<Value*> children;

    void build_topo(Value* v, std::vector<Value*> &topo);

    bool check(Value* v, std::vector<Value*> &topo);

public:
    Value ();
    Value(float data);

    Value operator+ (Value &obj);

    Value operator+ (float obj);

    Value operator* (Value &obj);

    Value operator* (float obj);


    Value relu();

    void backward();


    void operator= (int value);

    float get_value();

    float get_grad();

};
