# autograd
This struct creates objects that can automatically calculate their gradients.

```
Value a = -3.0;
Value b = 2.0;
Value c = a * b;
Value e = 10.0;
Value d = e+c;
d.backward();
```
After the code completed gradients will be:
```
a.grad = 2.0;
b.grad = -3.0;
c.grad = 1.0;
e.grad = 1.0;
d_grad = 1.0;
```
