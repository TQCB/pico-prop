import sys
sys.path.append(r"/mnt/c/Users/rapha/My Drive (raphael.pb.rialland@gmail.com)/Work/data_science/projects/deci_deriv_project")
sys.path.append(r"/mnt/c/Users/rapha/My Drive (raphael.pb.rialland@gmail.com)/Work/data_science/projects/deci_deriv_project/deci_deriv")

from deci_deriv import Variable, TapeContext

if __name__ == "__main__":
    tape = TapeContext()

    with tape:
        a = Variable(2.0)
        b = Variable(3.0)
        c = a * b
        d = Variable(4.0)
        e = c + d

        e.backward()

        print("Example 1:")
        print(f"{a=}\n{b=}\n{c=}\n{d=}\n{e=}")
    
    with tape:
        in_1 = Variable(2.0)
        in_2 = Variable(0.0)
        w_1 = Variable(-3.0)
        w_2 = Variable(1.0)
        bias = Variable(6.8)

        linear_1 = in_1 * w_1
        linear_2 = in_2 * w_2
        final_linear = linear_1 + linear_2
        neuron = final_linear + bias
        out = neuron * Variable(2.9)

        out.backward()

        print("Example 2:")
        print(f"{in_1=}\n{in_2=}\n{w_1=}\n{w_2=}\n{bias=}")
        print(f"{final_linear=}\n{neuron=}\n{out=}")