import sys
sys.path.append(r"/mnt/c/Users/rapha/My Drive (raphael.pb.rialland@gmail.com)/Work/data_science/projects/pico_prop_project")
sys.path.append(r"/mnt/c/Users/rapha/My Drive (raphael.pb.rialland@gmail.com)/Work/data_science/projects/pico_prop_project/pico_prop")

from pico_prop import Variable, TapeContext

if __name__ == "__main__":
    with TapeContext():
        # Input variables
        a = Variable(2.0)
        b = Variable(-3.0)
        c = Variable(10.0)
        x = Variable(-1.0)
        y = Variable(3.0)

        # Computation
        d1 = a * x
        d2 = b * y
        d3 = d1 + d2

        out = d3 + c

        # Perform the backward pass
        out.backward()

        # Print results (data and gradients)
        print("--- Input Variables ---")
        print(f"{a=}")
        print(f"{b=}")
        print(f"{c=}")
        print(f"{x=}")
        print(f"{y=}")
        print("--- Intermediates ---")
        print(f"{d1=}")
        print(f"{d2=}")
        print(f"{d3=}")
        print("--- Final Output")
        print(f"{out=}")