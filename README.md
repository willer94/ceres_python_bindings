# Ceres Python Wrapper

This project uses pybind11 to wrap Ceres with a python interface.

## Build Setup
There are two different ways to build this library. The easiest and recommended
way is to build it along with the Ceres library. The other way builds it 
standalone, and is more error prone.

### Recommended: Build Alongside Ceres

Clone the repository at https://github.com/Edwinem/ceres_python_bindings into 
your *ceres-solver* folder.

Initialize and download the pybind11 submodule
```shell
git clone https://github.com/Edwinem/ceres_python_bindings
cd ceres_python_bindings
git submodule init
git submodule update
```

If you cloned it somewhere else then you must now copy and paste the
 *ceres_python_bindings* directory to your *ceres-solver* directory.
 
Your ceres directory should now look something like this.
  ```
  ceres-solver/
  │
  ├── CMakeLists.txt
  ├── include
  ├── ...
  │
  ├── ceres_python_bindings/ - THIS REPOSITORY
  │   ├── pybind11 
  │   ├── python_bindings
  │   ├── ...
  │   └── AddToCeres.cmake - file to include in Ceres CMakeLists.txt
```

Open up your *ceres-solver/CMakeLists.txt* and add the following to the end
of the file.

```
include(ceres_python_bindings/AddToCeres.cmake)
```

If everything was successful then when call *cmake* in your build folder at the
end it should output 

```
-- Python Bindings for Ceres(PyCeres) have been added
```

Build Ceres as you would normally. To specifically build the bindings you should
 call _make PyCeres_ .

**CONDA PYTHON ENV**， cmake with python executable and lib
```
cmake -DPYTHON_EXECUTABLE=/home/ubuntu/anaconda3/envs/open-mmlab/bin/python  -DPYTHON_LIBRARY=/home/ubuntu/anaconda3/envs/open-mmlab/lib/ ..
cp lib/PyCeres*.so /home/ubuntu/anaconda3/envs/open-mmlab/lib/python3.7/site-packages
```


### Build seperately and link to Ceres

The CMakeLists.txt for this option still needs more work. For now you have to 
modify it yourself and set the location of *libceres.a*.

Clone as you would normally.

Run the standard cmake procedure.

```shell
cd ceres_python_bindings
mkdir build
cd build
cmake ..
make
```

Note this way has a couple of problems.

* The *CMakelists.txt* must define some of Ceres's compile time flags such as the threading model, and
it should match ,however, *libceres.a* was built. Else errors may occur.
* You have to take care of linking the other libraries such as *suitesparse* and *glog*.

## How to 

Assuming you built the library correctly you should now have a file called **PyCeres.so**. 
It probably more likely looks something like this **PyCeres.cpython-36m-x86_64-linux-gnu.so**.
Mark down the location of this file. This location is what you have to add to
python **sys.path** in order to use the library. An example of how to do this can
be seen below.

```python
pyceres_location="..."
import sys
sys.path.insert(0, pyceres_location)
```

After this you can now run 
```python
import PyCeres
```

to utilize the library.

You should peruse some of the examples listed below. It works almost exactly
like Ceres in C++. The only care you have to take is that the parameters you 
pass to the AddResidualBlock() function is a numpy array.

### Basic HelloWorld

Code for this example can be found in *python_tests/ceres_hello_world_example.py*

This example is the same as the hello world example from Ceres. 

```python
pyceres_location="../../build/lib" # assuming library was built along with ceres
# and cmake directory is called build
import sys
sys.path.insert(0, pyceres_location)

import PyCeres # Import the Python Bindings
import numpy as np

# The variable to solve for with its initial value.
initial_x=5.0
x=np.array([initial_x]) # Requires the variable to be in a numpy array

# Here we create the problem as in normal Ceres
problem=PyCeres.Problem()

# Creates the CostFunction. This example uses a C++ wrapped function which 
# returns the Autodiffed cost function used in the C++ example
cost_function=PyCeres.CreateHelloWorldCostFunction()

# Add the costfunction and the parameter numpy array to the problem
problem.AddResidualBlock(cost_function,None,x) 

# Setup the solver options as in normal ceres
options=PyCeres.SolverOptions()
# Ceres enums live in PyCeres and require the enum Type
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout=True
summary=PyCeres.Summary()
# Solve as you would normally
PyCeres.Solve(options,problem,summary)
print(summary.BriefReport() + " \n")
print( "x : " + str(initial_x) + " -> " + str(x) + "\n")
```

### CostFunction in Python

This library allows you to create your own custom CostFunction in Python to be
used with the Ceres Solver.

An custom CostFunction in Python can be seen here.
```python
# function f(x) = 10 - x.
# Comes from ceres/examples/helloworld_analytic_diff.cc
class QuadraticCostFunction(PyCeres.CostFunction):
    def __init__(self):
        # MUST BE CALLED. Initializes the Ceres::CostFunction class
        super().__init__()
        
        # MUST BE CALLED. Sets the size of the residuals and parameters
        self.set_num_residuals(1) 
        self.set_parameter_block_sizes([1])

    # The CostFunction::Evaluate(...) virtual function implementation
    def Evaluate(self,parameters, residuals, jacobians):
        x=parameters[0][0]

        residuals[0] = 10 - x

        if (jacobians!=None): # check for Null
            jacobians[0][0] = -1

        return True
```

Some things to be aware of for a custom CostFunction

* residuals is a numpy array
* parameters,jacobians are lists of numpy arrays ([arr1,arr2,...])
    * Indexing works similar to Ceres C++. parameters[i] is the ith parameter block
    * You must always index into the list first. Even if it only has 1 value. 
* You must call the base constructor with super.

### CostFunction defined in C++
It is possible to define your custom CostFunction in C++ and utilize it within 
the python framework. In order to do this we provide a file **python_bindings/custom_cpp_cost_functions.cpp**.
which provides a place to write your own wrapper code. The easiest way to do this
is create an initialization function that creates your custom CostFunction class
and returns a ceres::CostFunction* to it. That function should then be wrapped in
the *void add_custom_cost_functions(py::module& m)* function.

It should end up looking something like this.
```cpp

#include <CUSTOM_HEADER_FILES_WITH_COST_FUNCTION>

// Create a function that initiliazes your CostFunction and returns a ceres::CostFunction*

ceres::CostFunction* CreateCustomCostFunction(arg1,arg2,...){
    return new CustomCostFunction(arg1,arg2,...);
}

// In file custom_cpp_cost_function add the following line

void add_custom_cost_functions(py::module &m) {
    // ....
    m.def("CreateCustomCostFunction",&CreateCustomCostFunction);
}

```

We provide a basic example of this in *custom_cpp_cost_functions.cpp*. 
Note you are responsible for ensuring that all the dependencies and includes are
set correctly for your library.

### Running examples
We provide a couple examples of how to use the library under *./python_tests*.
They all assume the wrappers were built alongside Ceres for the PyCeres library.
If you did not do this then you need to set the *PYCERES_LOCATION* environment
variable. 

You need the following python libs to run these examples.

**Required:**
 
* numpy

**Optional:**

* pytest
* jax
 

## Experimental PyTorch functionality
**WARNING THIS IS CURRENTLY EXTREMELY EXPERIMENTAL**

In order to bypass the fundamental slowness of Python (due to GIL and other factors). This
library optionally provides the capability to utilize PyTorch's TorchScript.
This allows you to define a CostFunction in Python, but bypass having to touch
it when solving the Ceres::Problem.

Right now the only the standalone version of this bindings support it. Lots of 
the paths are hardcoded. So you will have to change them to.

To enable this functionality you must do the following things.

- Enable the option in cmake by turning on _WITH_PYTORCH_ 
- Build Ceres and GLOG that you link to with _-D_GLIBCXX_USE_CXX11_ABI=0_
    - The default PyTorch libs that you download from pip and other package managers
    is built with the old C++ ABI.
    
Note this will break normal functionality as all Python instantiations now requires
a *import Torch* before you import *PyCeres*.

Currently the TorchScript is passed by serialized files. 


## Warnings:

* Remember Ceres was designed with a C++ memory model. So you have to be careful
when using it from Python. The main problem is that Python does not really have
the concept of giving away ownership of memory. So it may try to delete something
that Ceres still believes is valid.
    * I think for most stuff I setup the proper procedures that this doesn't happen (
    e.g Ceres::Problem by default has Ownership turned off, cost_function can't be deleted
    until Problem is ,...) . But you never know what I missed.
* Careful with wrapping AutodiffCostfunction. It takes over the memory of a cost
functor which can cause errors.
* Python has **GIL**. Therefore cost functions written in Python have a fundamental
slowdown, and can't be truly multithreaded.


## TODOs
- [ ] The wrapper code that wraps the Evaluate pointers(residuals,parameters,..)
    needs a lot of improvement and optimization. We really need this to be a zero copy
operation.
- [ ] Wrap all the variables for Summary and other classes
- [ ] LocalParameterizations and Lossfunctions need to be properly wrapped
- [ ] Callbacks need to be wrapped
- [ ] Investigate how to wrap a basic python function for evaluate rather than 
go through the CostFunction( something like in the C api).
- [ ] Add docstrings for all the wrapped stuff
- [X] Add a place for users to define their CostFunctions in C++
- [ ] Evaluate speed of Python Cost Function vs C++
- [ ] Clean up AddResidualBlock() and set up the correct error checks
- [ ] Figure out how google log should work with Python
- [ ] Figure out if Jax or PyTorch could somehow be integrated so that we use
their tensor/numpy arrays.

## Status
Custom Cost functions work

## LICENSE
Same as Ceres New BSD.

## ADD
Add PnPCostFunction
Add more loss function wrapper
### TODOs
- [] wrapper for ComposedLoss and ScaledLoss

## Credit
This is just a wrapper over the hard work of the main 
[Ceres](http://ceres-solver.org/) project. All the examples derive from ceres-solver/examples
