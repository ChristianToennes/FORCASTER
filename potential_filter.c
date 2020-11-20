#include <Python.h>
#include <math.h>
#include <stdint.h>

// 1.0 / np.sqrt((1-i)*(1-i) + (1-j)*(1-j) + (1-k)*(1-k))
static const double w[27] = {0.57735027, 0.70710678, 0.57735027, 0.70710678, 1.0       ,
                     0.70710678, 0.57735027, 0.70710678, 0.57735027, 0.70710678,
                     1.0       , 0.70710678, 1.0       , 1.0       , 1.0       ,
                     0.70710678, 1.0       , 0.70710678, 0.57735027, 0.70710678,
                     0.57735027, 0.70710678, 1.0       , 0.70710678, 0.57735027,
                     0.70710678, 0.57735027};
// np.sum(w*f(data[13]-d) for d,w in zip(data,W))

// ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
static int potential_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void * user_data
) {
    double d=buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    double δ = *(double *)user_data;
    double fac = δ*δ*0.5;
    for(int i=0;i<filter_size;i++) {
        x = d-buffer[i];
        if(x<δ) {
            sum += w[i]*x*x*0.5;
        } else {
            if(x<0) {
                sum += w[i]*(δ*(-x)-fac);
            }
            else {
                sum += w[i]*(δ*x-fac);
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
static int potential_dx_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void * user_data
) {
    double d=buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    double δ = *(double *)user_data;
    for(int i=0;i<filter_size;i++) {
        x = d-buffer[i];
        if(x<δ) {
            sum += w[i]*x;
        } else {
            if(x<0) {
                sum -= w[i]*δ;
            }
            else {
                sum += w[i]*δ;
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
static int potential_dx_t_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void * user_data
) {
    double d=buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    double δ = *(double *)user_data;
    for(int i=0;i<filter_size;i++) {
        x = d-buffer[i];
        if(x<δ) {
            sum += w[i];
        } else {
            if(x<0) {
                sum -= w[i]*δ/x;
            }
            else {
                sum += w[i]*δ/x;
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δδψ = lambda x: 1 if x <= δ else 0
static int potential_dxdx_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void * user_data
) {
    double d=buffer[13];
    double sum = 0;
    double δ = *(double *)user_data;
    for(int i=0;i<filter_size;i++) {
        if((d-buffer[i])<δ) {
            sum += w[i];
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}



static char *filter_signature = "int (double *, intptr_t, double *, void *)";

static PyObject *
py_get_potential_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) return NULL;
    return PyCapsule_New(potential_filter, filter_signature, NULL);
}
static PyObject *
py_get_potential_dx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) return NULL;
    return PyCapsule_New(potential_dx_filter, filter_signature, NULL);
}
static PyObject *
py_get_potential_dx_t_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) return NULL;
    return PyCapsule_New(potential_dx_t_filter, filter_signature, NULL);
}
static PyObject *
py_get_potential_dxdx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) return NULL;
    return PyCapsule_New(potential_dxdx_filter, filter_signature, NULL);
}

static PyMethodDef ExampleMethods[] = {
    {"potential_filter", (PyCFunction)py_get_potential_filter, METH_VARARGS, ""},
    {"potential_dx_filter", (PyCFunction)py_get_potential_dx_filter, METH_VARARGS, ""},
    {"potential_dxdx_filter", (PyCFunction)py_get_potential_dxdx_filter, METH_VARARGS, ""},
    {"potential_dx_t_filter", (PyCFunction)py_get_potential_dx_t_filter, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
static struct PyModuleDef example = {
    PyModuleDef_HEAD_INIT,
    "PotentialFilter",
    NULL,
    -1,
    ExampleMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_PotentialFilter(void)
{
    return PyModule_Create(&example);
}