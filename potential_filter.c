#include <Python.h>
#include <math.h>
#include <stdint.h>

/*
static int _filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void * user_data
) {
    double d=buffer[13];
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    for(int i=0;i<filter_size;i++) {
        x = d-buffer[i];
        
    }
    *return_value = sum;
    return 1;
}
*/

// 1.0 / np.sqrt((1-i)*(1-i) + (1-j)*(1-j) + (1-k)*(1-k))
static const double w[27] = {0.57735027, 0.70710678, 0.57735027, 0.70710678, 1.0,
                             0.70710678, 0.57735027, 0.70710678, 0.57735027, 0.70710678,
                             1.0, 0.70710678, 1.0, 1.0, 1.0,
                             0.70710678, 1.0, 0.70710678, 0.57735027, 0.70710678,
                             0.57735027, 0.70710678, 1.0, 0.70710678, 0.57735027,
                             0.70710678, 0.57735027};
// 4 10 12 14 16 22
static const double w_n[27] = {0.0, 0.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 0.0, 0.0,
                               1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0};
// np.sum(w*f(data[13]-d) for d,w in zip(data,W))

// ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
static int potential_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    double fac = δ * δ * 0.5;
    for (int i = 0; i < filter_size; i++)
    {
        x = d - buffer[i];
        if (x < δ)
        {
            sum += w[i] * x * x * 0.5;
        }
        else
        {
            if (x < 0)
            {
                sum += w[i] * (δ * (-x) - fac);
            }
            else
            {
                sum += w[i] * (δ * x - fac);
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
static int potential_dx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    for (int i = 0; i < filter_size; i++)
    {
        x = d - buffer[i];
        if (x < δ)
        {
            sum += w[i] * x;
        }
        else
        {
            if (x < 0)
            {
                sum -= w[i] * δ;
            }
            else
            {
                sum += w[i] * δ;
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
static int potential_dx_t_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    for (int i = 0; i < filter_size; i++)
    {
        x = d - buffer[i];
        if (x < δ)
        {
            sum += w[i];
        }
        else
        {
            if (x < 0)
            {
                sum -= w[i] * δ / x;
            }
            else
            {
                sum += w[i] * δ / x;
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δδψ = lambda x: 1 if x <= δ else 0
static int potential_dxdx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    double sum = 0;
    double δ = ((double *)user_data)[0];
    for (int i = 0; i < filter_size; i++)
    {
        if ((d - buffer[i]) < δ)
        {
            sum += w[i];
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

static int square_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    //double δ = *(double *)user_data;
    //double fac = δ*δ*0.5;
    for (int i = 0; i < filter_size; i++)
    {
        if (w_n[i] == 1)
        {
            x = d - buffer[i];
            sum += x * x;
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

static int square_dx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    //double δ = *(double *)user_data;
    //double fac = δ*δ*0.5;
    for (int i = 0; i < filter_size; i++)
    {
        if (w_n[i] == 1)
        {
            x = d - buffer[i];
            sum += 2 * x;
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

static int square_dxdx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x;
    //double δ = *(double *)user_data;
    //double fac = δ*δ*0.5;
    for (int i = 0; i < filter_size; i++)
    {
        if (w_n[i] == 1)
        {
            x = d - buffer[i];
            sum += 2;
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// ψ = lambda x: (2δ)^-p*(δ*δ*p)^(p-1)*x**2 if x <= δ else 1/p*(np.abs(x-δ(1-p/2)*sgn(x)))**p
static int mod_p_norm_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x, x_b;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    double a = pow(2.0 * δ, -p) * pow(δ * δ * p, p - 1.0);
    double b = δ * (1.0 - 0.5 * p);
    for (int i = 0; i < filter_size; i++)
    {
        x = d - buffer[i];
        if (x < δ)
        {
            sum += w[i] * a * x * x;
        }
        else
        {
            if (x < 0)
            {
                x_b = x + b;
            }
            else
            {
                x_b = x - b;
            }
            if (x_b < 0)
            {
                sum += w[i] * pow(-x_b, p);
            }
            else
            {
                sum += w[i] * pow(x_b, p);
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δψ = lambda x: 2^(1-p)*δ^-p*(δ*δ*p)^(p-1) * x if x <= δ else (0.5*d(p-2)s+x)|0.5*d(p-2)s+x|^p-2
static int mod_p_norm_dx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x, x_b;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    double a = pow(2, 1 - p) * pow(δ, -p) * pow(δ * δ * p, p - 1.0);
    double b = 0.5 * δ * (p - 2);
    for (int i = 0; i < filter_size; i++)
    {
        x = d - buffer[i];
        if (x < δ)
        {
            sum += w[i] * a * x;
        }
        else
        {
            if (x < 0)
            {
                x_b = x - b;
            }
            else
            {
                x_b = x + b;
            }
            if (x_b < 0)
            {
                sum += w[i] * x_b * pow(-x_b, p - 2);
            }
            else
            {
                sum += w[i] * x_b * pow(x_b, p - 2);
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// δψ = lambda x: x if 2^(1-p)*δ^-p*(δ*δ*p)^(p-1) <= δ else (p-2)(0.5*δ(p-2)*s+x)^2*abs(0.5*δ*(p-2)*s+x)^(p-4)+abs(0.5*d*(p-2)*s+x)^(p-2)
// (p-2)(0.5*δ(p-2)*s+x)^2*abs(0.5*δ*(p-2)*s+x)^(p-4)+abs(0.5*δ*(p-2)*s+x)^(p-2)
// (p-2)(    b     *s+x)^2*abs(    b      *s+x)^(p-4)+abs(    b      *s+x)^(p-2)
// (p-2)(    x_b       )^2*abs(    x_b        )^(p-4)+abs(    x_b        )^(p-2)
static int mod_p_norm_dxdx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    //*return_value = 0;
    double sum = 0;
    double x, x_b;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    double a = pow(2, 1 - p) * pow(δ, -p) * pow(δ * δ * p, p - 1.0);
    double b = 0.5 * δ * (p - 2);
    for (int i = 0; i < filter_size; i++)
    {
        x = d - buffer[i];
        if (x < δ)
        {
            sum += w[i] * a;
        }
        else
        {
            if (x < 0)
            {
                x_b = x - b;
            }
            else
            {
                x_b = x + b;
            }
            if (x_b < 0)
            {
                sum += w[i] * ((p - 2) * pow(x_b, 2) * pow(-x_b, p - 4) + pow(-x_b, p - 2));
            }
            else
            {
                sum += w[i] * ((p - 2) * pow(x_b, 2) * pow(x_b, p - 4) + pow(x_b, p - 2));
            }
        }
    }
    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
}

// lambda t: δ^2(|t/δ| - log(1+|t/δ|) )
static int edge_preserving_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    for (int i = 0; i < filter_size; i++)
    {
        if (w_n[i] == 1)
        {
            x = (d - buffer[i]) / δ;
            if (x > 0)
            {
                sum = sum + δ * δ * (x - log(1 + x));
            }
            else
            {
                sum = sum + δ * δ * (-x - log(1 - x));
            }
        }
    }
    *return_value = sum;
    return 1;
}
// lambda t: δ^2(|t/δ| - log(1+|t/δ|) )
// lambda t: t / (|t/δ|+1)
static int edge_preserving_dx_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    for (int i = 0; i < filter_size; i++)
    {
        if (w_n[i] == 1)
        {
            x = (d - buffer[i]);
            if (x > 0)
            {
                sum = sum + x / (x / δ + 1);
            }
            else
            {
                sum = sum + x / (-x / δ + 1);
            }
        }
    }
    *return_value = sum;
    return 1;
}
// lambda t: δ^2(|t/δ| - log(1+|t/δ|) )
// lambda t: t / (|t/δ|+1)
static int edge_preserving_dx_t_filter(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data)
{
    double d = buffer[13];
    double sum = 0;
    double x;
    double δ = ((double *)user_data)[0];
    double p = ((double *)user_data)[1];
    for (int i = 0; i < filter_size; i++)
    {
        if (w_n[i] == 1)
        {
            x = (d - buffer[i]);
            if (x > 0)
            {
                sum = sum + 1 / (x / δ + 1);
            }
            else
            {
                sum = sum + 1 / (-x / δ + 1);
            }
        }
    }
    *return_value = sum;
    return 1;
}

static char *filter_signature = "int (double *, intptr_t, double *, void *)";

static PyObject *
py_get_potential_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(potential_filter, filter_signature, NULL);
}
static PyObject *
py_get_potential_dx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(potential_dx_filter, filter_signature, NULL);
}
static PyObject *
py_get_potential_dx_t_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(potential_dx_t_filter, filter_signature, NULL);
}
static PyObject *
py_get_potential_dxdx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(potential_dxdx_filter, filter_signature, NULL);
}
static PyObject *
py_get_square_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(square_filter, filter_signature, NULL);
}
static PyObject *
py_get_square_dx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(square_dx_filter, filter_signature, NULL);
}
static PyObject *
py_get_square_dxdx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(square_dxdx_filter, filter_signature, NULL);
}
static PyObject *
py_get_mod_p_norm_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(mod_p_norm_filter, filter_signature, NULL);
}
static PyObject *
py_get_mod_p_norm_dx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(mod_p_norm_dx_filter, filter_signature, NULL);
}
static PyObject *
py_get_mod_p_norm_dxdx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(mod_p_norm_dxdx_filter, filter_signature, NULL);
}
static PyObject *
py_get_edge_preserving_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(edge_preserving_filter, filter_signature, NULL);
}
static PyObject *
py_get_edge_preserving_dx_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(edge_preserving_dx_filter, filter_signature, NULL);
}
static PyObject *
py_get_edge_preserving_dx_t_filter(PyObject *obj, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    return PyCapsule_New(edge_preserving_dx_t_filter, filter_signature, NULL);
}

static PyMethodDef ExampleMethods[] = {
    {"potential_filter", (PyCFunction)py_get_potential_filter, METH_VARARGS, ""},
    {"potential_dx_filter", (PyCFunction)py_get_potential_dx_filter, METH_VARARGS, ""},
    {"potential_dxdx_filter", (PyCFunction)py_get_potential_dxdx_filter, METH_VARARGS, ""},
    {"potential_dx_t_filter", (PyCFunction)py_get_potential_dx_t_filter, METH_VARARGS, ""},
    {"square_filter", (PyCFunction)py_get_square_filter, METH_VARARGS, ""},
    {"square_dx_filter", (PyCFunction)py_get_square_dx_filter, METH_VARARGS, ""},
    {"square_dxdx_filter", (PyCFunction)py_get_square_dxdx_filter, METH_VARARGS, ""},
    {"mod_p_norm_filter", (PyCFunction)py_get_mod_p_norm_filter, METH_VARARGS, ""},
    {"mod_p_norm_dx_filter", (PyCFunction)py_get_mod_p_norm_dx_filter, METH_VARARGS, ""},
    {"mod_p_norm_dxdx_filter", (PyCFunction)py_get_mod_p_norm_dxdx_filter, METH_VARARGS, ""},
    {"edge_preserving_filter", (PyCFunction)py_get_edge_preserving_filter, METH_VARARGS, ""},
    {"edge_preserving_dx_filter", (PyCFunction)py_get_edge_preserving_dx_filter, METH_VARARGS, ""},
    {"edge_preserving_dx_t_filter", (PyCFunction)py_get_edge_preserving_dx_t_filter, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

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
    NULL};

PyMODINIT_FUNC
PyInit_PotentialFilter(void)
{
    return PyModule_Create(&example);
}