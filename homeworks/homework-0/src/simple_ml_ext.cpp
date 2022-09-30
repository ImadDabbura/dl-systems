#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float *matmul(float *X1, float *X2, int m, int n1, int n2)
{
    float *out = (float *)malloc(sizeof(float) * m * n2);
    float tmp;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            tmp = 0.0;
            for (int k = 0; k < n1; k++)
            {
                tmp += X1[i * n1 + k] * X2[k * n2 + j];
            }
            out[i * n2 + j] = tmp;
        }
    }
    return out;
}

/* Transpose matrix X of size m x n to become n x m */
float *transpose(float *X, int m, int n)
{
    float *X_T = (float *)malloc(sizeof(float) * m * n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            X_T[j * m + i] = X[i * n + j];
        }
    }
    return X_T;
}


/* Compute the softmax probabilities  */
float *softmax(float *X, int m, int n)
{
    float *Z = (float *)malloc(sizeof(float) * m * n);
    float *sum_exp = (float *)malloc(sizeof(float) * m);
    for (int i = 0; i < m; i++)
    {
        sum_exp[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            Z[i * n + j] = exp(X[i * n + j]);
            sum_exp[i] += Z[i * n + j];
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Z[i * n + j] /= sum_exp[i];
        }
    }
    return Z;
}


float *get_X_batch(float *X, int start_index, int batch_size, int n)
{
    float *X_batch = (float *)malloc(sizeof(float) * batch_size * n);
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            X_batch[i * n + j] = X[(i + start_index) * n + j];
        }
    }
    return X_batch;
}


unsigned char *get_y_batch(unsigned char *y, int start_index, int batch_size)
{
    unsigned char *y_batch = (unsigned char *)malloc(sizeof(unsigned char) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        y_batch[i] = y[i + start_index];
    }
    return y_batch;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    float *X_T;
    float *H;     /* linear transformation */
    float *Z;     /* final probabilities */
    float *grad;
    int n_batches = m / batch;
    int start_index;
    float *X_batch;
    unsigned char *y_batch;


    /* forward pass */
    for (int iteration = 0; iteration < n_batches; iteration++)
    {
        /* Forward pass */
        start_index = iteration * batch;
        printf("Start index = %d, iteration = %d\n", start_index, iteration);

        X_batch = get_X_batch(X, start_index, batch, n);
        y_batch = get_y_batch(y, start_index, batch);

        H = matmul(X_batch, theta, batch, n, k);
        /* Compute final probabilities */
        Z = softmax(H, batch, k);

        for (int i = 0; i < batch; i++)
        {
            Z[i * k + (int)y_batch[i]] -= 1.0;
        }

        /* backward pass */
        X_T = transpose(X_batch, batch, n);
        grad = matmul(X_T, Z, n, batch, k);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                theta[i * k + j] -= ((lr * grad[i * k + j]) / batch);
            }
        }
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
