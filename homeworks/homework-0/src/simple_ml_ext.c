#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int m = 50;
int n = 5;
int k = 3;
int batch = 50;
float lr = 1.0;

float theta[15] = {
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
};
unsigned char y[50] = {
    1, 0, 2, 0, 0, 1, 2, 1, 1, 1, 0, 0, 2, 0, 0, 2, 1, 2, 2, 0, 2, 2,
    0, 2, 1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 0, 1, 2, 0, 1, 2,
                    1, 2, 1, 2, 1, 2
};
float X[250] = {
    1.7640524 ,  0.4001572 ,  0.978738  ,  2.2408931 ,  1.867558  ,
    -0.9772779 ,  0.95008844, -0.1513572 , -0.10321885,  0.41059852,
    0.14404356,  1.4542735 ,  0.7610377 ,  0.12167501,  0.44386324,
    0.33367434,  1.4940791 , -0.20515826,  0.3130677 , -0.85409576,
    -2.5529897 ,  0.6536186 ,  0.8644362 , -0.742165  ,  2.2697546 ,
    -1.4543657 ,  0.04575852, -0.18718386,  1.5327792 ,  1.4693588 ,
    0.15494743,  0.37816253, -0.88778573, -1.9807965 , -0.34791216,
    0.15634897,  1.2302907 ,  1.2023798 , -0.3873268 , -0.30230275,
    -1.048553  , -1.420018  , -1.7062702 ,  1.9507754 , -0.5096522 ,
    -0.4380743 , -1.2527953 ,  0.7774904 , -1.6138978 , -0.21274029,
    -0.89546657,  0.3869025 , -0.51080513, -1.1806322 , -0.02818223,
    0.42833188,  0.06651722,  0.3024719 , -0.6343221 , -0.36274117,
    -0.67246044, -0.35955316, -0.8131463 , -1.7262826 ,  0.17742614,
    -0.40178093, -1.6301984 ,  0.46278226, -0.9072984 ,  0.0519454 ,
    0.7290906 ,  0.12898292,  1.1394007 , -1.2348258 ,  0.40234163,
    -0.6848101 , -0.87079716, -0.5788497 , -0.31155252,  0.05616534,
    -1.1651498 ,  0.9008265 ,  0.46566245, -1.5362437 ,  1.4882522 ,
    1.8958892 ,  1.1787796 , -0.17992483, -1.0707526 ,  1.0544517 ,
    -0.40317693,  1.222445  ,  0.20827498,  0.97663903,  0.3563664 ,
    0.7065732 ,  0.01050002,  1.7858706 ,  0.12691209,  0.40198937,
    1.8831507 , -1.347759  , -1.270485  ,  0.9693967 , -1.1731234 ,
    1.9436212 , -0.41361898, -0.7474548 ,  1.922942  ,  1.4805148 ,
    1.867559  ,  0.90604466, -0.86122566,  1.9100649 , -0.26800337,
    0.8024564 ,  0.947252  , -0.15501009,  0.61407936,  0.9222067 ,
    0.37642553, -1.0994008 ,  0.2982382 ,  1.3263859 , -0.69456786,
    -0.14963454, -0.43515354,  1.8492638 ,  0.67229474,  0.40746182,
    -0.76991606,  0.5392492 , -0.6743327 ,  0.03183056, -0.6358461 ,
    0.67643327,  0.57659084, -0.20829876,  0.3960067 , -1.0930616 ,
    -1.4912575 ,  0.4393917 ,  0.1666735 ,  0.63503146,  2.3831449 ,
    0.94447947, -0.91282225,  1.1170163 , -1.3159074 , -0.4615846 ,
    -0.0682416 ,  1.7133427 , -0.74475485, -0.82643855, -0.09845252,
    -0.6634783 ,  1.1266359 , -1.0799315 , -1.1474687 , -0.43782005,
    -0.49803245,  1.929532  ,  0.9494208 ,  0.08755124, -1.2254355 ,
    0.844363  , -1.0002153 , -1.5447711 ,  1.1880298 ,  0.3169426 ,
    0.9208588 ,  0.31872764,  0.8568306 , -0.6510256 , -1.0342429 ,
    0.6815945 , -0.80340964, -0.6895498 , -0.4555325 ,  0.01747916,
    -0.35399392, -1.3749512 , -0.6436184 , -2.2234032 ,  0.62523144,
    -1.6020577 , -1.1043833 ,  0.05216508, -0.739563  ,  1.5430146 ,
    -1.2928569 ,  0.26705086, -0.03928282, -1.1680934 ,  0.5232767 ,
    -0.17154633,  0.77179056,  0.82350415,  2.163236  ,  1.336528  ,
    -0.36918184, -0.23937918,  1.0996596 ,  0.6552637 ,  0.64013153,
    -1.616956  , -0.02432613, -0.7380309 ,  0.2799246 , -0.09815039,
    0.9101789 ,  0.3172182 ,  0.78632796, -0.4664191 , -0.94444627,
    -0.4100497 , -0.01702041,  0.37915173,  2.259309  , -0.04225715,
    -0.955945  , -0.34598178, -0.463596  ,  0.48148146, -1.540797  ,
    0.06326199,  0.15650654,  0.23218104, -0.5973161 , -0.23792173,
    -1.424061  , -0.49331987, -0.54286146,  0.41605005, -1.1561824 ,
    0.7811981 ,  1.4944845 , -2.069985  ,  0.42625874,  0.676908  ,
    -0.63743705, -0.3972718 , -0.13288058, -0.29779088, -0.30901298,
    -1.6760038 ,  1.1523316 ,  1.0796186 , -0.81336427, -1.4664243 
};


void print_matrix(float *X, int m, int n, char *description)
{
    printf("%s\n", description);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f, ", X[i * n + j]);
        }
    }
    printf("\nDone...........\n");
}


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
        print_matrix(H, batch, k, "Linear:");
        /* Compute final probabilities */
        Z = softmax(H, batch, k);
        print_matrix(Z, batch, k, "Prob:");

        for (int i = 0; i < batch; i++)
        {
            Z[i * k + (int)y_batch[i]] -= 1.0;
        }
        print_matrix(Z, batch, k, "Z - I:");

        /* backward pass */
        X_T = transpose(X_batch, batch, n);
        grad = matmul(X_T, Z, n, batch, k);
        print_matrix(grad, n, k, "Grad:");

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                theta[i * k + j] -= ((lr * grad[i * k + j]) / batch);
            }
        }
    }
}


int main(int argc, char *argv[])
{
    for (int i = 0; i < m; i++)
    {
        printf("%d, ", y[i]);
    }
    printf("\n");
    softmax_regression_epoch_cpp(X, y, theta, m, n, k, lr, batch);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            printf("%f\n", theta[i * k + j]);
        }
    }
    return 0;
}
