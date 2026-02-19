- Automatic differentiation
  - Forward mode AD: We start from input node to output node where at each step
    we take the partial derivative of that node with respect to the input node
    - For each input, we do full AD forward pass. I.e. If we have `n` inputs,
      we need to do `n` AD forward passes
    - Since most loss functions are scalars and input size `n` is very large -> inefficient
  - Backward mode AD: We start from output node to input node where at each
    step we take the partial derivative of output node with respect to input node.
    In other words, we compute the gradient of the scalar function with respect
    to all input nodes in a single backward pass. This will create a
    computational graph for the gradients that would also let us compute the
    gradients of the gradients if we add more operations to it.
    - The gradients at each node tells us how much the node needs to change to
      change output in the steepest ascent/descent direction (locally)
    - Pros: we can do more optimizations to utilize the underlying HW and fuse
      node/ops together. It is also very efficient since most of the functions in
      DL have 1 or very few outputs.
    - Cons: We need to store the intermediate results for each step. This means
      we need to store the input tensors and the operation that created the
      tensor so we know how to compute its gradients during the backward
      pass -> More memory usage
- Broadcasting doesn't copy data but only changes strides. With automatic
  differentiation, we basically sum across all the dimensions that we broadcasted
- Reshape/view can in sometimes copy data if we can't use striding to change
  the shape of the tensor
- Initialization
  - The effect of weights initialization persists throughout training
    - It affects the relative norms of the activations and the gradients over time
    - If we don't initialize the weights appropriately (such as $2/n$ for
      ReLU), the L2-norm of the activations/gradients will change over time
      during training, which will lead to either no training or exploding
  - Weights don't change much from their initial values after training
    - It may change in some direction/dimension more but overall the weights
      don't change much relative to the initial values especially for deep NN.
      We can see this if we plot the variance of the weights before and after
      training for each layer. We typically see very similar variances for the weights
      before/after training
  - Proper weight initializations speed up training and converge to much lower
    error rates
  - We can judge the effect of initialization by computing the norm of both
    the weights and the gradients at every layer over all iterations
- Normalization. The idea is that if we know that the norms of the
  activations/gradients would change over time during training due to poor
  initializations, we can add normalization layer that would fix this issue by
  always normalizing the activations of any layer to have zero mean and
  variance 1
  - Batch Normalization:
    - Helps trains much faster by normalizing across all examples and maintain
      different relative norms between activations within layers. This would
      overcome the issue of losing the discriminative nature of different sizes
      of activations since we normalize across batches for each feature
      separately
    - Drawbacks:
      - Dependency between samples in the same batch as each one is dependent
        on other examples in the same batch.
      - Unstable training with small batches where we may have infinite variance
      - Doesn't work with RNNs as hidden state depends on all previous hidden
        states/input in the sequence (temporal dependency) and we can't just
        compute stats for each time step independently
  - Layer Normalization:
    - It fixes the problem of varying norms of layer activations as well as
      solve the problem where we have only one sample in
      the batch that batch normalization have. It normalize across all features
      for each sample
    - It also fixes the problem of RNNs with BatchNorm as it applies the
      normalization of features (embedding dimension) for each token in each
      batch -> widely used in RNNs and Transformer architectures
    - The main drawback is that is it is hard to drive the network to low loss especially
      for fully connected networks because after applying the layer
      norm the difference between activations for different examples will be
      lost because it forces the mean to be zero and the standard deviation to
      be 1, but relative norms are very important feature to discriminate
      between classes
- Regularization.
  - Implicit regularization refers to optimizing for all NN that are only
    considered by the learning algorithms such as SGD with a given weight
    initialization (remember that weights change very little from their
    initial values)
  - Explicit regularization refers to changing the NN and training procedure
    to limit the functions learned such as L2-regularization and Dropout
    - The premise is that the magnitude of the parameters is a good proxy for
      complexity. Smoother functions don't change dramatically for small changes
      and thus they have smaller weights. As a result, we can limit the magnitude
      of the parameters to be small by adding a penalty to the loss function such
      as L2 regularization so the weights are smaller.
    - Dropout. It is useful to think of dropout as a stochastic approximation of
      the activations for each layer. It is very similar for what stochastic
      gradient descent does to approximate gradients. We need to divide activations
      and gradients by $1/(1 - p)$ to keep the expected activations/gradients the
      same since each node has probability $1 - p$ to stay active
    - L2 regularization is different than weight decay for optimizer like Adam
      even through they are equivalent for vanilla SGD. The reason is that the
      update includes computing the first and second moments of the gradients,
      which means adding the weight decay directly to the gradient leads to less
      effective results compared to adding the weight decay only when performing
      the update without changing the gradients
- CNN:
  - Parameter sharing: We don't care where is the object in the image. The
    filter should always be able to detect it w/o having different filters for
    different locations in the image
  - Sparse connectivity: Each receptive field has local spatial computation ->
    very few parameters needed
  - translation equivariant: Local transformation leads to same transformation
    after convolution
  - Captures local changes
  - Dilation: To increase receptive field of each feature by having access to
    bigger spatial area (spread out according to the dilation factor)
  - We can think of convolution in terms of matrix multiplication. The weight
    matrix just have to be constructed in a way that reflects the convolution
    operation filled with the actual weights and zeros
    - Note that we don't actually construct this matrix as it would be huge
      but we use that conceptually
  - Multiplying by the transpose of a convolution is equivalent to convolving
    with a flipped version of the filter. This means the derivative of a
    convolution is to flip the filter and convolve with that
- Sequence modeling helps when there is temporal dependence between x/y pairs.
  We typically would assume that x/y pair are i.i.d for tasks such as image
  classification. With tasks such as part of speech recognition or
  autoregressiove prediction (such as time series forecasting and language
  modeling), temporal dependence is key to predict next token or label because
  the token/input can't be understood alone without its context -> can't
  predict the label correctly without encoding the previous inputs.
  - RNNs: the whole past compacted in the last hidden state that makes
    learning harder due to the long history which may lead to exploding/vanishing gradients.
    Also, the far past such as x1 will have less weight if at all compared to
    xt even with LSTM.
    - Maintains a hidden state over time where the hidden state at time $t$
      is a function of the input at time $t$ and hidden state at time $t -1$.
      Therefore, in theory, the hidden state at the last time step should capture
      all the inputs from $t_0$ to $t_T$
    - Weights are shared across all time steps. When unrolling RNN, we can
      think of it as each time step is its own layer but the weights are shared
    - Gradients are calculated use **Backpropagation Through Time** because
      weights are shared across all time steps. This leads to gradient
      vanishing/exploding. If eigenvalue of weight is less than 1 -> vanishing.
      If it is greater than 1 -> explode
  - LSTM
    - Solve the problem of vanishing gradients by breaking down
      the overall hidden state into two states: hidden state and cell state
    - We have four gates (weights) that determine at each time step:
      - What information from the past to keep and what to forget using
        forget/input gates and previous cell state to update cell state
      - What information to pass to next hidden state using output gate and
        cell state
    - Both LSTM/RNN have issues remembering things further in the past in their
      last hidden state. The influence of recent tokens is much more than
      tokens further in the past as they have more of direct connection with the
      current hidden state
  - Direct prediction: We use all the past inputs to predict the next output;
    no hidden state. This is a drawback because we don't have compact representation
    (latent state) of the past.
  - Temporal CNN: It works well for tasks such as speech generation but
    suffers from the relatively small receptive field. This means hidden
    state at any time step t can only capture few of the past history and
    this is not good. We can increase filter size or use dilation but still
    is not optimal.
  - Transformer: We can use attention that take all the history and weight them
    (attention matrix) so we are sure we're incorporating all the past
    - Global receptive field -> q, k, v have access to all inputs over all
      time $t$
      - For autoregressiove modeling, add mask to restrict accesss to inputs
        to only current and previous inputs
    - Order invariant. Permutation to any weight matrix K/Q/V would lead to
      change the output in the same way. Add positional encoding to capture
      order of input
- DL frameworks:
  - Caffe (Only C++):
    - Define a computation in terms of `layers`
    - Each layer implements forward and backward methods
    - The change would be inplace
    - The backward pass implementation is natural and intuitive from Hinton's
      paper
  - Tensorflow:
    - Construct static graph before we can execute it (**define then run**)
    - Make it easy for optimizations such as fusing operations together,
      reusing memory, execute only what is needed at run-time
    - Hard to debug and experiment with the output of each step
    - It is not intuitive and has its own programming language
  - Pytorch:
    - Construct dynamic computation graph (**define by run**)
    - Very easy to debug and experiment with computation graph
    - Can mix python control flow and computations
    - Typically slower. New advancement allowed for JIT compilation to speed
      up the execution.
- Hardware acceleration:
  - We need to pay attention to memory alignment because hardware can only load
    data into caches in one-go as a cache line of 64 bytes if they are aligned.
    Otherwise, we may need more than one load.
  - Most BLAS libraries are implemented using Fortran that uses column-major
    format to store ND-arrays.
  - Stride format is more general and we can use it to get any format including
    row-major order or column-major order.
    - Example: With 2d array A:
      - row-major: stride[0] = A.shape[1], stride[1] = 1
      - column-major: stride[0] = 1, stride[1] = A.shape[0]
    - Advantages:
      - We can have slices that use the same underlying memory but with
        different offsets to begin the slice as well as shape and
        stride.
      - The same thing can be used with `transpose` by swapping
        the strides and change the shape.
      - Finally, `broadcast` can be done by inserting a stride equal to 0.
    - Disadvantages:
      - Memory access becomes not contiguous which makes vectorization
        hard. Some linear algebra libraries require arrays to be compact.
  - Parallelization: We can use `OpenMP` to parallelize the operation by executing
    it using multiple threads on different cores in parallel.
    - Example: `#pragma use omp parallel for` before the loop.
  - CPU/GPU or any other device let us create flat array which is a
    consecutive slots of memory together. Therefore, we need shape, stride,
    and offset to create ND arrays view of flat array
  - All operations lead to creating new output array to store the results
    regardless of the backend device
  - Changing the device of an array involves copying the data from one device
    to the other. In needle, we are using numpy as a bridge. So we first
    create a numpy array of the input array and then create the final array
    on the intended device
  - With shape, stride, and offset: we can create different views of the same
    array that all share the same underlying memory. The output array is just
    a different NDArray data structure with the same underlying array.
  - When we do slicing, the resulting view may not be contiguous (compact).
    This means we can't pass the array handler because the operation will be
    applied to all the elements of the original array. Therefore, we need to
    first check if the input array is compact before doing the operation.
    - The array is compact if offset = 0 & strides correspond to the
      row-major order. Otherwise, we need to make it compact by creating
      new array with the sliced data (creating a copy with new memory
      slots).
    - Some frameworks such as Numpy and Pytorch allows users to do
      operations on non-contiguous arrays and they handle that in their
      implementations w/o the need to create new contiguous array. However,
      in some cases they may still force us to have contiguous array such
      as in matmul operations to have faster access to the elements and
      speed up operations.
- Training larger models: large datasets require large models which have large
  capacity. This will put pressure on compute devices and put them to the
  limit.
  - Shared memory on each core is typically 64KB.
  - Global memory of GPU becomes the bottleneck for large models because it
    typically has 10GB for most devices and we can't fit most larger models
    in 10GB.
  - Sources of memory consumption:
    - Weights of the model
    - Optimizer states such as momentum averages
    - Intermediate activation values (to build computation graph)
  - Techniques to save memory:
    - Inference: Use few buffers that will hold the activations of the
      layers. Every few layers, we reuse the same buffers we used in
      earlier computations. So instead of allocating N buffers, we can use
      2 or 3 buffers that will be reused in the forward computations.
    - Training: Because each activation is needed for the backward pass to
      compute the gradients, we can't just deallocate the buffers. As a
      workaround we can use checkpointing. This Technique involves
      checkpointing `K` layers/activations which is typically `sqrt(N)` and
      not store the whole N activations. During the backward pass, we
      recompute the values needed by running forward pass again for small
      segments at a time. This would make sublinear memory allocation and
      allows us to have bigger models at the expense of more computations
      due to segments forward pass done during gradient computation. We can
      pick activations that don't require heavy computations such as ReLU
      as non-stored activations as opposed to something like convolution
      which is computation-heavy.
  - Parallel and Distributed Training: We can leverage multiple GPU devices
    that are possibly distributed across several worker nodes to train the
    model. Parallelism can either be by model partitioning or data
    partitioning:
    - Model partitioning: break up the computation graph into multiple
      parts and pipeline the computations such that different worker nodes
      will be performing different parts of the computation at the same
      time and do the communication through send/recv at the boundaries of
      the graph.
    - Data partitioning: every worker runs the same replica of the model
      but using different micro batches and then the gradients can be
      summed from all workers since they are independent.
      - We can either use parameter server that receives all the gradients
        from all workers and then sum them up and perform the updates on
        the weights which then is sent to all workers so that they can be
        used in next micro batches. Each worker can start sending the
        gradients as soon as they are ready before waiting for all the
        gradients to be finished. This will increase the parallelism and
        allows workers to send the gradients while other gradient
        computation is being performed.
      - Or we can use allreduce() which sums up all gradients from all
        workers and send them back so each worker then can perform the
        updates
- GANs: The main goal is to generate images by the generator model that looks
  real and hard to tell they are fake, which means making the distribution of
  the generated images as close as possible to the distribution of the real
  images. This is done by training a discriminator and a generator which are trained
  through iterative process:
  - Generator's input is random vector and tries to generate image that looks
    real by maximizing the negative log-likelihood of the discriminator loss.
    This means make it harder for the discriminator to predict that it is
    fake and tries to bring the probability to 1.
  - Discriminator's input is both fake images generated by the generator and
    real images where fake images would have label = 0 and real images would
    have label = 1 and tries to minimize the negative log-likelihood. So the
    role of the discriminator is to guide the generator to generate better
    images and make it focus on what it takes to generate fake images that
    can't be differentiated from real images.
  - It is called _adversarial_ because the generator is able to learn subtle
    corner cases that looks the same for the human eye but are actually
    different for each distribution
  - Deconvolution or Conv2dTranspose is the opposite of convolution when we
    take a small vector and convert it to a much bigger space such as image
- Model Deployment:
  - Considerations:
    - Application env restrictions: model-size, no-python
    - Leverage existing HW accelerators such as mobile GPUs, NPUs,
      accelerated CPU instructions
    - Integrate with applications
- im2col: Do convolution in a form of matrix multiplication by first using
  strided operation to create the im2col matrix then reshape it to 2D tensor.
  Also, the weight which is 4D tensor would be reshaped to 2D tensor.
  - For batches of images with multiple channels where images are represented
    in the form of $N x H x W x C_{in}$ tensors and weights as $K x K x C_{in} x
  C_{out}$:
    - Create a new 6D tensor for images with dimensions: $N x H_new x W_new x K
    x K x C_{in}$
    - Reshape the batch to create im2col matrix with dimensions: $(N*H_new*W_new) x (K*K*C_{in})$
    - Reshape weight tensor to $(K*K*C_{in}) x C_{out}$
    - Do the matrix-matrix multiplication
    - Reshape the 2D output to new dimension: $N x H_new x W_new x C_{out}$
  - Cons: im2col matrix would lead to the creation of new matrix with new
    memory which is huge and inefficient since reshape would have to copy the
    data as it can't create the new matrix with existing strides
