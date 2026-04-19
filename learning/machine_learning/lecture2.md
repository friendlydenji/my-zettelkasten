## 1. Linear regression
- Training set -> Learning algorithm -> Hypothesis
- X -> Hypothesis(h) -> Y
$$
\begin{aligned}
& h(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \dots \\
& h(x) = \sum_{i=0}^{n} \theta_i x_i \\
& where \space x_0 = 1 \\
& \mathbf{\theta} = 
\begin{bmatrix}
\theta_0 \\ \theta_1 \\ \vdots \\ \theta_n
\end{bmatrix}
\quad
\mathbf{x} =
\begin{bmatrix}
x_0 \\ x_1 \\ \vdots \\ x_n
\end{bmatrix}
\end{aligned}
$$
This is affine function
- $\theta$ is called "parameters" and the job of the learning algorithm is to choose parameters that allow us to make good predictions about outputs
- m is number of training examples
- x is "inputs" or features/attributes
- y is "outputs" or target variables
- (x,y) is a training example
- $(x^{(i)}, y^{(i)})$ is the training example i
- n is the number of the features
- $h(x) \approx y$ for the training examples \
Ordinary least squares is that we need to minimize the squared difference between the predictions and the real outputs
$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h(x^i) - y^i)^2 \\
minimize \space J(\theta)
$$
- $J(\theta)$ is called the cost function
## 2. Gradient descent
- Start with some $\theta$ ($\theta = \vec{0}$)
- Keep changing the $\theta$ to reduce J($\theta$)
$$
\theta_i := \theta_i - \alpha*\frac{\partial}{\partial\theta_i}(J(\theta))
$$
- $\frac{\partial}{\partial\theta_i}(J(\theta))$ is called partial derivative of the cost function J of $\theta$ with respect to the parameter $\theta_i$
- $\alpha$ is called learning rate
- Example with only one training example
$$
\begin{aligned}
& \frac{\partial}{\partial\theta_i}(J(\theta)) = \frac{\partial}{\partial\theta_i}(\frac{1}{2} (h(x) - y)^2) \\
& = 2 * \frac{1}{2} * (h(x) - y) * \frac{\partial}{\partial\theta_i}(h(x) - y) \\
& = (h(x) - y) * \frac{\partial}{\partial\theta_i}(\theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... - y) \\
& where \space \frac{\partial}{\partial\theta_i}(\theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_n x_n) = \frac{\partial}{\partial\theta_i}(\theta_i x_i) = x_i\\
& = (h(x) - y) * x_i
\end{aligned}
$$ 
So gradient descent algorithm is to repeat this until convergence
$$
\theta_i := \theta_i - \alpha*\sum_{j=1}^{m}(h(x^j) - y^j) * x_i^j \\
where \space i \space is \space from \space 0 \space to \space n 
$$
- Another name of this algorithm is Batch Gradient Descent
- The weakness of this algorithm is that if the batch (or dataset) is very large, the algorithm will slow because it need to iterate over entire training examples for every single step.
## 3. Stochastic gradient descent
- Instead of use entire batch for every single step, we only use one training example for each iteration
- This algorithm will never quite converge because every step the parameter is oscillated for difference training example so it will running around. However, it's faster for large dataset process
- Mini-batch gradient descent
## 4. Normal equation
$$\nabla_\theta J(\theta)=
\begin{bmatrix}
\frac{\partial J}{\partial\theta_0} \\ \frac{\partial J}{\partial\theta_1} \\ \vdots \\ \frac{\partial J}{\partial\theta_n}
\end{bmatrix}$$
- This is example about the partial derivative of a function f to matrix A
$$
\begin{aligned}
& A = 
\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix} \\
& if \space f(A) = A_{11}+A_{1n}^2 \\
& \nabla_A f(A) = 
\begin{bmatrix}
\frac{\partial f}{\partial A_{11}} & \frac{\partial f}{\partial A_{12}} \\
\frac{\partial f}{\partial A_{21}} & \frac{\partial f}{\partial A_{22}}
\end{bmatrix} \\
& = 
\begin{bmatrix}
1 & 2A_{1n} \\
0 & 0
\end{bmatrix} \\
\end{aligned}
$$
- If we have a squared matrix A (n*n), so the trace of A is equal to sum of the diagonal entries
$$
tr(A) = \sum_{i=1}^{n} A_{ii}
$$
- Trace of A equal to trace of A transpose.
  - If we have a fixed matrix B that $f(A) = tr(AB)$ then $\nabla_A f(A)=B^T$
  - $tr(AB) = tr(BA)$
  - $tr(ABC) = tr(CAB)$
  - $\nabla_A tr(AA^TC) = CA + C^TA$
- If we have the design matrix X and matrix y
$$
\begin{aligned}
& X\theta=
\begin{bmatrix}
-(x^1)^T- \\
\vdots \\
-(x^m)^T-
\end{bmatrix} *
\begin{bmatrix}
\theta_0 \\
\vdots \\
\theta_n
\end{bmatrix} 
=
\begin{bmatrix}
(x^1)^T \theta \\
\vdots \\
(x^m)^T \theta \\
\end{bmatrix} 
=
\begin{bmatrix}
h(x^1) \\
\vdots \\
h(x^m)
\end{bmatrix} \\
& y = 
\begin{bmatrix}
y^1 \\
\vdots \\
y^m
\end{bmatrix} \\
& \nabla_\theta J(\theta) = \nabla_\theta \frac{1}{2}(X\theta - y)^T(X\theta - y) \\
& = \frac{1}{2}\nabla_\theta (\theta^TX^T-y^T)(X\theta-y) \\
& = \frac{1}{2}\nabla_\theta(\theta^TX^TX\theta -\theta^TX^Ty - y^TX\theta + yy^T) \\
& = \frac{1}{2}(X^TX\theta+X^TX\theta-X^Ty-X^Ty) \\
& pre-set \space \nabla_\theta J(\theta) = 0 \\
& \Rightarrow X^TX\theta-X^Ty = \vec{0} \\
& \Rightarrow X^TX\theta = X^Ty \\
& \Rightarrow \theta = (X^TX)^{-1}X^Ty
\end{aligned}
$$
