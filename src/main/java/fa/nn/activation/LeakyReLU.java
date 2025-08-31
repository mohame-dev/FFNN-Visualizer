package fa.nn.activation;

public class LeakyReLU implements ActivationFunction {
    private final double alpha = 0.01;

    public double activate(double z) {
        return z > 0 ? z : alpha * z;
    }

    public double derivative(double z) {
        return z > 0 ? 1 : alpha;
    }
}
