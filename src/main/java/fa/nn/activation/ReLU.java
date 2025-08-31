package fa.nn.activation;

public class ReLU implements ActivationFunction {
    @Override
    public double activate(double z) {
        return Math.max(0, z);
    }

    @Override
    public double derivative(double z) {
        return z > 0 ? 1 : 0;
    }
}
