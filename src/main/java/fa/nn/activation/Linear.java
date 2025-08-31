package fa.nn.activation;

public class Linear implements ActivationFunction {
    @Override
    public double activate(double z) {
        return z;
    }

    @Override
    public double derivative(double z) {
        return 1;
    }
}
