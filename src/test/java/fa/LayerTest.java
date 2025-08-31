package fa;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import fa.nn.Layer;
import fa.nn.activation.ReLU;
import fa.nn.loss.MSE;

public class LayerTest {
    Layer l;

    @BeforeEach
    void setup() {
        this.l = new Layer(2, 2, new ReLU());

        double[][] w = { { 1.0, 0.5 }, { 2.0, 2.5 } };
        double[] b = { 0, 0 };
        l.set(w, b);
    }

    @Test
    public void constructor_invalidSize_ExceptionThrown() {
        assertThrows(IllegalArgumentException.class, () -> new Layer(-1, -1, new ReLU()));
        assertThrows(IllegalArgumentException.class, () -> new Layer(0, 0, new ReLU()));
    }

    @Test
    public void constructor_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> new Layer(1, 1, null));
        assertThrows(NullPointerException.class, () -> new Layer(1, 1, new ReLU(), null));
    }

    @Test
    public void constructor_validSize_correctShape() {
        Layer l = new Layer(2, 3, new ReLU());
        assertEquals(2, l.getInputSize());
        assertEquals(3, l.getOutputSize());

        double[][] w = l.getWeights();
        assertEquals(3, w.length);
        assertEquals(2, w[0].length);

        double[] b = l.getBiases();
        assertEquals(3, b.length);

        double[] a = l.getState();
        assertEquals(3, a.length);
    }

    @Test
    public void forward_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.l.forward(null));
    }

    @Test
    public void forward_invalidLength_ExceptionThrown() {
        assertThrows(IllegalArgumentException.class, () -> this.l.forward(new double[] { 0.0 }));
    }

    @Test
    public void forward_validParameters_ShapeExpected() {
        double[] x = new double[2];
        double[] yhat = this.l.forward(x);
        assertEquals(2, yhat.length);
    }

    @Test
    public void forward_validParameters_ComputesExpected() {
        double[] x = { 0.5, 1.0 };
        double[] yhat = this.l.forward(x);

        // z[1] = 1.0 * 0.5 + 0.5 * 1.0 = 1.0 => a(z[1]) = 1.0
        // z[2] = 2.0 * 0.5 + 2.5 * 1.0 = 3.5 => a(z[2]) = 3.5
        double[] expected = { 1.0, 3.5 };
        assertArrayEquals(expected, yhat);
    }

    @Test
    public void backwardOutput_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.l.backward(null, new MSE()));
    }

    @Test
    public void backwardOutput_invalidLength_ExceptionThrown() {
        double[] y = new double[1];
        assertThrows(IllegalArgumentException.class, () -> this.l.backward(y, new MSE()));
    }

    @Test
    public void backwardOutput_validParameters_ShapeExpected() {
        double[] x = new double[2];
        double[] y = new double[2];

        this.l.forward(x);
        double[] delta = this.l.backward(y, new MSE());

        assertEquals(2, delta.length);
    }

    @Test
    public void backwardOutput_validParameters_ComputesExpected() {
        double[] x = { 0.5, 1.0 };
        double[] y = { 1.0, 0.5 };

        // z[1] = a[1] = 1.0
        // z[2] = a[2] = 3.5
        this.l.forward(x);

        // δ[j] = dC/da[j] * da[j]/dz[j] = (ŷ - y) * (z[j] > 0 ? 1 : 0)
        // δ[1] = (1.0 - 1.0) * 1 = 0.0
        // δ[2] = (3.5 - 0.5) * 1 = 3.0
        double[] delta = this.l.backward(y, new MSE());
        assertArrayEquals(delta, new double[] { 0.0, 3.0 });
    }

    @Test
    public void backwardHidden_nullParameters_ExceptionThrown() {
        Layer nextLayer = new Layer(2, 1, new ReLU());
        assertThrows(NullPointerException.class, () -> this.l.backward(null, nextLayer));
    }

    @Test
    public void backwardHidden_invalidLength_ExceptionThrown() {
        Layer nextLayer = new Layer(2, 3, new ReLU());
        double[] nextDelta = new double[2];
        assertThrows(IllegalArgumentException.class, () -> this.l.backward(nextDelta, nextLayer));
    }

    @Test
    public void backwardHidden_layerMismatch_ExceptionThrown() {
        Layer nextLayer = new Layer(5, 3, new ReLU());
        double[] nextDelta = new double[3];
        assertThrows(IllegalArgumentException.class, () -> this.l.backward(nextDelta, nextLayer));
    }

    @Test
    public void backwardHidden_validParameters_ComputesExpected() {
        double[] x = { 0.5, 1.0 };
        double[] y = { 5 };

        Layer l1 = this.l;

        Layer l2 = new Layer(2, 1, new ReLU());
        double[][] w = { { 2.0, 1.0 } };
        double[] b = { 0 };
        l2.set(w, b);

        // z[1] = 2.0 * 1.0 + 1.0 * 3.5 = 5.5 => a(z[1]) = 5.5
        double[] yhat = l2.forward(l1.forward(x));
        assertArrayEquals(new double[] { 5.5 }, yhat);

        // δ[2][1] = (5.5 - 5.0) * 1 = 0.5
        double[] deltaL2 = l2.backward(y, new MSE());
        assertArrayEquals(new double[] { 0.5 }, deltaL2);

        // δ[L][j] = (Σ_k δ[L+1][k] * W[L+1][k][j]) * da[L][j]/dz[L][j]
        // = (Σ_k δ[L+1][k] * W[L+1][k][j]) * (z[j] > 0 ? 1 : 0)
        // δ[1][1] = (0.5 * 2.0) * 1 = 1.0
        // δ[1][2] = (0.5 * 1.0) * 1 = 0.5
        double[] deltaL1 = l1.backward(deltaL2, l2);
        assertArrayEquals(new double[] { 1.0, 0.5 }, deltaL1);
    }

    @Test
    public void updateDelta_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.l.update(0, new double[] {}, null));
    }

    @Test
    public void updateDelta_invalidLength_ExceptionThrown() {
        double[] data = new double[2];
        double[] dataInvalid = new double[3];

        assertThrows(IllegalArgumentException.class, () -> this.l.update(0, dataInvalid, data));
        assertThrows(IllegalArgumentException.class, () -> this.l.update(0, data, dataInvalid));
    }

    @Test
    public void updateDelta_validParameters_ComputesExpected() {
        double[] x = { 0.5, 1.0 };
        double[] y = { 1.0, 0.5 };

        // z[1] = a[1] = 1.0
        // z[2] = a[2] = 3.5
        this.l.forward(x);

        // δ[1] = (1.0 - 1.0) * 1 = 0.0
        // δ[2] = (3.5 - 0.5) * 1 = 3.0
        double[] delta = this.l.backward(y, new MSE());

        this.l.update(1.0, delta, x);

        // w[l][k][j] = w[l][k][j] - lr * dC/dw[l][k][j]
        // - where dC/dw[l][k][j] = delta[l][j] * a[l-1][k]
        // w[1][1] = 1.0 - 1.0 * 0.0 * 0.5 = 1.0
        // w[1][2] = 0.5 - 1.0 * 0.0 * 1.0 = 0.5
        // w[2][1] = 2.0 - 1.0 * 3.0 * 0.5 = 0.5
        // w[1][2] = 2.5 - 1.0 * 3.0 * 1.0 = -0.5
        double[][] w = this.l.getWeights();
        assertArrayEquals(new double[][] { { 1.0, 0.5 }, { 0.5, -0.5 } }, w);

        // b[l][j] = b[l][j] - lr * dC/db[l][j]
        // - where dC/db[l][j] = delta[l][j]
        // b[1] = 0.0 - 1.0 * 0.0 = 0.0
        // b[2] = 0.0 - 1.0 * 3.0 = -3.0
        double[] b = this.l.getBiases();
        assertArrayEquals(new double[] { 0, -3 }, b);

        // z[1] = 1.0 * 0.5 + 0.5 * 1.0 = 1.0 => a(z[1]) = 1.0
        // z[2] = 0.5 * 0.5 + -0.5 * 1.0 = -0.25 => a(z[2]) = 0
        double[] yhat = this.l.forward(x);
        assertArrayEquals(new double[] { 1.0, 0.0 }, yhat);
    }

    @Test
    public void updateGradient_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.l.update(0, new double[][] {}, null));
    }

    @Test
    public void updateGradient_validParameters_ComputesExpected() {
        double[] x = { 0.5, 1.0 };
        double[] y = { 1.0, 0.5 };

        // z[1] = a[1] = 1.0
        // z[2] = a[2] = 3.5
        this.l.forward(x);

        // δ[1] = (1.0 - 1.0) * 1 = 0.0
        // δ[2] = (3.5 - 0.5) * 1 = 3.0
        double[] delta = this.l.backward(y, new MSE());

        // dC/dw[l][k][j] = delta[l][j] * a[l-1][k]
        // {0.0, 0.0}
        // {1.5, 3.0}
        double[][] gW = {
                { delta[0] * x[0], delta[0] * x[1] },
                { delta[1] * x[0], delta[1] * x[1] }
        };
        double[] gB = { delta[0], delta[1] };

        this.l.update(1.0, gW, gB);

        // w[1][1] = 1.0 - 1.0 * 0.0 * 0.5 = 1.0
        // w[1][2] = 0.5 - 1.0 * 0.0 * 1.0 = 0.5
        // w[2][1] = 2.0 - 1.0 * 3.0 * 0.5 = 0.5
        // w[1][2] = 2.5 - 1.0 * 3.0 * 1.0 = -0.5
        double[][] w = this.l.getWeights();
        assertArrayEquals(new double[][] { { 1.0, 0.5 }, { 0.5, -0.5 } }, w);

        // b[1] = 0.0 - 1.0 * 0.0 = 0.0
        // b[2] = 0.0 - 1.0 * 3.0 = -3.0
        double[] b = this.l.getBiases();
        assertArrayEquals(new double[] { 0, -3 }, b);

        // z[1] = 1.0 * 0.5 + 0.5 * 1.0 + 0 = 1.0 => a(z[1]) = 1.0
        // z[2] = 0.5 * 0.5 + -0.5 * 1.0 + -3 = -3.25 => a(z[2]) = 0
        double[] yhat = this.l.forward(x);
        assertArrayEquals(new double[] { 1.0, 0.0 }, yhat);
    }

    @Test
    public void set_nullParameters_ExceptionThrown() {
        double[][] w = new double[2][2];
        double[] b = new double[2];

        assertThrows(NullPointerException.class, () -> this.l.set(null, b));
        assertThrows(NullPointerException.class, () -> this.l.set(w, null));
    }

    @Test
    public void set_invalidLength_ExceptionThrown() {
        double[][] w = new double[2][2];
        double[][] weightsOutputInvalid = { { 0.1, 0.2 }, { 0.3, 0.4 }, { 0.5, 0.6 } };
        double[][] weightsInputInvalid = { { 0.1, 0.2 }, { 0.3, 0.4, 0.5 } };

        double[] b = new double[2];
        double[] biasesInvalid = { 0.0 };

        assertThrows(IllegalArgumentException.class, () -> this.l.set(weightsOutputInvalid, b));
        assertThrows(IllegalArgumentException.class, () -> this.l.set(weightsInputInvalid, b));
        assertThrows(IllegalArgumentException.class, () -> this.l.set(w, biasesInvalid));
    }
}
