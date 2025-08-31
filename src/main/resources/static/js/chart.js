let primary = "#4281A4";
let secondary = "#FD685D";

let chartInstance = null;
Chart.defaults.font.size = 18;

const ctx = document.getElementById("chart");

// plot the initial 'red' curve
plotExpression();

function plotChart(x, y) {
  if (chartInstance) {
    chartInstance.destroy();
  }

  const data = Array.from({ length: x.length }, (_, i) => ({
    x: x[i],
    y: y[i]
  }));

  chartInstance = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "f(x)",
          data: data,
          backgroundColor: primary,
        }
      ]
    },
    options: {
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: { display: true, text: "x" },
        },
        y: {
          title: { display: true, text: "y" },
        }
      },
    }
  });

  // lock the scale
  const ys = chartInstance.scales.y;
  chartInstance.options.scales.y.min = ys.min;
  chartInstance.options.scales.y.max = ys.max;
}

function update(x, y) {
  if (!chartInstance) return;

  const data = Array.from({ length: x.length }, (_, i) => ({
    x: x[i],
    y: y[i]
  }));

  // get the previous predicted curve
  let lineDs = chartInstance.data.datasets.find(ds => ds._isPrediction === true);

  if (!lineDs) {
    // create the predicted curve
    lineDs = {
      label: "predicted",
      type: "scatter",
      data: data,
      backgroundColor: secondary,
      _isPrediction: true
    };
    // push in front of the main curve, so this curve appears above it
    chartInstance.data.datasets.unshift(lineDs);
  } else {
    // update the predicted curve
    lineDs.data = data;
  }

  chartInstance.update();
}