const expressionInp = document.querySelector('input.equation');
const dropdownBtn = document.querySelector('button.advanced');
const advanced = document.querySelector('.form-advanced');
const saveBtn = document.querySelector('button.save');
const learnBtn = document.querySelector('button.learn');

let xmin = document.querySelector('.xmin').value;
let xmax = document.querySelector('.xmax').value;
let npts = document.querySelector('.npts').value;
let epochs = document.querySelector('.epochs').value
let interval = document.querySelector('.interval').value

// expression handling
expressionInp.addEventListener('input', plotExpression);

async function plotExpression() {
  let expression = expressionInp.value || expressionInp.placeholder;

  const { valid, x, y } = await validateExpression(expression, xmin, xmax, npts);
  expressionInp.classList.toggle('invalid', !valid);

  plotChart(x, y);
}

async function validateExpression(expr, xmin, xmax, npts) {
  try {
    const res = await fetch("/validate", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expression: expr,
        xmin: xmin,
        xmax: xmax,
        npoints: npts,
        epochs: epochs,
        interval: interval
      })
    });

    if (!res.ok) {
      return { valid: false, x: [], y: [] };
    }

    return await res.json();
  } catch (err) {
    console.error("Validate failed:", err);
    return { valid: false, x: [], y: [] };
  }
}

// advanced settings handling
dropdownBtn.addEventListener('click', () => {
  dropdownBtn.classList.toggle('open');
  advanced.classList.toggle('open');
})

saveBtn.addEventListener('click', () => {
  xmin = document.querySelector('.xmin').value;
  xmax = document.querySelector('.xmax').value;
  npts = document.querySelector('.npts').value;
  epochs = document.querySelector('.epochs').value;
  interval = document.querySelector('.interval').value;

  setStatus({ maxEpoch: epochs })
  plotExpression();
})

// learning-stream handling; click again = cancel previous and start fresh.
let evtSource = null;

function stopStream() {
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
}

function startStream() {
  stopStream();

  evtSource = new EventSource("/stream-sse");

  evtSource.addEventListener('epoch', () => {
    const { x, y, epoch, loss, valLoss } = JSON.parse(event.data);

    update(x, y)
    setStatus({ epoch: epoch, loss: loss, valLoss: valLoss })
  });

  evtSource.addEventListener('done', () => {
    stopStream();
  });

  evtSource.onerror = (e) => {
    console.error('SSE error:', e);
    stopStream();
  };
}

learnBtn.addEventListener('click', startStream);

// set status
function setStatus({ epoch, maxEpoch, loss, valLoss }) {
  if (epoch != null) {
    const el = document.getElementById('epoch');
    el.textContent = parseInt(epoch, 10);
  }
  if (maxEpoch != null) {
    const el = document.getElementById('maxEpoch');
    el.textContent = parseInt(maxEpoch, 10);
  }
  if (loss != null) {
    const el = document.getElementById('loss');
    el.textContent = (+loss).toFixed(6);
  }
  if (valLoss != null) {
    const el = document.getElementById('valLoss');
    el.textContent = (+valLoss).toFixed(6);
  }
}

// sets initial status
setStatus({ epoch: 0, maxEpoch: epochs, loss: 0, valLoss: 0 })