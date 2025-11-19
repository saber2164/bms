document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    const rulDisplay = document.getElementById('rul-display');
    const polyRulDisplay = document.getElementById('poly-rul-display');
    const ctx = document.getElementById('rulChart').getContext('2d');

    // State to store history of inputs (needed for sequence prediction)
    // We need 15 steps.
    let history = [];
    const SEQUENCE_LENGTH = 15;

    // Initialize Chart
    let chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Predicted RUL (DL)',
                data: [],
                borderColor: '#4CAF50',
                tension: 0.1
            }, {
                label: 'Predicted RUL (Poly)',
                data: [],
                borderColor: '#FF5722',
                tension: 0.1,
                borderDash: [5, 5]
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'RUL (Cycles)' }
                },
                x: {
                    title: { display: true, text: 'Prediction Step' }
                }
            }
        }
    });

    predictBtn.addEventListener('click', async () => {
        // Get inputs
        const soh = parseFloat(document.getElementById('soh').value);
        const temp = parseFloat(document.getElementById('temp').value);
        const re = parseFloat(document.getElementById('re').value);
        const rct = parseFloat(document.getElementById('rct').value);
        const icPeak = parseFloat(document.getElementById('ic_peak').value);
        const icVolt = parseFloat(document.getElementById('ic_volt').value);

        if (isNaN(soh) || isNaN(temp) || isNaN(re) || isNaN(rct) || isNaN(icPeak) || isNaN(icVolt)) {
            alert("Please enter valid numbers for all fields.");
            return;
        }

        // Add to history
        const currentFeatures = [soh, temp, re, rct, icPeak, icVolt];
        history.push(currentFeatures);

        // If history is too short, pad with the first value
        let inputSequence = [];
        if (history.length < SEQUENCE_LENGTH) {
            // Pad with the first available data point (or current if it's the only one)
            const paddingCount = SEQUENCE_LENGTH - history.length;
            for (let i = 0; i < paddingCount; i++) {
                inputSequence.push(history[0]);
            }
            inputSequence = inputSequence.concat(history);
        } else {
            // Take last 15
            inputSequence = history.slice(-SEQUENCE_LENGTH);
        }

        // Call API
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: inputSequence })
            });

            const data = await response.json();

            if (response.ok) {
                rulDisplay.innerText = `${data.rul.toFixed(1)} Cycles`;
                polyRulDisplay.innerText = data.poly_rul === "N/A" ? "N/A" : `${data.poly_rul.toFixed(1)} Cycles`;

                // Update Chart
                const step = chart.data.labels.length + 1;
                chart.data.labels.push(step);
                chart.data.datasets[0].data.push(data.rul);
                chart.data.datasets[1].data.push(data.poly_rul === "N/A" ? null : data.poly_rul);
                chart.update();
            } else {
                alert(`Error: ${data.error} `);
            }
        } catch (error) {
            console.error('Error:', error);
            statusDot.style.boxShadow = '0 0 8px #f87171';
            alert('Failed to get prediction: ' + error.message);
        }
    });

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    function updateChart(chart, currentSoH, predictedRUL) {
        // Generate a simple degradation curve from current SoH to 0.8 (EOL) over RUL cycles
        const cycles = [];
        const sohValues = [];
        const steps = 20;

        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const cycle = Math.round(t * predictedRUL);
            // Linear degradation assumption for visualization
            const soh = currentSoH - (t * (currentSoH - 0.8));

            cycles.push(cycle);
            sohValues.push(soh);
        }

        chart.data.labels = cycles;
        chart.data.datasets[0].data = sohValues;
        chart.update();
    }
});
